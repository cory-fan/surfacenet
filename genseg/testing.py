from dataloader import SegData
import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import provider
import importlib
import shutil
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import torch.nn as nn
import time

import models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_count = 3

makeLog = False

def handle_state_dict(state_dict):
    out = dict()
    for key in state_dict:
        out[key.replace("module.","")] = state_dict[key]
    return out
    
def euclidean(tensor1, tensor2):
    selfdots1 = torch.tensor([torch.dot(i,i) for i in tensor1]).cuda()
    selfdots2 = torch.tensor([torch.dot(i,i) for i in tensor2]).cuda()
    #rint(tensor1,tensor2)
    mm = torch.matmul(tensor1,tensor2.transpose(0,1))
    sd1 = (selfdots1*torch.ones((mm.shape[1],mm.shape[0])).cuda()).transpose(0,1)
    sd2 = (selfdots2*torch.ones((mm.shape[0],mm.shape[1])).cuda())
    #print(sd1.device,sd2.device)
    return (sd1+sd2-2*mm)

def interpolate(pointset,target,npoints):
    fullpoints = []
    oset = (torch.zeros(pointset.shape[1],device=pointset.device)+1.0)
    for i,points in enumerate(pointset):
        #print(np.logical_not((np.abs(points[:,0])==oset)*(np.abs(points[:,1])==oset)*(np.abs(points[:,2])==oset)))
        cpoints = points[torch.logical_not((torch.abs(points[:,0])==oset)*(torch.abs(points[:,1])==oset)*(torch.abs(points[:,2])==oset))].cuda()
        x = torch.rand((npoints-len(cpoints)))*2.0-1.0
        y = torch.rand((npoints-len(cpoints)))*2.0-1.0
        xy = torch.stack([x,y]).to(cpoints.device)
        #print(xy.device,cpoints.device)
        #print(xy.device,cpoints.device)
        dist = euclidean(xy.transpose(0,1),cpoints[:,0:2].float())
        ret = torch.topk(dist,10)
        val = ret[0]
        idx = ret[1]

        newz = []
        newtargets = []
        for j,i in enumerate(val):
            #print(i.shape,idx.shape)
            base = torch.sum(i**(-1))
            top = torch.sum((i**(-1)*cpoints[idx[j],2]))
            z = top/base
            #print(base,top,cpoints[idx[j],2])\
            #print(z)
            newtargets.append(target[idx[0]])
            newz.append(z)

        newpoints = torch.stack([x,y,torch.tensor(newz,device=x.device)]).to(cpoints.device).transpose(0,1)
        #print(newpoints.shape)
        augpoints = torch.concat([cpoints,newpoints])
        fullpoints.append(augpoints)
    return torch.stack(fullpoints)

def truncate(points,target):
    B,N,C = points.shape
    cond = torch.max(torch.abs(torch.tensor(points)),dim=-1)[0]<1
    points = torch.where(cond.unsqueeze(2),torch.tensor(points),torch.ones((B,N,C)))
    targets = torch.where(cond,torch.tensor(target),torch.zeros((B,N)))
            
    return points,target
        

def augdata(points,target,hard=True):
    points = points.cpu().numpy()
    if (hard):
        points[:, :, 0:3] = provider.jitter_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3], shift_range=0.4)
        points,target,_ = provider.shuffle_data(points,target)
        points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3], shift_range=0.2)
        points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
    #points = interpolate(points,target,600)
    points = torch.tensor(points).float()
    #points = interpolate(points,720)
    return points,target

def compute_overall_iou(pred, target, num_classes):
    shape_ious = []
    pred = pred.max(dim=2)[1]    # (batch_size, num_points)  the pred_class_idx of each point in each sample
    pred_np = pred.cpu().data.numpy()

    target_np = target.cpu().data.numpy()
    for shape_idx in range(pred.size(0)):   # sample_idx
        part_ious = []
        for part in range(num_classes):   # class_idx! no matter which category, only consider all part_classes of all categories, check all 50 classes
            # for target, each point has a class no matter which category owns this point! also 50 classes!!!
            # only return 1 when both belongs to this class, which means correct:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            # always return 1 when either is belongs to this class:
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))

            F = np.sum(target_np[shape_idx] == part)

            if F != 0:
                iou = I / float(U)    #  iou across all points for this class
                part_ious.append(iou)   #  append the iou of this class
        shape_ious.append(np.mean(part_ious))   # each time append an average iou across all classes of this sample (sample_level!)
    return shape_ious   # [batch_size]

def parse_args():
    """ PARAMETERS """
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--model', default='SurfaceNet', help='model name [default: POINTNET3_MSG_CLS]')
    parser.add_argument('--ngpu', type=int, default=[0, 1, 2], help='specify how many gpus to train the model [default: 0]')
    parser.add_argument('--num_point', type=int, default=32000, help='Point Number [default: 1024]')
    parser.add_argument('--num_votes', type=int, default=1, help='Aggregate classification scores with voting [default: 3]')
    parser.add_argument('--test_path', default='./data/haiti/test', help='path of the test data')
    parser.add_argument('--checkpoint', default='log/classification/2024-08-30_12-21_SurfaceNet_haiti/checkpoints/best_model.pth', 
                        help='path of the checkpoint')
    parser.add_argument("--worker", type=int, default=torch.get_num_threads())
    return parser.parse_args()

def cal_iou(pred,target):
    class_iou = [[],[],[]]
    pred = pred.max(dim=2)[1]
    for i in range(pred.shape[0]):
        for c in range(3):
            U = np.logical_or(pred[i].cpu()==c,target[i].cpu()==c).sum()
            if (U>0):
                I = np.logical_and(pred[i].cpu()==c,target[i].cpu()==c).sum()
                iou = I/U
                class_iou[c].append(iou)
    return class_iou

def test(model, loader, num_class=class_count, vote_num=1):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    accMat = np.zeros((num_class,2,2))
    
    class_iou = [[],[],[]]
    
    model.eval()
    
    for points, target in tqdm(loader):
        points,target = augdata(points,target,hard=False)
        points = points.transpose(2, 1)
        points, target = points.float().to(device), target.to(device)
        classifier = model.eval()
        pred = classifier(points)
        
        new_class_iou = cal_iou(pred,target)
        class_iou = [class_iou[i]+new_class_iou[i] for i in range(3)]
        
        pred_choice = pred.max(dim=2)[1]
        pred_choice = pred_choice.cpu()
        
        target = target.cpu()
        
        for cat in torch.unique(target):
            c = int(cat.item())
            if (c==3):
                continue
            classacc = pred_choice[target == cat].eq(target[target == cat]).cpu().sum()
            classacc = classacc.cpu()
            class_acc[c, 0] += classacc.item() / (target==c).float().sum()
            class_acc[c, 1] += 1
            
        correct = pred_choice.eq(target.cpu()).cpu().sum()
        batch_num_samples = points.size()[0]
        mean_correct.append(correct.item() / (batch_num_samples*8000-(target.cpu()==3).sum()))
        
    #print(class_iou[0])
    class_iou = [np.mean(i) for i in class_iou]
    print(class_iou)
    miou = np.mean(class_iou)
    #print("mIOU:",miou)
    print("C:",class_acc[:,0],class_acc[:,1])
    class_acc[:, 2] = np.nan_to_num(class_acc[:, 0] / class_acc[:, 1])
    
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc, miou


def main(args):
    
    """ DATA LOADING """
    TEST_DATASET = SegData(args.test_path, npoint=args.num_point,test_mode=True)
    testDataLoader = DataLoader(TEST_DATASET, batch_size=1, shuffle=False, num_workers=14)
    print('num_val_files: ' + str(len(TEST_DATASET.filepaths)))

   
    """ MODEL LOADING """
    num_class = class_count
    #shutil.copy('./models/%s.py' % args.model, str(experiment_dir))
    #shutil.copy('./models/pointnet_util.py', str(experiment_dir))
    #train on the single gpu
    #classifier = MODEL.get_model(num_class).to(device)
    #train on the multiple gpu
    
    MODEL = importlib.import_module("pointMLP")
    UTIL = importlib.import_module("util")
    
    classifier = models.__dict__[args.model]().cuda()
    
    checkpoint = torch.load(args.checkpoint)
    classifier.load_state_dict(handle_state_dict(checkpoint['model_state_dict']))
    
    instance_acc,class_acc,miou = test(classifier,testDataLoader)
    print("INSTANCE ACC:",instance_acc)
    print("CLASS ACC:",class_acc)
    print("Mean IOU:",miou)

if __name__ == '__main__':
    args = parse_args()
    main(args)
