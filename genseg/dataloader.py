import numpy as np
import glob
import torch.utils.data
import torch
from pointnet2_ops import pointnet2_utils
import os

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask].float()
        farthest = torch.max(distance, -1)[1]
    return centroids

class SegData(torch.utils.data.Dataset):
    def __init__(self, root_dir, scale_aug=False, rot_aug=False, npoint=8000, test_mode=False):

        self.filepaths = []
        
        self.npoints = npoint
        self.test_mode = test_mode
        all_files = sorted(glob.glob(root_dir + '/*.txt'))
        #print(root_dir+'/*.xyz')
        self.filepaths.extend(all_files)

    def __len__(self):
        # print(len(self.filepaths))
        return len(self.filepaths)
    
    def pad(self,item):  
        
        #print(item)
        
        if (len(item)>self.npoints):
            if (self.test_mode):
                idx = farthest_point_sample(torch.tensor(np.array([item.astype(float)])),self.npoints).numpy()[0]
            else:
                idx = torch.randperm(len(item))[:self.npoints]
            item = item[idx]
            return item
        item = item.tolist()

        for i in range(len(item),self.npoints):
            item.append([-1,-1,-1,0])
        
        return np.array(item)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        if (self.test_mode):
            if (os.path.exists(self.filepaths[idx].replace(".txt","")+"_proccessed.npy")):
                allInfo = np.load(self.filepaths[idx].replace(".txt","")+"_proccessed.npy")
            else:
                allInfo = np.loadtxt(self.filepaths[idx])
                allInfo = self.pad(allInfo)
                np.save(self.filepaths[idx].replace(".txt","")+"_proccessed.npy",allInfo)
        else:
            allInfo = np.loadtxt(self.filepaths[idx])
            allInfo = self.pad(allInfo)
        point_set = allInfo[:,0:3]
        class_id = allInfo[:,3]
        #print(point_set.shape, path)
        return (point_set ,class_id)
