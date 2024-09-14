
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
#from einops import rearrange, repeat
from pointnet2_ops import pointnet2_utils
from pykeops.torch import LazyTensor

def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    elif activation.lower() == 'leakyrelu0.2':
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    else:
        return nn.ReLU(inplace=True)

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


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
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx

def keops_knn(nsample,xyz,new_xyz):
    xyz = LazyTensor(xyz.unsqueeze(2))
    new_xyz = LazyTensor(new_xyz.unsqueeze(1))
    distance = ((xyz-new_xyz)**2).sum(dim=-1)
    idx = distance.argKmin(nsample,dim=1)
    return idx

def grouping_knn(xyz,x,k): #in: B,C,N: out, B,k,C,N
    B,C,N = x.shape
    idx = keops_knn(k,xyz,xyz)
    x = index_points(x.transpose(1,2),idx.reshape(B,-1)).view(B,N,k,C).permute(0,2,3,1)
    return x

class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)


class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D, self).__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(in_channels=channel, out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)

class LaplacianUnit(nn.Module):
    
    def __init__(self,channel,k):
        super().__init__()
        self.conv = nn.Conv1d(channel,channel,1)
        self.bn = nn.BatchNorm1d(channel)
        self.k = k
    
    def forward(self,xyz,x):
        dx = grouping_knn(xyz,x,self.k).sum(dim=1)-x
        x = x+self.bn(F.relu(self.conv(dx)))
        return x
    
class GroupMLP(nn.Module):
    
    def __init__(self,channel,k):
        super().__init__()
        self.k = k
        self.conv = ConvBNReLU1D(channel,channel,bias=False)
    
    def forward(self,xyz,x):
        B,C,N = x.shape
        x = grouping_knn(xyz,x,self.k).transpose(1,2) #out: B,C,k,N
        x = self.conv(x.reshape(B,C,self.k*N)).view(B,C,self.k,N)
        x = x.max(dim=2)[0]
        return x
    
class LU_MLP(nn.Module):
    
    def __init__(self,channel,k,dim_factor=0.5):
        super().__init__()
        middle = int(channel*dim_factor)
        self.mlp1 = ConvBNReLU1D(channel,middle,bias=False)
        self.group_mlp = GroupMLP(middle,16)
        self.lu = LaplacianUnit(middle,16)
        self.mlp2 = ConvBNReLU1D(middle,channel,bias=False)
        
    def forward(self,xyz,x):
        skip_x = x
        x = self.mlp1(x)
        x = self.group_mlp(xyz,x)
        x = self.lu(xyz,x)
        x = self.mlp2(x)
        x = x+skip_x
        return x
    
class DownSampler(nn.Module):
    
    def __init__(self,out_points,in_channels,out_channels):
        super().__init__()
        self.out_points = out_points
        self.conv = nn.Conv1d(in_channels,out_channels,1)
    
    def forward(self,xyz,x):
        idx = pointnet2_utils.furthest_point_sample(xyz,self.out_points)
        new_xyz = index_points(xyz,idx)
        new_x = self.conv(index_points(x.transpose(1,2),idx).transpose(1,2))
        return new_xyz,new_x
    
class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu'):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):  # [b, d, g]
        return self.operation(x)
    
class UpsampleBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpsampleBlock, self).__init__()
        self.fuse = ConvBNReLU1D(in_channel, out_channel, 1)
        self.lu = LaplacianUnit(out_channel,16)


    def forward(self, xyz1, xyz2, points1, points2):
        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.topk(3,dim=-1)  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        new_points = self.fuse(new_points)
        new_points = self.lu(xyz1,new_points)
        return new_points
    
    
class DSNet(nn.Module):
    
    def __init__(self,num_classes=3,points=16000,embedding_channel=150,blocks=[2,6,2],down_factor=[2,2,2],up_blocks=[1,1,1]):
        super().__init__()
        channels = embedding_channel
        self.embedding = nn.Conv1d(3,channels,1)
        self.LU_MLP0 = nn.ModuleList([LU_MLP(embedding_channel,32),LU_MLP(embedding_channel,32)])
        self.LU_MLPs = nn.ModuleList()
        self.DownSamplers = nn.ModuleList()
        self.UpSamplers = nn.ModuleList()
        
        channel_list = [channels]
        point_list = [points]
        for i in range(len(blocks)):
            points = points//2
            point_list.append(points)
            self.DownSamplers.append(DownSampler(points,channels,channels*2))
            channels = channels*2
            channel_list.append(channels)
            self.LU_MLP_stage = nn.ModuleList()
            for i in range(blocks[i]):
                self.LU_MLP_stage.append(LU_MLP(channels,32))
            self.LU_MLPs.append(self.LU_MLP_stage)
        
        for i in range(len(blocks)):
            self.UpSamplers.append(UpsampleBlock(channel_list[i*(-1)-1]+channel_list[i*(-1)-2],channel_list[i*(-1)-2]))
            
        self.classifier = nn.Sequential(
            nn.Conv1d(embedding_channel, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(128, 3, 1)
        )
    
    def forward(self,x):
        xyz = x.transpose(1,2)
        
        x = self.embedding(x)
        #initial LU MLP
        for lu_module in self.LU_MLP0:
            x = lu_module(xyz,x)
            
        x_list = []
        xyz_list = []
        for i in range(len(self.LU_MLPs)):
            x_list.append(x)
            xyz_list.append(xyz)
            xyz,x = self.DownSamplers[i](xyz,x)
            for lu_module in self.LU_MLPs[i]:
                x = lu_module(xyz,x)
        
        for i in range(len(self.UpSamplers)):
            new_xyz = xyz_list[i*(-1)-1]
            new_x = x_list[i*(-1)-1]
            x = self.UpSamplers[i](new_xyz,xyz,new_x,x)
            xyz = new_xyz
        
        x = F.log_softmax(self.classifier(x).transpose(1,2),dim=-1)
        return x
    
        
class get_loss(nn.Module): #4th class for padded points 
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, weight=torch.tensor([1.0,1.3,6.4,0]).cuda()): #[0.5,1.4,3.43]
        B,_ = pred.shape
        pred = torch.concat((pred,torch.zeros(B,1).to(pred.device)),dim=-1)
        total_loss = F.nll_loss(pred, target, weight=weight)
        return total_loss


if __name__ == '__main__':
    data = torch.rand(2, 3, 2048)
    norm = torch.rand(2, 3, 2048)
    cls_label = torch.rand([2, 16])
    print("===> testing modelD ...")
    model = pointMLP(50)
    out = model(data, cls_label)  # [2,2048,50]
    print(out.shape)
