 #last dim is 6 with laplacian
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as func
# from torch import einsum
# from einops import rearrange, repeat


from pointnet2_ops import pointnet2_utils

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

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def bknn(x, k):
    inner = 2*torch.bmm(x.transpose(2,1), x)
    xx = torch.sum(torch.square(x), dim=1, keepdim = True)
    pairwise_dist = xx - inner + xx.transpose(2, 1)
    inv_dist = (-1)*pairwise_dist
    
    idx = inv_dist.topk(k=k,dim=-1)
    
    return idx[1]

def addToIDX(tensor1,idx,tensor2):
    device = tensor1.device
    B = tensor1.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    #print("INDEXING:",tensor1.shape,idx.shape,tensor1[batch_indices, idx].shape,tensor2.shape)
    for i in range(B):
        #print("SIDX",tensor2[i])
        tensor1[i].put_(idx[i],tensor2[i],accumulate=True)
        
    return tensor1

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

class SurfaceConv(nn.Module):
    
    def getGroup(self,xyz, k, feat, idx=None):
                
        B,F,N = feat.shape
        K = self.k
        
        if (idx==None):
            kNeighbors = bknn(xyz[:,:,0:3].transpose(1,2),k)
            idx = kNeighbors
        
        group = index_points(feat.transpose(1,2),idx.reshape(B,-1))
        group = group.view(B,N,K,F).transpose(1,2)
        return group,idx
        
    def __init__(self,channels,k,M,dim_expand=0.5):
        super().__init__()
                
        self.k = k
        self.M = M
        self.g_func = nn.Linear(channels+3,int(channels*dim_expand),bias=False)
        self.h_func = nn.Linear(M*int(dim_expand*channels),channels,bias=False)
        self.bn0 = nn.BatchNorm1d(channels)
        self.bn1 = nn.BatchNorm1d(M*int(dim_expand*channels))
        self.bn2 = nn.BatchNorm1d(channels)
        
    def forward(self,xyz,feat):
        
        B, C, N = xyz.shape
        _, F, _ = feat.shape
        K = self.k
        
        feat = self.bn0(feat)
        
        feat0 = self.g_func(torch.concat((feat,xyz.transpose(1,2)),dim=1).transpose(1,2)).transpose(1,2)
        idx = None
        all_feat = []
        for i in range(self.M):
            feat,idx = self.getGroup(xyz,self.k,feat0,idx=idx)
            feat = feat.max(dim=1)[0].transpose(1,2)
            all_feat.append(feat-feat0)
        #print(torch.concat((feat1-feat0,feat2-feat0,feat3-feat0),dim=1).shape,3*int(0.25*self.n_feat))
        relative_feat = torch.concat(all_feat,dim=1)
        feat = self.h_func(relative_feat.transpose(1,2)).transpose(1,2)
        
        return feat
        
class SurfaceBlock(nn.Module):
    
    def __init__(self,channels,k,m,mlp_expand):
        super().__init__()
        self.surfaceconv = SurfaceConv(channels,k,m)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.mlp = nn.Sequential(nn.Conv1d(channels,channels*mlp_expand,1),
                                 nn.ReLU(),
                                 nn.Conv1d(channels*mlp_expand,channels,1))
        
    def forward(self,xyz,x):
        x = self.surfaceconv(xyz,self.norm1(x))+x
        x = self.mlp(self.norm2(x))+x
        return xyz,x
    
class SetAbstraction(nn.Module):
    
    def __init__(self,channels,k,radius,stride=4): #channels should be a list
        super().__init__()
        self.radius = radius
        self.stride = stride
        self.query_group = pointnet2_utils.QueryAndGroup(radius,k,use_xyz=True)
        self.mlp = nn.Sequential(nn.Conv1d(3+channels[0],channels[1],1),
                                 nn.ReLU(),
                                 nn.Conv1d(channels[1],channels[2],1))
        self.norm = nn.BatchNorm1d(channels[-1])
    
    def forward(self,xyz,x):
        fps_idx = pointnet2_utils.furthest_point_sample(xyz,xyz.shape[1]//self.stride)
        # norm_xyz = xyz/self.radius
        # norm_new_xyz = new_xyz/self.radius
        new_xyz = pointnet2_utils.gather_operation(xyz.transpose(1,2).contiguous(),fps_idx).transpose(1,2).contiguous()
        group_x = self.query_group(xyz,new_xyz,x.contiguous())
        B,C,N,K = group_x.shape
        group_x = self.mlp(group_x.view(B,C,N*K))
        x = group_x.view(B,-1,N,K).max(dim=-1)[0]
        x = self.norm(x)
        return new_xyz,x
    
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

    
class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, out_channel, blocks=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(PointNetFeaturePropagation, self).__init__()
        self.in_channel = in_channel
        self.fuse = ConvBNReLU1D(in_channel, out_channel, 1, bias=bias)
        self.extraction = PosExtraction(out_channel, blocks, groups=groups,
                                        res_expansion=res_expansion, bias=bias, activation=activation)


    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: sampled input points position data, [B, S, 3]
            points1: input points data, [B, D', N]
            points2: input points data, [B, D'', S]
        Return:
            new_points: upsampled points data, [B, D''', N]
        """

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        dists = square_distance(xyz1, xyz2)
        dists, idx = dists.sort(dim=-1)
        dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

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
        new_points = self.extraction(new_points)
        return new_points

class SurfaceNet2(nn.Module):
    
    def __init__(self,num_class=3,in_channels=3,blocks=[2,2,2],block_channels=[64,128,256],down=[4,4,4],surface_k=8,
                 surface_M=4,radius=[0.1,0.4,1.6,6.4],k=32,
                 decoder_channels=[256,128,128],decoder_blocks=[1,1,1]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.save = []
        channels = in_channels
        for i in range(len(blocks)):
            self.encoder.append(SetAbstraction([channels,block_channels[i],block_channels[i]],k,radius[i],stride=down[i]))
            channels = block_channels[i]
            for j in range(blocks[i]):
                self.save.append(False)
                self.encoder.append(SurfaceBlock(channels,surface_k,surface_M,mlp_expand=4))
            self.save.append(True)
            
        enc_channels = [in_channels]+block_channels[:-1] #skip input to decoder, length N
        dec_channels = [block_channels[-1]]+decoder_channels #direct input to decoder, length N+1
            
        self.decoder = nn.ModuleList() 
        for i in range(len(blocks)):
            self.decoder.append(PointNetFeaturePropagation(dec_channels[i]+enc_channels[-(i+1)], dec_channels[i+1],
                                           blocks=decoder_blocks[i]))
                
        self.classifier = nn.Sequential(nn.BatchNorm1d(decoder_channels[-1]),
                                        nn.Dropout(0.5),
                                        nn.Linear(decoder_channels[-1],num_class))
                
    def forward(self,x):
        xyz = x.permute(0, 2, 1)
        B,_,N = x.shape
                
        u_list = [[xyz,x]]
        for i,layer in enumerate(self.encoder):
            xyz,x = layer(xyz,x)
            if (self.save[i]):
                u_list.append([xyz,x])
                
        u_list.reverse()
        for i,layer in enumerate(self.decoder):
            x = layer(u_list[i+1][0],u_list[i][0],u_list[i+1][1],x)
            
        x = self.classifier(x.transpose(1,2).flatten(0,1)).unflatten(0,(B,N))
        x = F.log_softmax(x, dim=-1)
        return x