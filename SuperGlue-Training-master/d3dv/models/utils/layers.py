import torch
from torch import nn


@torch.no_grad()
def knn(x, k):
    inner = -2*torch.einsum('bin,bim->bnm', x, x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    return pairwise_distance.topk(k=k, dim=-1).indices  # B x N x k


def gather_graph_feature(x, idx):
    _, D, N = x.shape
    _, _, K = idx.shape
    idx = idx.unsqueeze(1).expand(-1, D, -1, -1)
    x_gather = x.unsqueeze(2).expand(-1, -1, N, -1)
    feat = x_gather.gather(-1, idx)
    x = x.unsqueeze(-1).expand(-1, -1, -1, K)
    feat = torch.cat([feat - x, x], 1)
    return feat


class kNNPropagation(nn.Module):
    def __init__(self, num_dim):
        super().__init__()
        self.mlp = MLP([num_dim*2, num_dim], is_2d=True, act_out=True)

    def forward(self, x, idx):
        feat = gather_graph_feature(x, idx)
        message = self.mlp(feat).max(-1).values
        return x + message


class DGCNN(nn.Module):
    def __init__(self, out_dim, k):
        super().__init__()
        self.k = k
        in_dim = 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim*2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_dim), nn.LeakyReLU(negative_slope=0.2))

    def forward(self, pts):
        graph = knn(pts, self.k)

        x = gather_graph_feature(pts, graph)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False).values

        x = gather_graph_feature(x1, graph)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False).values

        x = gather_graph_feature(x2, graph)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False).values

        x = gather_graph_feature(x3, graph)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False).values

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        return x
