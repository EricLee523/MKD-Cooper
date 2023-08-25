
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple

def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x

class CollaborativeAttentionFusion(nn.Module):
    def __init__(self, feature_dim):
        super(CollaborativeAttentionFusion, self).__init__()
        self.pixel_weight_layer = PixelWeightLayer(feature_dim)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if feature_dim == 64:
            self.fcScale = 4
        elif feature_dim == 128:
            self.fcScale = 8
        elif feature_dim == 256:
            self.fcScale = 16
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // self.fcScale, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // self.fcScale, feature_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, record_len, pairwise_t_matrix):
        ########## FUSION START ##########
        # we concat ego's feature with other agent
        # first transform feature to ego's coordinate
        split_x = regroup(x, record_len)

        B = pairwise_t_matrix.shape[0]
        _, C, H, W = x.shape

        out = []

        for b in range(B):
            # number of valid agent
            N = record_len[b]
            # (N,N,4,4)
            # t_matrix[i, j]-> from i to j
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]

            # update each node i
            i = 0 # ego
            # (N, C, H, W) neighbor_feature is agent i's neighborhood warping to agent i's perspective
            # Notice we put i one the first dim of t_matrix. Different from original.
            # t_matrix[i,j] = Tji
            neighbor_feature = warp_affine_simple(split_x[b],
                                            t_matrix[i, :, :, :],
                                            (H, W))

            # (N, C, H, W)
            ego_feature = split_x[b][0].view(1, C, H, W).expand(N, -1, -1, -1)
            # (N, 2C, H, W)
            neighbor_feature_cat = torch.cat((neighbor_feature, ego_feature), dim=1)
            # -------------------------------------SpatialAttention-----------------------------------
            # (N, 1, H, W)
            spatial_feature = self.pixel_weight_layer(neighbor_feature_cat)
            # (N, 1, H, W)
            spatial_feature = F.softmax(spatial_feature, dim=0)
            spatial_feature = spatial_feature.expand(-1, C, -1, -1)
            # -------------------------------------ChannelAttention-----------------------------------
            channel_weight = self.avg_pool(ego_feature).view(-1, C)
            channel_weight = self.fc(channel_weight).view(-1, C, 1, 1)
            CAF_attention_feature = spatial_feature * channel_weight.expand_as(spatial_feature)

            # (N, C, H, W)
            feature_fused = torch.sum(CAF_attention_feature * neighbor_feature, dim=0)
            out.append(feature_fused)

        return torch.stack(out)



class PixelWeightLayer(nn.Module):
    def __init__(self, channel):
        super(PixelWeightLayer, self).__init__()

        self.conv1_1 = nn.Conv2d(channel * 2, 128, kernel_size=1, stride=1, padding=0)
        self.bn1_1 = nn.BatchNorm2d(128)

        self.conv1_2 = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)
        self.bn1_2 = nn.BatchNorm2d(32)

        self.conv1_3 = nn.Conv2d(32, 8, kernel_size=1, stride=1, padding=0)
        self.bn1_3 = nn.BatchNorm2d(8)

        self.conv1_4 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)
        # self.bn1_4 = nn.BatchNorm2d(1)

    def forward(self, x):
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))
        x_1 = F.relu(self.bn1_3(self.conv1_3(x_1)))
        x_1 = F.relu(self.conv1_4(x_1))

        return x_1
