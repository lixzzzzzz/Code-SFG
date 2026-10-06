import torch
from torch import nn


class ConvFuser(nn.Module):
    def __init__(self,model_cfg) -> None:
        super().__init__()
        #读取config
        self.model_cfg = model_cfg 
        #定义输入输出通道数
        in_channel = self.model_cfg.IN_CHANNEL  
        out_channel = self.model_cfg.OUT_CHANNEL
        #定义就卷积模块
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
            )
        
    def forward(self,batch_dict):
        """
        Args:
            batch_dict:
                spatial_features_img (tensor): Bev features from image modality
                spatial_features (tensor): Bev features from lidar modality

        Returns:
            batch_dict:
                spatial_features (tensor): Bev features after muli-modal fusion
        """
        #图像 和 lidar 对应bev特征
        img_bev = batch_dict['spatial_features_img']
        lidar_bev = batch_dict['spatial_features']
        #将两bev特征进行拼接 按行进行拼接
        cat_bev = torch.cat([img_bev,lidar_bev],dim=1)
        #卷积层对拼接后的特征进行融合
        mm_bev = self.conv(cat_bev)
        #融合结果输出
        batch_dict['spatial_features'] = mm_bev
        return batch_dict

class ConvFuser_vod(nn.Module):
    def __init__(self,model_cfg) -> None:
        super().__init__()
        #读取config
        self.model_cfg = model_cfg 
        #定义输入输出通道数
        in_channel = self.model_cfg.IN_CHANNEL  
        mid_channel = self.model_cfg.MID_CHANNEL 
        out_channel = self.model_cfg.OUT_CHANNEL
        #定义就卷积模块
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(True)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
            )
        
    def forward(self,batch_dict):
        """
        Args:
            batch_dict:
                spatial_features_img (tensor): Bev features from image modality
                spatial_features (tensor): Bev features from lidar modality

        Returns:
            batch_dict:
                spatial_features (tensor): Bev features after muli-modal fusion
        """
        #图像 和 lidar 对应bev特征
        img_bev = batch_dict['spatial_features_img']
        lidar_bev = batch_dict['spatial_features']
        #将两bev特征进行拼接 按行进行拼接
        cat_bev = torch.cat([img_bev,lidar_bev],dim=1)
        #卷积层对拼接后的特征进行融合
        cat_bev = self.conv1(cat_bev)
        mm_bev = self.conv2(cat_bev)
        #融合结果输出
        batch_dict['spatial_features'] = mm_bev
        return batch_dict

class ConvFuser_DUAL(nn.Module):
    def __init__(self,model_cfg) -> None:
        super().__init__()
        #读取config
        self.model_cfg = model_cfg 
        #定义输入输出通道数
        in_channel = self.model_cfg.IN_CHANNEL  
        mid_channel = self.model_cfg.MID_CHANNEL 
        out_channel = self.model_cfg.OUT_CHANNEL
        #定义就卷积模块
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(True)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
            )
        
    def forward(self,batch_dict):
        """
        Args:
            batch_dict:
                spatial_features_img (tensor): Bev features from image modality
                spatial_features (tensor): Bev features from lidar modality

        Returns:
            batch_dict:
                spatial_features (tensor): Bev features after muli-modal fusion
        """
        #图像 和 lidar 对应bev特征
        img_bev = batch_dict['spatial_features_img']
        lidar_bev = batch_dict['spatial_features']
        surface_bev = batch_dict['surface_spatial_features']
        #将两bev特征进行拼接 按行进行拼接
        cat_bev = torch.cat([img_bev,lidar_bev,surface_bev],dim=1)
        #卷积层对拼接后的特征进行融合
        cat_bev = self.conv1(cat_bev)
        mm_bev = self.conv2(cat_bev)
        #融合结果输出
        batch_dict['spatial_features'] = mm_bev
        return batch_dict
