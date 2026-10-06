import torch
import torch.nn as nn
import torch.nn.functional as F
from ...model_utils.basic_block_2d import BasicBlock2D


class GeneralizedLSSFPN(nn.Module):
    """
        This module implements FPN, which creates pyramid features built on top of some input feature maps.
        This code is adapted from https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/necks/fpn.py with minimal modifications.
    """
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        in_channels =  self.model_cfg.IN_CHANNELS   #输入通道数
        out_channels = self.model_cfg.OUT_CHANNELS  #输出通道数
        num_ins = len(in_channels)                  #确定一共几个输入通道   bevfusion中  IN_CHANNELS: [192, 384, 768]
        num_outs = self.model_cfg.NUM_OUTS          #确定一共几个输出通道   bevfusion中  NUM_OUTS: 3
        start_level = self.model_cfg.START_LEVEL    #确定FPN的起始级别     bevfusion中  START_LEVEL: 0
        end_level = self.model_cfg.END_LEVEL        #确定FPN的结束级别     bevfusion中  END_LEVEL: -1

        self.in_channels = in_channels

        #如果end_level为-1  则将最后一个输入通道作为 结束level
        if end_level == -1:   
            self.backbone_end_level = num_ins - 1
        #如果指定了end_level的具体数值  则采用具体数值  但是前提是  输入通道个数  结束级别  起始级别  输出通道个数 之间的关系需要合理
        else:
            self.backbone_end_level = end_level  
            assert end_level <= len(in_channels)   # end_level必须小于 输入通道数的个数
            assert num_outs == end_level - start_level  #输出通道数等于结束级别减去起始级别的差值，以确保输出通道数的合法性
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = nn.ModuleList()    #初始化 Lateral连接的卷积层
        self.fpn_convs = nn.ModuleList()        #初始化 FPN连接的卷积层
        #对 lateral_convs 和 fpn_convs网络进行构造
        #lateral_convs是1✖️1的卷积核  重点是为了修改通道数，将不同层的不同输入通道数（除了当前级别是最后一个级别外，其输入通道数都是in_channels[i] + out_channels） 在输出时都统一改成OUT_CHANNELS大小
        #fpn_convs是3✖️3的卷积核  目的是为了对当前层 和 当前层上一层拼接后的特征进行进一步特征提取  输入和输出通道数都是 OUT_CHANNELS
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = BasicBlock2D(
                in_channels[i] + (in_channels[i + 1] if i == self.backbone_end_level - 1 else out_channels),
                out_channels, kernel_size=1, bias = False
            )
            fpn_conv = BasicBlock2D(out_channels,out_channels, kernel_size=3, padding=1, bias = False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                image_features (list[tensor]): Multi-stage features from image backbone.
        Returns:
            batch_dict:
                image_fpn (list(tensor)): FPN features.
        """
        # upsample -> cat -> conv1x1 -> conv3x3
        inputs = batch_dict['image_features']
        assert len(inputs) == len(self.in_channels)

        # build laterals
        #向lateral中赋值，传入的是backbone中提取的不同层级的 image_feature
        laterals = [inputs[i + self.start_level] for i in range(len(inputs))]

        # build top-down path
        used_backbone_levels = len(laterals) - 1
        for i in range(used_backbone_levels - 1, -1, -1):
            #采用双线性插值，将上一层的特征图 上采样到 当前层的大小 
            x = F.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[2:],
                mode='bilinear', align_corners=False,
            )
            laterals[i] = torch.cat([laterals[i], x], dim=1)   #将当前层的特征与 上采样的结果进行拼接
            laterals[i] = self.lateral_convs[i](laterals[i])   #采用lateral_convs（1✖️1卷积） 对拼接后的特征进行特征提取
            laterals[i] = self.fpn_convs[i](laterals[i])       #采用 fpn_convs（3✖️3卷积） 对特征进行进一步提取

        # build outputs
        outs = [laterals[i] for i in range(used_backbone_levels)]
        batch_dict['image_fpn'] = tuple(outs)
        return batch_dict
