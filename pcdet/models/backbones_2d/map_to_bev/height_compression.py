import torch.nn as nn


class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        #在bevfusion算法中   此处接收来自 上一模块BACKBONE_3D经过六层卷积生成的 encoded_spconv_tensor
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()  #将稀疏张量转化为密集张量
        N, C, D, H, W = spatial_features.shape  #读取密集张量的形状信息    N 表示批次大小，C 表示通道数，D、H、W 分别表示高度、深度和宽度
        spatial_features = spatial_features.view(N, C * D, H, W)   #将特征的维数进行调整  将通道数 与 高度进行合并，完成bev投影处理
        batch_dict['spatial_features'] = spatial_features   #将完成 MAP_TO_BEV 后的特征进行存储
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']   #由于在 MAP_TO_BEV环节并没有产生stride，所以此处采用与上一步骤相同的值
        return batch_dict
