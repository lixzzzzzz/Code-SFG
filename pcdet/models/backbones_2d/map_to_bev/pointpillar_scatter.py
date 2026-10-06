import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg    #读取模型config
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES  #确定bev特征个数
        self.nx, self.ny, self.nz = grid_size  #确定网格尺寸
        assert self.nz == 1 #检查self.nz是否为1

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']  #获取 点云pillar特征 和 对应坐标
        batch_spatial_features = []  #初始化空列表
        batch_size = coords[:, 0].max().int().item() + 1 #计算batch size的大小 找到coords[:, 0]第一列中（批次索引）的最大值 然后加1 就是 batchsize
        for batch_idx in range(batch_size):  #针对每一个 batchsize
            spatial_feature = torch.zeros(  #创建全零张量
                self.num_bev_features,   #bev特征的数量
                self.nz * self.nx * self.ny,   #空间特征总数量
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx  #筛选当前批次特征对应的掩码
            this_coords = coords[batch_mask, :]  #根据掩码确定当前批次的点云坐标
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]  #根据点云坐标计算索引，用于将pillar特征散布到三维空间
            indices = indices.type(torch.long)  #类型转换
            pillars = pillar_features[batch_mask, :]  #筛选当前批次的 pillar特征
            pillars = pillars.t()  #转置pillar特征，使得特征维度位于第一维
            spatial_feature[:, indices] = pillars  #将pillar特征重新散步到空间对应索引位置
            batch_spatial_features.append(spatial_feature)   #将按照空间位置排序后的 pillar特征加入batch_spatial_features
        #上面的forward部分都是为了排序，让没有3d空间位置的pillar特征重新按照 坐标的标号排列
        batch_spatial_features = torch.stack(batch_spatial_features, 0)  #将张量进行堆叠
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)  #重新改变特征张量的形状  这个就是向bev投影的过程
        batch_dict['spatial_features'] = batch_spatial_features  #输出投影后的特征张量
        # print('batch_spatial_features',batch_spatial_features.shape)
        return batch_dict


class PointPillarScatter3d(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()
        
        self.model_cfg = model_cfg
        self.nx, self.ny, self.nz = self.model_cfg.INPUT_SHAPE
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_bev_features_before_compression = self.model_cfg.NUM_BEV_FEATURES // self.nz

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features_before_compression,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] * self.ny * self.nx + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features_before_compression * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict

class PointPillarScatter_DUAL(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg    #读取模型config
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES  #确定bev特征个数
        self.nx, self.ny, self.nz = grid_size  #确定网格尺寸
        assert self.nz == 1 #检查self.nz是否为1

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['surface_pillar_features'], batch_dict['surface_voxel_coords']  #获取 点云pillar特征 和 对应坐标
        batch_spatial_features = []  #初始化空列表
        batch_size = coords[:, 0].max().int().item() + 1 #计算batch size的大小 找到coords[:, 0]第一列中（批次索引）的最大值 然后加1 就是 batchsize
        for batch_idx in range(batch_size):  #针对每一个 batchsize
            spatial_feature = torch.zeros(  #创建全零张量
                self.num_bev_features,   #bev特征的数量
                self.nz * self.nx * self.ny,   #空间特征总数量
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            batch_mask = coords[:, 0] == batch_idx  #筛选当前批次特征对应的掩码

            this_coords = coords[batch_mask, :]  #根据掩码确定当前批次的点云坐标
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]  #根据点云坐标计算索引，用于将pillar特征散布到三维空间
            indices = indices.type(torch.long)  #类型转换
            pillars = pillar_features[batch_mask, :]  #筛选当前批次的 pillar特征
            pillars = pillars.t()  #转置pillar特征，使得特征维度位于第一维
            spatial_feature[:, indices] = pillars  #将pillar特征重新散步到空间对应索引位置
            batch_spatial_features.append(spatial_feature)   #将按照空间位置排序后的 pillar特征加入batch_spatial_features
        #上面的forward部分都是为了排序，让没有3d空间位置的pillar特征重新按照 坐标的标号排列
        batch_spatial_features = torch.stack(batch_spatial_features, 0)  #将张量进行堆叠
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)  #重新改变特征张量的形状  这个就是向bev投影的过程
        batch_dict['surface_spatial_features'] = batch_spatial_features  #输出投影后的特征张量
        return batch_dict