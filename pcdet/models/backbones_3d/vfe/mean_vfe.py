import torch

from .vfe_template import VFETemplate
#总体来说就是相当于把一个 voxel中的特征进行 加合 然后 再除以里面点的数量
#获得每一个体素内所有点的特征的平均值
#这个应该是最简单的 vef
#而且不需要额外的参数输入

class MeanVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):   #模型配置 点云特征数量（一个点云所具有的特征数量 例如x，y，z，反射强度 这就是4）  和  其他可能参数
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features
    
    #获取输出特征维度
    def get_output_feature_dim(self):   
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']   #体素特征  每个体素中点的个数
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)  #选择所有体素 所有点 所有特征  对第二维度进行求和 就是对于每一个voxel中的点特征进行求和
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)  #计算一个用于标准化的值  列向量  最小值为1.0
        points_mean = points_mean / normalizer   #总体来说就是相当于把一个 voxel中的特征进行 加合 然后 再除以里面点的数量，获得每一个体素内所有点的特征的平均值
        batch_dict['voxel_features'] = points_mean.contiguous()  #张量储存方式为连续

        return batch_dict
