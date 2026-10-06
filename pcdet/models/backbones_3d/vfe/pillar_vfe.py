import torch
import torch.nn as nn
import torch.nn.functional as F

from .vfe_template import VFETemplate

# 点云特征网络
# linear   batchnorm(选择性有)   relu    最大值输出
class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer  #是否为最后一层
        self.use_norm = use_norm    #使用归一化
        if not self.last_vfe:       #如果不是最后层  则通道数减半，因为在最后一层之前，输出特征都会被拆分成两半，一半向下传递，一半用于连接到输入特征
            out_channels = out_channels // 2  #地板除 结果取整数 7//2=3 10//3=3 -7//3=-3
        
        #决定是否使用归一化
        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000  #设置了一个阈值，当输入数据的批次大于该阈值时，将输入数据分成多个部分进行处理，以避免超出内存限制

    def forward(self, inputs):
        if inputs.shape[0] > self.part:  #如果输入数据较大
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part  #看把输入分成几部分
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])  #分几部分进行linear计算
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)

        torch.backends.cudnn.enabled = False    #禁用CuDNN 以确保 Batch Normalization 的计算正确性
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x  #归一化处理
        torch.backends.cudnn.enabled = True  #启用CuDNN 
        x = F.relu(x)   #relu函数
        x_max = torch.max(x, dim=1, keepdim=True)[0] #确定x中第二维度的最大值  并且通过 【0】 取出元组的第一个元素 即每一行的最大值

        if self.last_vfe:
            return x_max   #如果是最后一层就输出每一行特征的最大值
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)  #如果不是最后一层，就将每一行的最大值与 x 进行拼接
            x_concatenated = torch.cat([x, x_repeat], dim=2) #在第三维进行拼接
            return x_concatenated
class PFNLayer_DUAL(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer  #是否为最后一层
        self.use_norm = use_norm    #使用归一化
        if not self.last_vfe:       #如果不是最后层  则通道数减半，因为在最后一层之前，输出特征都会被拆分成两半，一半向下传递，一半用于连接到输入特征
            out_channels = out_channels // 2  #地板除 结果取整数 7//2=3 10//3=3 -7//3=-3
        
        #决定是否使用归一化
        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000  #设置了一个阈值，当输入数据的批次大于该阈值时，将输入数据分成多个部分进行处理，以避免超出内存限制

    def forward(self, inputs):
        if inputs.shape[0] > self.part:  #如果输入数据较大
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part  #看把输入分成几部分
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])  #分几部分进行linear计算
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)

        torch.backends.cudnn.enabled = False    #禁用CuDNN 以确保 Batch Normalization 的计算正确性
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x  #归一化处理
        torch.backends.cudnn.enabled = True  #启用CuDNN 
        x = F.relu(x)   #relu函数
        x_max = torch.max(x, dim=1, keepdim=True)[0] #确定x中第二维度的最大值  并且通过 【0】 取出元组的第一个元素 即每一行的最大值

        if self.last_vfe:
            return x_max   #如果是最后一层就输出每一行特征的最大值
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)  #如果不是最后一层，就将每一行的最大值与 x 进行拼接
            x_concatenated = torch.cat([x, x_repeat], dim=2) #在第三维进行拼接
            return x_concatenated

class PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        
        #读取 config中的设置  对该模块进行调整   一共三个值 在config中都有声明
        self.use_norm = self.model_cfg.USE_NORM   #是否进行归一化
        self.with_distance = self.model_cfg.WITH_DISTANCE  #是否计算距离
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ  #是否使用绝对坐标
        num_point_features += 6 if self.use_absolute_xyz else 3  #使用绝对坐标的话 就是加6维 不使用就加3维
        if self.with_distance:
            num_point_features += 1   #计算距离的话加一维

        self.num_filters = self.model_cfg.NUM_FILTERS   #指定特征提取网络中每个卷积层 滤波器的数量
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)  #相当于把输入层的滤波器数量（num_point_features特征维度） 和 config中设置的其他层的滤波器数量（特征维度）给拼起来
        
        #初始化 point feature network
        pfn_layers = []  
        for i in range(len(num_filters) - 1):  #输入 、、、、 输出  所以需要num_filters个数减一
            in_filters = num_filters[i]         #输入滤波器数量
            out_filters = num_filters[i + 1]    #输出滤波器数量
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        #初始化点云特征提取网络 pfn_layers
        self.pfn_layers = nn.ModuleList(pfn_layers)
        
        #提取 单个voxel 在 x y z上的大小
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        #在 点云范围 x y z 的起始位置上 加上单个 voxel大小的一半
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]  
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]
   
    #用于获取输出特征的维度，该方法返回了PFN网络中最后一层的输出特征维度
    def get_output_feature_dim(self):
        return self.num_filters[-1]
    #确定哪些位置是进行了特征填充的  返回一个布尔张量
    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        #包含体素特征  聚类特征（与voxel平均中心点坐标差距） 中心特征（与voxel坐标中心点坐标差距） 距离特征（每个点云到坐标原点欧式距离）
        #然后输入到 pfn（point feature network）中进行特征提取

        #获取 体素特征 体素中点的数量 体素的坐标
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)   #此句代码的作用跟 MEAN_VFE类似  对于每一个voxel中的点云坐标进行平均（先sum，然后除以voxel中点云的个数），获取一个voxel中所有点云坐标的平均值
        f_cluster = voxel_features[:, :, :3] - points_mean  #聚类特征，即 一个voxel中 每一点 与这个voxel所有点平均坐标之间的差距
        
        #计算各体素中 每一个点 与 体素中心点之间的坐标差距
        f_center = torch.zeros_like(voxel_features[:, :, :3])   #初始化一个 与 voxel_features[:, :, :3]大小相同的 tensor
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)  #确定每一个点 与 点所在voxel中心点 之间的x坐标差距  计算方法（voxel_features[:, :, 0] 【点云的x坐标】 - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1)【点云所在voxel在空间中的坐标编号】 * self.voxel_x 【单个voxel在x轴方向上的大小】+ self.x_offset【x轴上第一个voxel的中心点的x坐标】) ）
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)
        
        #如果使用 use_absolute_xyz  那么voxel feature要多6个维度  提取体素特征中所有的内容
        #如果不使用use_absolute_xyz  那么voxel feature要多3个维度  提取体素特征中前三列内容，只包括点云的xyz坐标
        #包括f_cluster聚类特征（一个voxel中 每一点 与这个voxel所有点平均坐标之间的差距）
        #以及f_center 中心特征（一个voxel中 每一点 与这个voxel坐标中心点之间的差距）
        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        #如果 model_cfg.WITH_DISTANCE 为 TRUE   则计算 每个点到 坐标原点的欧式距离
        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)  #计算 点云特征中 每一个点 相较于 坐标原点的 欧氏距离
            features.append(points_dist)
        #将 列表转成 tensor
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1] #确定体素个数
        #体素过滤  仅保留有效体素
        #确定有效体素 生成掩码
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features) #修改mask的 大小 和 数据类型
        features *= mask  #将有效体素掩码与体素特征相乘   过滤掉无用的体素特征
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()  #压缩特征张量的维度，去除大小为1的维度
        batch_dict['pillar_features'] = features
        return batch_dict
    
# Vod
# 与常规的 PillarVFE基本相同  只是多了几个维度的数据
class Radar7PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range):
        super().__init__(model_cfg=model_cfg)

        num_point_features = 0
        self.use_norm = self.model_cfg.USE_NORM  # whether to use batchnorm in the PFNLayer
        self.use_xyz = self.model_cfg.USE_XYZ
        self.with_distance = self.model_cfg.USE_DISTANCE
        self.selected_indexes = []

        ## check if config has the correct params, if not, throw exception
        radar_config_params = ["USE_RCS", "USE_VR", "USE_VR_COMP", "USE_TIME", "USE_ELEVATION"]
        
        #需要检查 vod雷达中特有的属性是否在 config中已经声明需要
        if all(hasattr(self.model_cfg, attr) for attr in radar_config_params):
            self.use_RCS = self.model_cfg.USE_RCS
            self.use_vr = self.model_cfg.USE_VR
            self.use_vr_comp = self.model_cfg.USE_VR_COMP
            self.use_time = self.model_cfg.USE_TIME
            self.use_elevation = self.model_cfg.USE_ELEVATION

        else:
            raise Exception("config does not have the right parameters, please use a radar config")

        #可以radar的数据特征
        self.available_features = ['x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time']

        num_point_features += 6  # center_x, center_y, center_z, mean_x, mean_y, mean_z, time, we need 6 new

        self.x_ind = self.available_features.index('x')
        self.y_ind = self.available_features.index('y')
        self.z_ind = self.available_features.index('z')
        self.rcs_ind = self.available_features.index('rcs')
        self.vr_ind = self.available_features.index('v_r')
        self.vr_comp_ind = self.available_features.index('v_r_comp')
        self.time_ind = self.available_features.index('time')

        if self.use_xyz:  # if x y z coordinates are used, add 3 channels and save the indexes
            num_point_features += 3  # x, y, z
            self.selected_indexes.extend((self.x_ind, self.y_ind, self.z_ind))  # adding x y z channels to the indexes

        if self.use_RCS:  # add 1 if RCS is used and save the indexes
            num_point_features += 1
            self.selected_indexes.append(self.rcs_ind)  # adding  RCS channels to the indexes

        if self.use_vr:  # add 1 if vr is used and save the indexes. Note, we use compensated vr!
            num_point_features += 1
            self.selected_indexes.append(self.vr_ind)  # adding  v_r_comp channels to the indexes

        if self.use_vr_comp:  # add 1 if vr is used (as proxy for sensor cue) and save the indexes
            num_point_features += 1
            self.selected_indexes.append(self.vr_comp_ind)

        if self.use_time:  # add 1 if time is used and save the indexes
            num_point_features += 1
            self.selected_indexes.append(self.time_ind)  # adding  time channel to the indexes

        ### LOGGING USED FEATURES ###
        print("number of point features used: " + str(num_point_features))
        print("6 of these are 2 * (x y z)  coordinates realtive to mean and center of pillars")
        print(str(len(self.selected_indexes)) + " are selected original features: ")

        for k in self.selected_indexes:
            print(str(k) + ": " + self.available_features[k])

        self.selected_indexes = torch.LongTensor(self.selected_indexes)  # turning used indexes into Tensor

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)
        
        #point feature network特征提取网络初始化
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        ## saving size of the voxel
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]

        ## saving offsets, start of point cloud in x, y, z + half a voxel, e.g. in y it starts around -39 m
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]
    
    #获取输出维度
    def get_output_feature_dim(self):
        return self.num_filters[-1]  # number of outputs in last output channel

    #获取补充体素掩码
    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        ## coordinate system notes
        # x is pointing forward, y is left right, z is up down
        # spconv returns voxel_coords as  [batch_idx, z_idx, y_idx, x_idx], that is why coords is indexed backwards

        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict[
            'voxel_coords']

        if not self.use_elevation:  # if we ignore elevation (z) and v_z
            voxel_features[:, :, self.z_ind] = 0  # set z to zero before doing anything

        orig_xyz = voxel_features[:, :, :self.z_ind + 1]  # selecting x y z

        # calculate mean of points in pillars for x y z and save the offset from the mean
        # Note: they do not take the mean directly, as each pillar is filled up with 0-s. Instead, they sum and divide by num of points
        points_mean = orig_xyz.sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = orig_xyz - points_mean  # offset from cluster mean

        # calculate center for each pillar and save points' offset from the center. voxel_coordinate * voxel size + offset should be the center of pillar (coords are indexed backwards)
        f_center = torch.zeros_like(orig_xyz)
        f_center[:, :, 0] = voxel_features[:, :, self.x_ind] - (
                    coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, self.y_ind] - (
                    coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, self.z_ind] - (
                    coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        voxel_features = voxel_features[:, :, self.selected_indexes]  # filtering for used features

        features = [voxel_features, f_cluster, f_center]

        if self.with_distance:  # if with_distance is true, include range to the points as well
            points_dist = torch.norm(orig_xyz, 2, 2, keepdim=True)  # first 2: L2 norm second 2: along 2. dim
            features.append(points_dist)

        ## finishing up the feature extraction with correct shape and masking
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask

        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict

# TJ4DRadar
class Radar8PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range):
        super().__init__(model_cfg=model_cfg)

        num_point_features = 0
        self.use_norm = self.model_cfg.USE_NORM  # whether to use batchnorm in the PFNLayer
        self.use_xyz = self.model_cfg.USE_XYZ
        self.with_distance = self.model_cfg.USE_DISTANCE
        self.selected_indexes = []

        ## check if config has the correct params, if not, throw exception
        # radar_config_params = ["USE_RCS", "USE_VR", "USE_VR_COMP", "USE_TIME", "USE_ELEVATION"]
        radar_config_params = ["USE_VR", "USE_RANGE", "USE_POWER", "USE_ALPHA", "USE_BETA", "USE_ELEVATION"]

        #需要检查 vod雷达中特有的属性是否在 config中已经声明需要
        if all(hasattr(self.model_cfg, attr) for attr in radar_config_params):
            # self.use_RCS = self.model_cfg.USE_RCS
            # self.use_vr = self.model_cfg.USE_VR
            # self.use_vr_comp = self.model_cfg.USE_VR_COMP
            # self.use_time = self.model_cfg.USE_TIME
            # self.use_elevation = self.model_cfg.USE_ELEVATION

            self.use_vr = self.model_cfg.USE_VR
            self.use_range = self.model_cfg.USE_RANGE
            self.use_power = self.model_cfg.USE_POWER
            self.use_alpha = self.model_cfg.USE_ALPHA
            self.use_beta = self.model_cfg.USE_BETA
            self.use_elevation = self.model_cfg.USE_ELEVATION

        else:
            raise Exception("config does not have the right parameters, please use a radar config")

        #可以radar的数据特征
        # self.available_features = ['x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time']
        self.available_features = ['x', 'y', 'z', 'v_r', 'range', 'power', 'alpha', 'beta']

        num_point_features += 6  # center_x, center_y, center_z, mean_x, mean_y, mean_z, time, we need 6 new

        # self.x_ind = self.available_features.index('x')
        # self.y_ind = self.available_features.index('y')
        # self.z_ind = self.available_features.index('z')
        # self.rcs_ind = self.available_features.index('rcs')
        # self.vr_ind = self.available_features.index('v_r')
        # self.vr_comp_ind = self.available_features.index('v_r_comp')
        # self.time_ind = self.available_features.index('time')

        self.x_ind = self.available_features.index('x')
        self.y_ind = self.available_features.index('y')
        self.z_ind = self.available_features.index('z')
        self.vr_ind = self.available_features.index('v_r')
        self.range_ind = self.available_features.index('range')
        self.power_ind = self.available_features.index('power')
        self.alpha_ind = self.available_features.index('alpha')
        self.beta_ind = self.available_features.index('beta')

        # if self.use_xyz:  # if x y z coordinates are used, add 3 channels and save the indexes
        #     num_point_features += 3  # x, y, z
        #     self.selected_indexes.extend((self.x_ind, self.y_ind, self.z_ind))  # adding x y z channels to the indexes

        # if self.use_RCS:  # add 1 if RCS is used and save the indexes
        #     num_point_features += 1
        #     self.selected_indexes.append(self.rcs_ind)  # adding  RCS channels to the indexes

        # if self.use_vr:  # add 1 if vr is used and save the indexes. Note, we use compensated vr!
        #     num_point_features += 1
        #     self.selected_indexes.append(self.vr_ind)  # adding  v_r_comp channels to the indexes

        # if self.use_vr_comp:  # add 1 if vr is used (as proxy for sensor cue) and save the indexes
        #     num_point_features += 1
        #     self.selected_indexes.append(self.vr_comp_ind)

        # if self.use_time:  # add 1 if time is used and save the indexes
        #     num_point_features += 1
        #     self.selected_indexes.append(self.time_ind)  # adding  time channel to the indexes

        if self.use_xyz:  # if x y z coordinates are used, add 3 channels and save the indexes
            num_point_features += 3  # x, y, z
            self.selected_indexes.extend((self.x_ind, self.y_ind, self.z_ind))  # adding x y z channels to the indexes

        if self.use_vr:  # add 1 if vr is used and save the indexes. Note, we use compensated vr!
            num_point_features += 1
            self.selected_indexes.append(self.vr_ind)  # adding  v_r_comp channels to the indexes

        if self.use_range:  
            num_point_features += 1
            self.selected_indexes.append(self.range_ind)

        if self.use_power:  
            num_point_features += 1
            self.selected_indexes.append(self.power_ind)
        
        if self.use_alpha:  
            num_point_features += 1
            self.selected_indexes.append(self.alpha_ind)

        if self.use_beta:  
            num_point_features += 1
            self.selected_indexes.append(self.beta_ind)

        ### LOGGING USED FEATURES ###
        print("number of point features used: " + str(num_point_features))
        print("6 of these are 2 * (x y z)  coordinates realtive to mean and center of pillars")
        print(str(len(self.selected_indexes)) + " are selected original features: ")

        for k in self.selected_indexes:
            print(str(k) + ": " + self.available_features[k])

        self.selected_indexes = torch.LongTensor(self.selected_indexes)  # turning used indexes into Tensor

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)
        
        #point feature network特征提取网络初始化
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        ## saving size of the voxel
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]

        ## saving offsets, start of point cloud in x, y, z + half a voxel, e.g. in y it starts around -39 m
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]
    
    #获取输出维度
    def get_output_feature_dim(self):
        return self.num_filters[-1]  # number of outputs in last output channel

    #获取补充体素掩码
    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        ## coordinate system notes
        # x is pointing forward, y is left right, z is up down
        # spconv returns voxel_coords as  [batch_idx, z_idx, y_idx, x_idx], that is why coords is indexed backwards

        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict[
            'voxel_coords']

        if not self.use_elevation:  # if we ignore elevation (z) and v_z
            voxel_features[:, :, self.z_ind] = 0  # set z to zero before doing anything

        orig_xyz = voxel_features[:, :, :self.z_ind + 1]  # selecting x y z

        # calculate mean of points in pillars for x y z and save the offset from the mean
        # Note: they do not take the mean directly, as each pillar is filled up with 0-s. Instead, they sum and divide by num of points
        points_mean = orig_xyz.sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = orig_xyz - points_mean  # offset from cluster mean

        # calculate center for each pillar and save points' offset from the center. voxel_coordinate * voxel size + offset should be the center of pillar (coords are indexed backwards)
        f_center = torch.zeros_like(orig_xyz)
        f_center[:, :, 0] = voxel_features[:, :, self.x_ind] - (
                    coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, self.y_ind] - (
                    coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, self.z_ind] - (
                    coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        voxel_features = voxel_features[:, :, self.selected_indexes]  # filtering for used features

        features = [voxel_features, f_cluster, f_center]

        if self.with_distance:  # if with_distance is true, include range to the points as well
            points_dist = torch.norm(orig_xyz, 2, 2, keepdim=True)  # first 2: L2 norm second 2: along 2. dim
            features.append(points_dist)

        ## finishing up the feature extraction with correct shape and masking
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask

        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        # print('features',features.shape)
        return batch_dict


class PillarVFE_DUAL(VFETemplate):
    def __init__(self, model_cfg, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        
        #读取 config中的设置  对该模块进行调整   一共三个值 在config中都有声明
        self.use_norm = self.model_cfg.USE_NORM   #是否进行归一化
        self.with_distance = self.model_cfg.WITH_DISTANCE  #是否计算距离
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ  #是否使用绝对坐标
        num_point_features = self.model_cfg.NUM_POINT_FEATURES
        num_point_features += 6 if self.use_absolute_xyz else 3  #使用绝对坐标的话 就是加6维 不使用就加3维
        if self.with_distance:
            num_point_features += 1   #计算距离的话加一维

        self.num_filters = self.model_cfg.NUM_FILTERS   #指定特征提取网络中每个卷积层 滤波器的数量
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)  #相当于把输入层的滤波器数量（num_point_features特征维度） 和 config中设置的其他层的滤波器数量（特征维度）给拼起来
        # print('num_filters',num_filters)
        #初始化 point feature network
        pfn_layers = []  
        for i in range(len(num_filters) - 1):  #输入 、、、、 输出  所以需要num_filters个数减一
            in_filters = num_filters[i]         #输入滤波器数量
            out_filters = num_filters[i + 1]    #输出滤波器数量
            pfn_layers.append(
                PFNLayer_DUAL(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        #初始化点云特征提取网络 pfn_layers
        self.pfn_layers = nn.ModuleList(pfn_layers)
        
        #提取 单个voxel 在 x y z上的大小
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        #在 点云范围 x y z 的起始位置上 加上单个 voxel大小的一半
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]  
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]
   
    #用于获取输出特征的维度，该方法返回了PFN网络中最后一层的输出特征维度
    def get_output_feature_dim(self):
        return self.num_filters[-1]
    #确定哪些位置是进行了特征填充的  返回一个布尔张量
    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
       
        voxel_features, voxel_num_points, coords = batch_dict['surface_voxels'], batch_dict['surface_voxel_num_points'], batch_dict['surface_voxel_coords']
        # print('voxel_features\n',voxel_features.shape)
        # print('voxel_num_points\n',voxel_num_points.shape)
        # print('coords\n',coords.shape)
        
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)   #此句代码的作用跟 MEAN_VFE类似  对于每一个voxel中的点云坐标进行平均（先sum，然后除以voxel中点云的个数），获取一个voxel中所有点云坐标的平均值
        f_cluster = voxel_features[:, :, :3] - points_mean  #聚类特征，即 一个voxel中 每一点 与这个voxel所有点平均坐标之间的差距
        
        #计算各体素中 每一个点 与 体素中心点之间的坐标差距
        f_center = torch.zeros_like(voxel_features[:, :, :3])   #初始化一个 与 voxel_features[:, :, :3]大小相同的 tensor
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)  #确定每一个点 与 点所在voxel中心点 之间的x坐标差距  计算方法（voxel_features[:, :, 0] 【点云的x坐标】 - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1)【点云所在voxel在空间中的坐标编号】 * self.voxel_x 【单个voxel在x轴方向上的大小】+ self.x_offset【x轴上第一个voxel的中心点的x坐标】) ）
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        #如果 model_cfg.WITH_DISTANCE 为 TRUE   则计算 每个点到 坐标原点的欧式距离
        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)  #计算 点云特征中 每一个点 相较于 坐标原点的 欧氏距离
            features.append(points_dist)
        #将 列表转成 tensor
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1] #确定体素个数
        #体素过滤  仅保留有效体素
        #确定有效体素 生成掩码
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features) #修改mask的 大小 和 数据类型
        features *= mask  #将有效体素掩码与体素特征相乘   过滤掉无用的体素特征
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()  #压缩特征张量的维度，去除大小为1的维度
        batch_dict['surface_pillar_features'] = features
        # print('VFE_surface_pillar_features', features.shape)
        return batch_dict
    
