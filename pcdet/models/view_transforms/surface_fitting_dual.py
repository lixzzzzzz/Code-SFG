import torch
from torch import nn
from pcdet.ops.bev_pool import bev_pool
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time

from torchvision.ops import roi_align
import pandas as pd


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor(
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]
    )
    return dx, bx, nx

def filter_outliers_iqr(tensor, return_mask=False):
    q1 = torch.quantile(tensor, 0.25)
    q3 = torch.quantile(tensor, 0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    mask = (tensor >= lower_bound) & (tensor <= upper_bound)
    if return_mask:
        return tensor[mask], mask
    else:
        return tensor[mask]

def transform_bboxes(bbox, conv_layers):
    transformed_bboxes = []
    batch, x1, y1, x2, y2 = bbox
    for layer in conv_layers:
        if isinstance(layer, nn.Conv2d):
            kernel_size = layer.kernel_size[0]
            stride = layer.stride[0]
            padding = layer.padding[0]
            x1 = (x1 + padding - (kernel_size // 2)) // stride
            y1 = (y1 + padding - (kernel_size // 2)) // stride
            x2 = (x2 + padding - (kernel_size // 2)) // stride
            y2 = (y2 + padding - (kernel_size // 2)) // stride
        elif isinstance(layer, nn.MaxPool2d):
            kernel_size = layer.kernel_size
            stride = layer.stride
            padding = layer.padding
            x1 = (x1 + 2 * padding - kernel_size) // stride + 1
            y1 = (y1 + 2 * padding - kernel_size) // stride + 1
            x2 = (x2 + 2 * padding - kernel_size) // stride + 1
            y2 = (y2 + 2 * padding - kernel_size) // stride + 1

    transformed_bboxes = [batch, x1, y1, min(x2+1,88), min(y2+1,32)]
    return transformed_bboxes

tv = None
try:
    import cumm.tensorview as tv
except:
    pass
class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        # print('self.spconv_ver',self.spconv_ver)
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            points = points.detach().cpu().numpy()
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points



class SurfaceFitting_Dual(nn.Module):
    """
        This module implements LSS, which lists images into 3D and then splats onto bev features.
        This code is adapted from https://github.com/mit-han-lab/bevfusion/ with minimal modifications.
    """
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        in_channel = self.model_cfg.IN_CHANNEL
        out_channel = self.model_cfg.OUT_CHANNEL
        self.image_size = self.model_cfg.IMAGE_SIZE
        self.feature_size = self.model_cfg.FEATURE_SIZE
        xbound = self.model_cfg.XBOUND
        ybound = self.model_cfg.YBOUND
        zbound = self.model_cfg.ZBOUND
        self.dbound = self.model_cfg.DBOUND
        downsample = self.model_cfg.DOWNSAMPLE

        dx, bx, nx = gen_dx_bx(xbound, ybound, zbound)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.C = out_channel
        self.frustum = self.create_frustum()
        self.D = self.frustum.shape[0]

        self.roi_align = roi_align

        self.voxel_size = self.model_cfg.VOXEL_SIZE_DUAL
        self.point_cloud_range = self.model_cfg.POINT_CLOUD_RANGE
        self.num_point_features = self.model_cfg.NEM_POINT_FEATURES
        self.max_num_points_per_voxel = self.model_cfg.MAX_POINTS_PER_VOXEL
        self.max_num_voxels = self.model_cfg.MAX_NUMBER_OF_VOXELS

        self.dtransform_point = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        self.dtransform_mask = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 5, stride=4, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 16, 5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
        )
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channel + 48 , in_channel, 3, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channel, in_channel, 3, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channel, self.D + self.C, 1),
        )

       
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
                nn.Conv2d(out_channel, out_channel, 3, stride=downsample, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
                nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

        self.depth_feature_mask = nn.Sequential(
            nn.Conv2d(1, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.depth_feature_fpn = nn.Sequential(
            nn.Conv2d(256, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )


        self.fc1 = nn.Linear(9408, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 6)
        

    def create_frustum(self):
        iH, iW = self.image_size
        fH, fW = self.feature_size

        ds = torch.arange(*self.dbound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, iW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, iH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
        frustum = torch.stack((xs, ys, ds), -1)
        
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, camera2lidar_rots, camera2lidar_trans, intrins, post_rots, post_trans, **kwargs):
        camera2lidar_rots = camera2lidar_rots.to(torch.float)
        camera2lidar_trans = camera2lidar_trans.to(torch.float)
        intrins = intrins.to(torch.float)
        post_rots = post_rots.to(torch.float)
        post_trans = post_trans.to(torch.float)
        B, N, _ = camera2lidar_trans.shape

        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:3]), 5)
        combine = camera2lidar_rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)

        if "extra_rots" in kwargs:
            extra_rots = kwargs["extra_rots"]
            points = extra_rots.view(B, 1, 1, 1, 1, 3, 3).repeat(1, N, 1, 1, 1, 1, 1) \
                .matmul(points.unsqueeze(-1)).squeeze(-1)
            
        if "extra_trans" in kwargs:
            extra_trans = kwargs["extra_trans"]
            points += extra_trans.view(B, 1, 1, 1, 1, 3).repeat(1, N, 1, 1, 1, 1)
        return points

    def project_depth_to_lidar(self, depth_final_network, camera2lidar_rots, camera2lidar_trans, intrins, post_rots, post_trans, **kwargs):
        camera2lidar_rots = camera2lidar_rots.to(torch.float)
        camera2lidar_trans = camera2lidar_trans.to(torch.float)
        intrins = intrins.to(torch.float)
        post_rots = post_rots.to(torch.float)
        post_trans = post_trans.to(torch.float)
        B, N, _ = camera2lidar_trans.shape

        H, W = depth_final_network.shape[3], depth_final_network.shape[4]

        h_coords = torch.arange(H, dtype=torch.float32).view(1, 1, 1, H, 1).expand(depth_final_network.shape[0], 1, 1, H, W)
        h_coords = h_coords.to(depth_final_network.device)
        w_coords = torch.arange(W, dtype=torch.float32).view(1, 1, 1, 1, W).expand(depth_final_network.shape[0], 1, 1, H, W)
        w_coords = w_coords.to(depth_final_network.device)

        depth_final_network = torch.stack((w_coords, h_coords, depth_final_network), dim=-1)

        points = depth_final_network - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        
        # cam_to_lidar
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:3]), 5)
        combine = camera2lidar_rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)

        if "extra_rots" in kwargs:
            extra_rots = kwargs["extra_rots"]
            points = extra_rots.view(B, 1, 1, 1, 1, 3, 3).repeat(1, N, 1, 1, 1, 1, 1) \
                .matmul(points.unsqueeze(-1)).squeeze(-1)
            
        if "extra_trans" in kwargs:
            extra_trans = kwargs["extra_trans"]
            points += extra_trans.view(B, 1, 1, 1, 1, 3).repeat(1, N, 1, 1, 1, 1)
        
        points = points.squeeze(dim=2)
  
        return points


    def bev_pool(self, geom_feats, x):
        geom_feats = geom_feats.to(torch.float)
        x = x.to(torch.float)

        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        x = x.reshape(Nprime, C)

        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]
        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

        final = torch.cat(x.unbind(dim=2), 1)

        return final

    def get_cam_feats(self, x, d, mask, mask_org):
        
        B, N, C, fH, fW = x.shape

        d = d.view(B * N, *d.shape[2:])

        mask = mask.view(B * N, *mask.shape[2:])
        mask_output = torch.zeros(mask.shape[0], int(mask.shape[2]/8), int(mask.shape[3]/8))
        for h in range(mask.shape[0]):
            for i in range(int(mask.shape[2]/8)):
                for j in range(int(mask.shape[3]/8)):
                    window = mask[h,0,i*8:i*8+8, j*8:j*8+8]
                    if torch.any(window):
                        mask_output[h, i, j] = 1           
        
        x = x.view(B * N, C, fH, fW)

        d = self.dtransform_point(d)
        mask = self.dtransform_mask(mask)

        x = torch.cat([mask, d, x], dim=1)
        x = self.depthnet(x)
        depth = x[:, : self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        zero_indices = torch.nonzero(mask_output == 0)
        x_depth = x.clone()
        x = x_depth.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x
    

    def network_surface(self, img_fpn, masked_dist, depth_result, mask_aug, dist_cal):
        loss_fitting = torch.tensor(0.0, device=mask_aug.device, requires_grad=True)
        depth_network = torch.zeros_like(mask_aug)
        depth_network_loss = torch.zeros_like(mask_aug)
        img_fpn_network = img_fpn
        img_fpn_network = self.depth_feature_fpn(img_fpn_network)
        depth_result = depth_result.unsqueeze(0).unsqueeze(1)
        depth_result = self.depth_feature_mask(depth_result)
        fusion = torch.cat([depth_result, img_fpn_network], dim=1)

        unique_classes = torch.unique(mask_aug)
    
        bboxe = []
        for instance_id in unique_classes:
            if instance_id != 0:
                device = mask_aug.device
                mask_indices = (mask_aug == instance_id).nonzero().to(device)
                dist_values = dist_cal[mask_indices[:, 0], mask_indices[:, 1]]
                dist_cal = dist_cal.to(device)

                non_zero_mask = dist_values != 0
                non_zero_mask = non_zero_mask.to(device)
                non_zero_dist_values = dist_values[non_zero_mask]
                
                non_zero_mask_indices = mask_indices[non_zero_mask]


                if len(non_zero_dist_values) == 0:
                    continue

                non_zero_dist_values, filter_mask = filter_outliers_iqr(non_zero_dist_values, return_mask=True)
                non_zero_dist_values = non_zero_dist_values.clone().detach().to(device)
                filter_mask = filter_mask.clone().detach().to(device)


                filtered_mask_indices = non_zero_mask_indices[filter_mask]

                if len(non_zero_dist_values) == 0:
                    continue
                
                if (len(non_zero_dist_values) > 6) & (max(non_zero_dist_values)-min(non_zero_dist_values) < 15) & (non_zero_dist_values.mean() < 50):  
               
                    instance_mask = mask_aug == instance_id
                    y_indices, x_indices = torch.where(instance_mask)
                    x1, x2 = x_indices.min().item(), x_indices.max().item()
                    y1, y2 = y_indices.min().item(), y_indices.max().item()
                    bbox = [0, x1, y1, x2, y2]
                    bbox_tran = transform_bboxes(bbox,self.depth_feature_mask)
                    bbox_tran = torch.tensor(bbox_tran,device=img_fpn.device)
                    bbox_tran = bbox_tran.unsqueeze(0)
                    bbox_tran = bbox_tran.float()

                    aligned_features = self.roi_align(fusion, bbox_tran, output_size=(7, 7))
                    aligned_features = aligned_features.view(aligned_features.size(0), -1)
                    liner_feature = self.fc1(aligned_features)
                    liner_feature = nn.ReLU()(liner_feature)
                    liner_feature = self.fc2(liner_feature)
                    liner_feature = nn.ReLU()(liner_feature)
                    plane_params = self.fc3(liner_feature)

                    a_network_plane = plane_params[0,0].to(mask_aug.device)
                    b_network_plane = plane_params[0,1].to(mask_aug.device)
                    c_network_plane = plane_params[0,2].to(mask_aug.device)
                    d_network_plane = plane_params[0,3].to(mask_aug.device)
                    e_network_plane = plane_params[0,4].to(mask_aug.device)
                    f_network_plane = plane_params[0,5].to(mask_aug.device)
                    
                    def predict_z(x, y):
                        return a_network_plane * x**2 + b_network_plane * y**2 + c_network_plane * x * y + d_network_plane * x + e_network_plane * y + f_network_plane
                    
                    x_grid, y_grid = torch.meshgrid(torch.arange(mask_aug.shape[0]), torch.arange(mask_aug.shape[1]))
                    z_grid = predict_z(x_grid, y_grid)

                    if torch.any(z_grid[mask_aug == instance_id] < 0):
                        mean_dist_value = non_zero_dist_values.mean() 
                        depth_network[mask_aug == instance_id] = mean_dist_value
                    else:    
                        depth_network[mask_aug == instance_id] = z_grid[mask_aug == instance_id]

                else:
                    mean_dist_value = non_zero_dist_values.mean()  
                    depth_network[mask_aug == instance_id] = mean_dist_value
        return depth_network,loss_fitting
     
    def fitting_average(self,unique_classes,mask_aug,dist_cal):
        depth_means = []
        for cls in unique_classes:
            if cls.item() == 0:
                continue
            mask_indices = (mask_aug == cls).nonzero()  

            dist_values = dist_cal[mask_indices[:, 0], mask_indices[:, 1]] 
            non_zero_dist_values = dist_values[dist_values != 0]
            if len(non_zero_dist_values) > 0:
                mean_dist_value = non_zero_dist_values.mean()  
                depth_means.append((cls.item(), mean_dist_value.item()))

        depth_result = torch.zeros_like(mask_aug)
        for cls, mean_dist in depth_means:
            depth_result[mask_aug == cls] = mean_dist
        return depth_result

    def fitting_plane(self,unique_classes,mask_aug,dist_cal):
        depth_result = torch.zeros_like(mask_aug)
        for cls in unique_classes:
            if cls.item() != 0:  
                mask_indices = (mask_aug == cls).nonzero() 
                dist_values = dist_cal[mask_indices[:, 0], mask_indices[:, 1]]  
                non_zero_dist_values = dist_values[dist_values != 0] 

                if len(non_zero_dist_values) > 0 & len(non_zero_dist_values) < 4:
                    mean_dist_value = non_zero_dist_values.mean()  
                    depth_result[mask_aug == cls] = mean_dist_value
            
                if len(non_zero_dist_values) > 3:
                    dist_values = dist_values.to(mask_indices.device)
                    coordinate_xydepth = torch.cat((mask_indices, dist_values.unsqueeze(1)), dim=1)
                    xydepth = coordinate_xydepth[coordinate_xydepth[:, 2] != 0]
                    x_values = xydepth[:, 0]
                    y_values = xydepth[:, 1]
                    z_values = xydepth[:, 2]
                    inter_matrix = np.column_stack((x_values, y_values, np.ones(len(xydepth))))

                    coefficients, residuals, _, _ = np.linalg.lstsq(inter_matrix, z_values, rcond=None)

                    a_plane, b_plane, c_plane = coefficients

                    def predict_z(x, y):
                        return a_plane * x + b_plane * y + c_plane
                    
                    x_grid, y_grid = torch.meshgrid(torch.arange(mask_aug.shape[0]), torch.arange(mask_aug.shape[1]))
                    
                    z_grid = predict_z(x_grid, y_grid)
                    depth_result[mask_aug == cls] = z_grid[mask_aug == cls]
        return depth_result

    def fitting_curve(self,unique_classes,mask_aug,dist_cal):  
        depth_result = torch.zeros_like(mask_aug)
        for cls in unique_classes:
            if cls.item() != 0: 
                mask_indices = (mask_aug == cls).nonzero()  

                dist_values = dist_cal[mask_indices[:, 0], mask_indices[:, 1]]  
                non_zero_dist_values = dist_values[dist_values != 0]  

                if len(non_zero_dist_values) > 0 & len(non_zero_dist_values) < 4:
                    mean_dist_value = non_zero_dist_values.mean()  
                    depth_result[mask_aug == cls] = mean_dist_value
            
                if len(non_zero_dist_values) > 3:
                    dist_values = dist_values.to(mask_indices.device)
                    coordinate_xydepth = torch.cat((mask_indices, dist_values.unsqueeze(1)), dim=1)
                    xydepth = coordinate_xydepth[coordinate_xydepth[:, 2] != 0]
                    from sklearn.linear_model import LinearRegression
                    from sklearn.preprocessing import PolynomialFeatures

                    X_train = xydepth[:, :2] 
                    y_train = xydepth[:, 2]  

                    poly = PolynomialFeatures(degree=2) 
                    X_train_poly = poly.fit_transform(X_train)

                    model = LinearRegression()
                    model.fit(X_train_poly, y_train)

                    def predict_z_batch(x, y):
                        points = torch.stack((x, y), dim=1)
                        points_poly = torch.tensor(poly.transform(points.cpu().numpy()), dtype=torch.float32)
                        z_predicted = torch.tensor(model.predict(points_poly), dtype=torch.float32).squeeze()
                        return z_predicted

                    rows = coordinate_xydepth[:, 0].long()
                    cols = coordinate_xydepth[:, 1].long()

                    predicted_z_values = predict_z_batch(rows, cols)

                    depth_result[rows, cols] = predicted_z_values
        return depth_result


    def forward(self, batch_dict):

        frame_idex = batch_dict['frame_id']

        x = batch_dict['image_fpn']
        x = x[0]
        BN, C, H, W = x.size()
        img = x.view(BN, 1, C, H, W)        

        camera_intrinsics = batch_dict['camera_intrinsics']
        camera2lidar = batch_dict['camera2lidar']
        img_aug_matrix = batch_dict['img_aug_matrix']
        lidar_aug_matrix = batch_dict['lidar_aug_matrix']
        lidar2image = batch_dict['lidar2image']
        img_process_infos = batch_dict["img_process_infos"]

        lidar2image = torch.unsqueeze(lidar2image,dim=1)
        camera2lidar = torch.unsqueeze(camera2lidar, dim=1)


        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        intrins = intrins[:1, ...]

        points = batch_dict['points']
        images_shape = batch_dict['image_shape']
        images = batch_dict['images']
    
        batch_size = BN
        depth = torch.zeros(batch_size, img.shape[1], 1, *self.image_size).to(points[0].device)
        depth_final = torch.zeros_like(depth)
        depth_final_network = torch.zeros_like(depth)
        mask_org = torch.zeros_like(depth)
        loss_surface_fitting = torch.tensor(0.0, device=x.device, requires_grad=True)
        for b in range(batch_size):

            batch_mask = points[:,0] == b
            cur_coords = points[batch_mask][:, 1:4]
            cur_img_aug_matrix = img_aug_matrix[b]
            cur_lidar_aug_matrix = lidar_aug_matrix[b]
            cur_lidar2image = lidar2image[b]
            cur_img_process = img_process_infos[b]
            image_shape = images_shape[b]
            image_org = images[b]
            img_fpn = img[b]

            with open("/home/xiaozhi/data/code_py/openPCDet/batch_results_TJ4D/index_map.json", "r") as f:
                index_map = json.load(f)
            img_index = frame_idex[b]
            index_int = int(img_index)
            tensor_index = index_map.get(str(index_int), -1)
            load_path = '/home/xiaozhi/data/code_py/openPCDet/batch_results_TJ4D/mask_spilt_modfiy/mask_{}.pt'.format(tensor_index)
            mask_result = torch.load(load_path)

            resize = cur_img_process[0][0]
            H_img, W_img = image_shape
            resize_dim = (int(W_img * resize), int(H_img * resize))
            crop = cur_img_process[0][1]
            
            mask_img = Image.fromarray(mask_result)
            mask_img = mask_img.resize(resize_dim,resample=Image.NEAREST)
            mask_img = mask_img.crop(crop)
            mask_aug = np.array(mask_img)

            cur_coords -= cur_lidar_aug_matrix[:3, 3]
            cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                cur_coords.transpose(1, 0)
            )

            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)

            dist = cur_coords[:, 2, :]
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)

            cur_coords = cur_coords[..., [1, 0]]

            on_img = (
                (cur_coords[..., 0] < self.image_size[0])
                & (cur_coords[..., 0] >= 0)
                & (cur_coords[..., 1] < self.image_size[1])
                & (cur_coords[..., 1] >= 0)
            )
            
            for c in range(on_img.shape[0]):
                masked_coords = cur_coords[c, on_img[c]].long()
                masked_dist = dist[c, on_img[c]]
                depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist

                dist_cal = torch.zeros(*self.image_size).to(points[0].device)
                dist_cal[masked_coords[:, 0], masked_coords[:, 1]] = masked_dist
                mask_aug = torch.tensor(mask_aug)
                mask_org[b, c, 0, :, :] = mask_aug
                unique_classes = torch.unique(mask_aug) 

                depth_result = self.fitting_average(unique_classes,mask_aug,dist_cal)
                # depth_result = self.fitting_plane(unique_classes,mask_aug,dist_cal)
                # depth_result = self.fitting_curve(unique_classes,mask_aug,dist_cal)
                
                depth_final[b, c, 0, :, :] = depth_result
                

                depth_result = depth_result.to(depth_final.device)

                depth_network,loss_fitting = self.network_surface(img_fpn, masked_dist, depth_result, mask_aug, dist_cal)
                depth_final_network[b, c, 0, :, :] = depth_network
                loss_surface_fitting = loss_surface_fitting + loss_fitting              
       
        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]
        geom = self.get_geometry(
            camera2lidar_rots, camera2lidar_trans, intrins, post_rots, 
            post_trans, extra_rots=extra_rots, extra_trans=extra_trans,
        )
        
        surface_pointcloud = self.project_depth_to_lidar(depth_final_network, camera2lidar_rots, camera2lidar_trans, intrins, post_rots, post_trans, extra_rots=extra_rots, extra_trans=extra_trans,)

        self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=self.voxel_size,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=self.max_num_points_per_voxel,
                max_num_voxels=self.max_num_voxels,
            )
        def process_batch(surface_pointcloud, depth_final_network, voxel_generator):
            batch_size = surface_pointcloud.shape[0]
            all_voxels = []
            all_coordinates = []
            all_num_points = []

            for i in range(batch_size):
                frame_pointcloud = surface_pointcloud[i, 0] 
                frame_depth = depth_final_network[i, 0, 0]    

                non_zero_mask = frame_depth > 0
                non_zero_points = frame_pointcloud[non_zero_mask]
                
                points = non_zero_points.reshape(-1, 3)  
                
                if points.size(0) == 0:
                    points = torch.tensor([[0.00, 0.00, 0.00]])

                voxels, coordinates, num_points = voxel_generator.generate(points)
                index_column = np.full((coordinates.shape[0], 1), i, dtype=coordinates.dtype)
                coordinates = np.hstack((index_column, coordinates))
                
                all_voxels.append(voxels)
                all_coordinates.append(coordinates)
                all_num_points.append(num_points)
            voxels = np.concatenate(all_voxels, axis=0)
            num_points = np.concatenate(all_num_points, axis=0)
            coordinates = np.concatenate(all_coordinates, axis=0)
            
            return voxels, num_points, coordinates

        surface_voxels, surface_voxel_num_points, surface_voxel_coords = process_batch(surface_pointcloud, depth_final_network, self.voxel_generator)
        
        x = self.get_cam_feats(img, depth, depth_final_network,mask_org)
        x = self.bev_pool(geom, x)
        x = self.downsample(x)
        x = x.permute(0, 1, 3, 2)

        batch_dict['spatial_features_img'] = x
        surface_voxels = torch.tensor(surface_voxels,device=x.device)
        surface_voxel_num_points = torch.tensor(surface_voxel_num_points,device=x.device)
        surface_voxel_coords = torch.tensor(surface_voxel_coords,device=x.device)
        batch_dict['surface_voxels'] = surface_voxels
        batch_dict['surface_voxel_num_points'] = surface_voxel_num_points
        batch_dict['surface_voxel_coords'] = surface_voxel_coords
        return batch_dict