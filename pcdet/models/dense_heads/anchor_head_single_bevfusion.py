import numpy as np
import torch.nn as nn

from .anchor_head_template_bevfusion import AnchorHeadTemplate_bevfusion

class AnchorHeadSingle_bevfusion(AnchorHeadTemplate_bevfusion):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )
        #每个位置的anchor数量
        self.num_anchors_per_location = sum(self.num_anchors_per_location)
        #类别卷积层 输出通道为 anchor个数 * 类别数量
        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        #bbox回归卷积层  输出通道为 anchor个数 * 每个边界框参数个数（一般4个2d 6个3d）
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )
        #方向分类器
        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        #权重初始化
        self.init_weights()
    
    #权重初始化
    def init_weights(self):
        pi = 0.01
        # 设置分类卷积层的偏执
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        # 设置bbox回归卷积层的权重
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        #读取 backbone2d 的特征，bev特征
        spatial_features_2d = data_dict['spatial_features_2d']
        
        #输入2d的bev特征  经过self中定义的对应网络 确定 每一个anchor对应的 类别、bbox、方向
        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        #根据数据集中的ground truth数值，生成该batch中训练所需要的对应标签数据 其中包括 类别标签、bbox回归标签、方向标签，这个过程跟前面的预测完全没关系，只需要输入ground truth就行
        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        #预测结果生成
        #如果不在训练模式  或者  训练过程中需要输出预测结果
        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict
