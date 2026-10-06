from .detector3d_template import Detector3DTemplate
from .. import backbones_image, view_transforms
from ..backbones_image import img_neck
from ..backbones_2d import fuser

class BevFusion(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        #这个地方跟是因为 Detector3DTemplate的模板中没有图像相关的模块  也就是第二行的内容，所以要重新声明一下
        self.module_topology = [ 
            'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
            'image_backbone','neck','vtransform','fuser',
            'backbone_2d', 'dense_head',  'point_head', 'roi_head'    #这应该是当前openpcdet中包含的全部模块了
        ]
        self.module_list = self.build_networks()   #构建网络
    
    #此处定义这些是因为 Detector3DTemplate的模板中没有这些模块
    #创建 neck
    def build_neck(self,model_info_dict):
        if self.model_cfg.get('NECK', None) is None:   #查看model_cfg中是否有neck这个模块，如果没有就返回none
            return None, model_info_dict
        neck_module = img_neck.__all__[self.model_cfg.NECK.NAME](   #根据model_cfg.NECK.NAME 来从前面import中的 img_neck所有模块中找出来需要的那个模块，并使用model_cfg.NECK 里面的参数进行实例化
            model_cfg=self.model_cfg.NECK
        )
        model_info_dict['module_list'].append(neck_module)   #将创建好的模块添加到 module_list列表中

        return neck_module, model_info_dict
    #创建 视角转换    
    def build_vtransform(self,model_info_dict):
        if self.model_cfg.get('VTRANSFORM', None) is None:
            return None, model_info_dict
        
        vtransform_module = view_transforms.__all__[self.model_cfg.VTRANSFORM.NAME](
            model_cfg=self.model_cfg.VTRANSFORM
        )
        model_info_dict['module_list'].append(vtransform_module)

        return vtransform_module, model_info_dict
    #创建 image backbone
    def build_image_backbone(self, model_info_dict):
        if self.model_cfg.get('IMAGE_BACKBONE', None) is None:
            return None, model_info_dict
        image_backbone_module = backbones_image.__all__[self.model_cfg.IMAGE_BACKBONE.NAME](
            model_cfg=self.model_cfg.IMAGE_BACKBONE
        )
        image_backbone_module.init_weights()
        model_info_dict['module_list'].append(image_backbone_module)

        return image_backbone_module, model_info_dict
    #创建 融合模块
    def build_fuser(self, model_info_dict):
        if self.model_cfg.get('FUSER', None) is None:
            return None, model_info_dict
    
        fuser_module = fuser.__all__[self.model_cfg.FUSER.NAME](
            model_cfg=self.model_cfg.FUSER
        )
        model_info_dict['module_list'].append(fuser_module)
        model_info_dict['num_bev_features'] = self.model_cfg.FUSER.OUT_CHANNEL
        return fuser_module, model_info_dict
    

    #前向传播过程
    def forward(self, batch_dict):

        for i,cur_module in enumerate(self.module_list):    #遍历模型每个模块
            batch_dict = cur_module(batch_dict)   #每个模块向前传播计算
        
        if self.training:       #如果是训练模式
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)   #计算loss  返回值为  损失  可视化字典  显示字典

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:   #不处于训练模型  就不需要计算loss
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts  #返回预测字典 和 召回字典
    #计算训练loss
    def get_training_loss(self,batch_dict):   
        disp_dict = {}    #存储训练过程中显示信息

        loss_trans, tb_dict = batch_dict['loss'],batch_dict['tb_dict']
        tb_dict = {
            'loss_trans': loss_trans.item(),
            **tb_dict
        }

        loss = loss_trans
        return loss, tb_dict, disp_dict
    #推理后处理 返回模型预测结果 和 召回率
    def post_processing(self, batch_dict):  #推理时后处理
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        #这个Detector3DTemplate模版中没有 所以自己实现
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict
