import torch
from torch import nn
import torch.nn.functional as F

def clip_sigmoid(x, eps=1e-4):
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y

#位置坐标 embedding（相当于输入坐标，输出一个坐标的映射，方便往网络里面输入）
#就是一个简单的一维卷积块
class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding

#定义TransformerDecoder
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 self_posembed=None, cross_posembed=None, cross_only=False):
        super().__init__()
        self.cross_only = cross_only        #判断是否只使用 cross attention
        if not self.cross_only:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        #前馈模型实现
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        #确定激活函数类型
        def _get_activation_fn(activation):
            """Return an activation function given a string"""
            if activation == "relu":
                return F.relu
            if activation == "gelu":
                return F.gelu
            if activation == "glu":
                return F.glu
            raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

        self.activation = _get_activation_fn(activation)
        
        #位置坐标embedding初始化
        self.self_posembed = self_posembed
        self.cross_posembed = cross_posembed
    #判断是否需要 位置embedding
    def with_pos_embed(self, tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, query, key, query_pos, key_pos, key_padding_mask=None, attn_mask=None):
        # NxCxP to PxNxC
        if self.self_posembed is not None:
            query_pos_embed = self.self_posembed(query_pos).permute(2, 0, 1)  #对query_pos进行位置编码  并转换维度顺序   原始张量的维度顺序是(batch_size, seq_length, embedding_dim)，而经过permute操作后，维度顺序变为(seq_length, batch_size, embedding_dim)。这种维度排列方式是Transformer模型所需要的标准输入形式，其中seq_length表示序列长度，batch_size表示批量大小，embedding_dim表示每个位置的嵌入维度。
        else:
            query_pos_embed = None
        if self.cross_posembed is not None:
            key_pos_embed = self.cross_posembed(key_pos).permute(2, 0, 1)   ##对key_pos进行位置编码  并转换维度顺序
        else:
            key_pos_embed = None
        #调整输入维度顺序
        query = query.permute(2, 0, 1)
        key = key.permute(2, 0, 1)
        #采用 self_attetion
        if not self.cross_only:
            q = k = v = self.with_pos_embed(query, query_pos_embed)
            query2 = self.self_attn(q, k, value=v)[0]
            query = query + self.dropout1(query2)
            query = self.norm1(query)
        #采用 cross_attention
        query2 = self.multihead_attn(query=self.with_pos_embed(query, query_pos_embed),
                                     key=self.with_pos_embed(key, key_pos_embed),
                                     value=self.with_pos_embed(key, key_pos_embed), 
                                     key_padding_mask=key_padding_mask, attn_mask=attn_mask)[0]

        query = query + self.dropout2(query2)
        query = self.norm2(query)
        #层数继续加深
        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm3(query)

        # NxCxP to PxNxC
        query = query.permute(1, 2, 0)  #重新调整输出的tensor顺序
        return query

