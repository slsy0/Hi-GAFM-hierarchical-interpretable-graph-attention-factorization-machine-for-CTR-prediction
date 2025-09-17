import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import EmbeddingLayer, FiGAT_Layer3, FM_Layer, InnerProductLayer, LR_Layer, MLP_Layer
from fuxictr.pytorch.layers import BilinearInteractionLayer
from fuxictr.pytorch.models.AutoInt import MultiHeadSelfAttention
from fuxictr.pytorch.models.InterHAt import AttentionalAggregation
import pdb
import numpy as np


class GAFM5(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="FiGAT_FM",
                 gpu=-1,
                 task="binary_classification",
                 learning_rate=1e-3,
                 embedding_dim=10,
                 gat_layers=3,
                 use_residual=True,
                 use_gru=True,
                 reuse_graph_layer=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(GAFM5, self).__init__(feature_map,
                                    model_id=model_id,
                                    gpu=gpu,
                                    embedding_regularizer=embedding_regularizer,
                                    net_regularizer=net_regularizer,
                                    **kwargs)
        self.embedding_dim = embedding_dim
        self.num_fields = feature_map.num_fields
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        self.att_w = FiGAT_Layer3(self.num_fields,
                                  self.embedding_dim,
                                  gat_layers=gat_layers,
                                  reuse_graph_layer=reuse_graph_layer,
                                  use_gru=use_gru,
                                  use_residual=use_residual,
                                  device=self.device)
        self.fm_fields = int(self.num_fields * (self.num_fields - 1) / 2)
        self.fm_fields4 = int(3*self.num_fields * (3*self.num_fields - 1) / 2)
        # self.fc = AttentionalPrediction(self.fm_fields, embedding_dim).cuda()
        self.fc3 = AttentionalPrediction(3*self.fm_fields, embedding_dim)
        self.fc4 = AttentionalPrediction(self.fm_fields4, embedding_dim)
        # self.fc5 = AttentionalPrediction(self.fm_fields4+3*self.fm_fields, embedding_dim)
        self.output_activation = self.get_output_activation(task)  # 使用sigmoid函数激活
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()
        self.inner_product_layer = InnerProductLayer(feature_map.num_fields, output="elementwise_product").cuda()
        self.inner_product_layer3 = InnerProductLayer(3*feature_map.num_fields, output="elementwise_product").cuda()# product_sum_pooling
        # Bilinear_Interaction Layer field_all
        # self.lr_layer = LR_Layer(feature_map, output_activation=None, use_bias=True).cuda()
        # self.batch_norm = nn.BatchNorm1d(1, affine=True).cuda()
        # self.batch_norm3 = nn.BatchNorm1d(3).cuda()

        self.gat_layers = gat_layers
        # self.append = np.append(axis=2)
        # attention
        # self.self_attention = MultiHeadSelfAttention(1, use_residual=use_residual).cuda()
        # self.attentional_score = AttentionalAggregation(embedding_dim, None).cuda()
        # self.dropout = nn.Dropout(0.5)
        self.gru = nn.GRUCell(embedding_dim, embedding_dim).cuda()
        self.linear = nn.Linear(in_features=feature_map.num_fields, out_features=feature_map.num_fields).cuda()

    def forward(self, inputs):
        X, y = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X)

        # h_out = self.figat(feature_emb)
        att_w = self.att_w(feature_emb)  # [batch_size, N, N][3]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # lr_out = self.lr_layer(X)
        feature_emb = feature_emb.to(device)

        # 3 concat后输入fc
        attention_sum = []
        for i in range(self.gat_layers):
            attention_sum.append(torch.matmul(att_w[i], feature_emb))
            # gru
        for i in range(self.gat_layers):
            attention_sum[i] = attention_sum[i].view(-1, self.embedding_dim)
            h = feature_emb.view(-1, self.embedding_dim)
            h = self.gru(h, attention_sum[i])
            h = h.view(-1, self.num_fields, self.embedding_dim)
            h += feature_emb  # use residual
            attention_sum[i] = h
            # FM
        '''
        fc_input = []

        for i in range(self.gat_layers):
            inner_product_vec = self.inner_product_layer(attention_sum[i])
            fc_input.append(inner_product_vec)

        fc_input2 = torch.cat(attention_sum, dim=1)
        fc_input2 = self.inner_product_layer3(fc_input2)

        fc_input.append(fc_input2)
        fc_input = torch.cat(fc_input, dim=1)

        y_pred = self.fc5(fc_input)
        '''
        fc_input = []
        for i in range(self.gat_layers):
            inner_product_vec = self.inner_product_layer(attention_sum[i])
            # inner_product_vec = self.dropout(inner_product_vec)
            fc_input.append(inner_product_vec)
        fc_input = torch.cat(fc_input, dim=1)
        # fc_input = self.dropout(fc_input)
        y_pred = self.fc3(fc_input)

        fc_input2 = torch.cat(attention_sum, dim=1)
        fc_input2 = self.inner_product_layer3(fc_input2)
        # fc_input2 = self.dropout(fc_input2)
        y_pred += self.fc4(fc_input2)

        y_pred = y_pred.to(device)
        if self.output_activation is not None:
            y_pred = self.output_activation(y_pred)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict


# 计算最终输出的注意力分数的预测
class AttentionalPrediction(nn.Module):
    def __init__(self, num_fields, embedding_dim):
        super(AttentionalPrediction, self).__init__()
        self.mlp1 = nn.Linear(embedding_dim, 1, bias=False).cuda()  # 对传入的数据应用线性转换
        self.mlp2 = nn.Sequential(nn.Linear(num_fields * embedding_dim, num_fields, bias=False),
                                  nn.Sigmoid()).cuda()  # nn.Sequential():顺序容器。模块将按照在构造函数中传递它们的顺序添加到其中。

    def forward(self, h):   # 1000x3x1
        score = self.mlp1(h).squeeze(-1).cuda()  # b x f   numpy.squeeze()函数  axis=-1时删除倒数第一个维度   1000x3
        weight = self.mlp2(h.flatten(start_dim=1))  # b x f   1000x3
        # flatten()是对多维数据的降维函数从第dim个维度开始展开，将后面的维度转化为一维.也就是说，只保留dim之前的维度，其他维度的数据全都挤在dim这一维。
        logit = (weight * score).sum(dim=1).unsqueeze(-1)  # .sum(dim):tensor对dim维度进行求和，压缩维度  1000x1
        # tensor*tensor为按元素相乘  # .unsqueeze():在特定维度升一维
        return logit  # batch_size x 1
