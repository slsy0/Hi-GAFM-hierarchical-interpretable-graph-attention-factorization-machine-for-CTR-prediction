import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import EmbeddingLayer, FiGAT_Layer3, FM_Layer, InnerProductLayer, LR_Layer, MLP_Layer
from fuxictr.pytorch.layers import BilinearInteractionLayer
from fuxictr.pytorch.models.InterHAt import MultiHeadSelfAttention, FeedForwardNetwork
from fuxictr.pytorch.models.InterHAt import AttentionalAggregation
import pdb
import numpy as np


class GAFM4(BaseModel):
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
        super(GAFM4, self).__init__(feature_map,
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
        self.fc = AttentionalPrediction(self.fm_fields, embedding_dim).cuda()
        self.fc3 = AttentionalPrediction(3*self.fm_fields, embedding_dim)
        self.fc4 = AttentionalPrediction(self.fm_fields4, embedding_dim)
        self.output_activation = self.get_output_activation(task)  # 使用sigmoid函数激活
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()
        self.inner_product_layer = InnerProductLayer(feature_map.num_fields, output="elementwise_product").cuda()
        self.inner_product_layer3 = InnerProductLayer(3*feature_map.num_fields, output="elementwise_product").cuda()# product_sum_pooling
        # Bilinear_Interaction Layer field_all
        self.lr_layer = LR_Layer(feature_map, output_activation=None, use_bias=True).cuda()
        # self.batch_norm = nn.BatchNorm1d(1, affine=True).cuda()
        # self.batch_norm3 = nn.BatchNorm1d(3).cuda()

        self.gat_layers = gat_layers
        # self.append = np.append(axis=2)
        # attention
        self.self_attention = MultiHeadSelfAttention(embedding_dim, attention_dim=32, num_heads=1, use_residual=use_residual, use_scale=True, layer_norm=True).cuda()
        self.feedforward = FeedForwardNetwork(embedding_dim,
                                              hidden_dim=None,
                                              layer_norm=True,
                                              use_residual=use_residual).cuda()
        # self.attentional_score = AttentionalAggregation(embedding_dim, None).cuda()
        # self.dropout = nn.Dropout(0.1)
        self.gru = nn.GRUCell(embedding_dim, embedding_dim).cuda()
        self.linear = nn.Linear(in_features=feature_map.num_fields, out_features=feature_map.num_fields).cuda()
        self.W = nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim).cuda()

    def forward(self, inputs):
        X, y = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X)

        # h_out = self.figat(feature_emb)
        att_w = self.att_w(feature_emb)  # [batch_size, N, N][3]


        a_score = []
        for i in range(self.gat_layers):
            a_score.append(att_w[i].sum(dim=0))


        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # lr_out = self.lr_layer(X)
        feature_emb = feature_emb.to(device)
        '''
        # 1 做平均
        
        # out = att_w[0]
        out = self.linear(att_w[0])
        for i in range(self.gat_layers-1):
            out = torch.add(out, self.linear(att_w[i+1]))
        out = torch.div(out, self.gat_layers)
        attention_sum = torch.matmul(out, feature_emb)

        # gru
        attention_sum = attention_sum.view(-1, self.embedding_dim)
        h = feature_emb.view(-1, self.embedding_dim)
        h = self.gru(h, attention_sum)
        h = h.view(-1, self.num_fields, self.embedding_dim)
        h += feature_emb  # use residual
        attention_sum = h

        # fm
        # self.inner_product_layer = self.inner_product_layer.to(device)
        fm_out = self.inner_product_layer(attention_sum)
        y_pred = fm_out

        y_pred = self.fc(y_pred)
        y_pred += self.lr_layer(X)
        '''
        # 3 concat后输入fc
        attention_sum = []
        for i in range(self.gat_layers):
            attention_sum.append(torch.matmul(self.linear(att_w[i]), feature_emb))  # att_w[i]可以用self.linear()
            # W = nn.Parameter(torch.Tensor(self.num_fields, self.embedding_dim, self.embedding_dim))  # (torch.zeros(size=(batch_size, feature_emb.shape[2], feature_emb.shape[2])))
            # nn.init.xavier_uniform_(W.data, gain=1.414)
            # W = W.to(device)
            # attention_sum[i] = torch.matmul(W, attention_sum[i].unsqueeze(-1)).squeeze(-1)
            # attention_sum[i] = self.W(attention_sum[i])

            # gru
        for i in range(self.gat_layers):
            attention_sum[i] = attention_sum[i].view(-1, self.embedding_dim)
            h = feature_emb.view(-1, self.embedding_dim)
            h = self.gru(h, attention_sum[i])
            h = h.view(-1, self.num_fields, self.embedding_dim)
            h += feature_emb  # use residual
            attention_sum[i] = h
            # FM
        fc_input = []
        batch_size = feature_emb.shape[0]
        for i in range(self.gat_layers):
            # W = nn.Parameter(torch.Tensor(self.fm_fields, self.embedding_dim, self.embedding_dim))  # (torch.zeros(size=(batch_size, feature_emb.shape[2], feature_emb.shape[2])))
            # nn.init.xavier_uniform_(W.data, gain=1.414)
            # W = W.to(device)
            inner_product_vec = self.inner_product_layer(attention_sum[i])
            # inner_product_vec = self.W(inner_product_vec)
            # inner_product_vec = torch.matmul(W, inner_product_vec.unsqueeze(-1)).squeeze(-1)
            # inner_product_vec = self.feedforward(inner_product_vec)
            fc_input.append(inner_product_vec)
        fc_input = torch.cat(fc_input, dim=1)
        # fc_input = self.self_attention(fc_input)  # self attention
        # fc_input = self.feedforward(fc_input)
        y_pred = self.fc3(fc_input)

        # fc_input = torch.cat(attention_sum, dim=1)
        # fc_input = self.inner_product_layer3(fc_input)
        # y_pred = self.fc4(fc_input)

        # y_pred = self.batch_norm(output)
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

    def forward(self, h):   # b x f x emb
        score = self.mlp1(h).squeeze(-1).cuda()  # b x f   numpy.squeeze()函数  axis=-1时删除倒数第一个维度
        weight = self.mlp2(h.flatten(start_dim=1))  # b x f
        # flatten()是对多维数据的降维函数从第dim个维度开始展开，将后面的维度转化为一维.也就是说，只保留dim之前的维度，其他维度的数据全都挤在dim这一维。

        b_score = (weight.sum(dim=0))

        logit = (weight * score).sum(dim=1).unsqueeze(-1)  # .sum(dim):tensor对dim维度进行求和，压缩维度  1000x1
        # tensor*tensor为按元素相乘  # .unsqueeze():在特定维度升一维
        return logit  # batch_size x 1
