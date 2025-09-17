import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import EmbeddingLayer, FiGATv2_Layer


class FiGATv2(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="FiGAT",
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
        super(FiGATv2, self).__init__(feature_map,
                                    model_id=model_id,
                                    gpu=gpu,
                                    embedding_regularizer=embedding_regularizer,
                                    net_regularizer=net_regularizer,
                                    **kwargs)
        self.embedding_dim = embedding_dim
        self.num_fields = feature_map.num_fields
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        self.figat = FiGATv2_Layer(self.num_fields,
                                 self.embedding_dim,
                                 gat_layers=gat_layers,
                                 reuse_graph_layer=reuse_graph_layer,
                                 use_gru=use_gru,
                                 use_residual=use_residual,
                                 device=self.device)
        self.fc = AttentionalPrediction(self.num_fields, embedding_dim)
        self.output_activation = self.get_output_activation(task)  # 使用sigmoid函数激活
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, inputs):
        X, y = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X)
        h_out = self.figat(feature_emb)
        # h_out = self.dropout(h_out)
        y_pred = self.fc(h_out)  # h_out
        if self.output_activation is not None:
            y_pred = self.output_activation(y_pred)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict


# 计算最终输出的注意力分数的预测
class AttentionalPrediction(nn.Module):
    def __init__(self, num_fields, embedding_dim):
        super(AttentionalPrediction, self).__init__()
        self.mlp1 = nn.Linear(embedding_dim, 1, bias=False)   # 对传入的数据应用线性转换
        self.mlp2 = nn.Sequential(nn.Linear(num_fields * embedding_dim, num_fields, bias=False),
                                  nn.Sigmoid())   # 顺序容器。模块将按照在构造函数中传递它们的顺序添加到其中。

    def forward(self, h):
        score = self.mlp1(h).squeeze(-1)  # b x f   numpy.squeeze()函数  axis=-1时删除倒数第一个维度
        weight = self.mlp2(h.flatten(start_dim=1))  # b x f
        # flatten()是对多维数据的降维函数从第dim个维度开始展开，将后面的维度转化为一维.也就是说，只保留dim之前的维度，其他维度的数据全都挤在dim这一维。
        logit = (weight * score).sum(dim=1).unsqueeze(-1)
        return logit
