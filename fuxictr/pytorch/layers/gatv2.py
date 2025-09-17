import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product



class FiGATv2_Layer(nn.Module):
    def __init__(self,
                 num_fields,
                 embedding_dim,
                 gat_layers=3,
                 reuse_graph_layer=False,
                 use_gru=True,
                 use_residual=True,
                 device=None):
        super(FiGATv2_Layer, self).__init__()
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.gat_layers = gat_layers
        self.use_residual = use_residual
        self.reuse_graph_layer = reuse_graph_layer
        self.device = device
        if reuse_graph_layer:
            self.gat = GraphAttentionLayer(num_fields, embedding_dim)
        else:
            self.gat = nn.ModuleList([GraphAttentionLayer(num_fields, embedding_dim)
                                     for _ in range(gat_layers)])
        self.gru = nn.GRUCell(embedding_dim, embedding_dim) if use_gru else None
        self.src_nodes, self.dst_nodes = zip(*list(product(range(num_fields), repeat=2)))   # source，destination
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.W_attn = nn.Linear(embedding_dim * 2, 1, bias=False)  # nn.Linear()用于设置神经网络全连接层

    def build_graph(self, feature_emb):
        src_emb = feature_emb[:, self.src_nodes, :]
        dst_emb = feature_emb[:, self.dst_nodes, :]
        concat_emb = torch.cat([src_emb, dst_emb], dim=-1)  # 对embedding层连接
        alpha = self.W_attn(concat_emb)  # self.leaky_relu(self.W_attn(concat_emb))
        alpha = alpha.view(-1, self.num_fields, self.num_fields)
        mask = torch.eye(self.num_fields).to(self.device)
        alpha = alpha.masked_fill(mask.bool(), float('-inf'))  # mask.byte与mask.bool 将对角线元素全部变成无穷
        graph = F.softmax(alpha, dim=-1)  # batch x field x field without self-loops
        return graph

    def forward(self, feature_emb):
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        g = self.build_graph(feature_emb)
        h = feature_emb
        # if feature_emb.shape[0] != 1000:     # batch_size
            # s = torch.zeros([1000 - feature_emb.shape[0], feature_emb.shape[1], feature_emb.shape[2]])
            # t = torch.zeros([1000 - g.shape[0], g.shape[1], g.shape[2]])
            # h = h.to(device)
            # s = s.to(device)
            # g = g.to(device)
            # t = t.to(device)
            # feature_emb = feature_emb.to(device)
            # h = torch.cat([h, s], dim=0)
            # g = torch.cat([g, t], dim=0)
            # feature_emb = torch.cat([feature_emb, s], dim=0)
        # h = h.reshape(feature_emb.shape[0], feature_emb.shape[1], feature_emb.shape[1])
        for i in range(self.gat_layers):
            if self.reuse_graph_layer:
                a = self.gat(h, g, feature_emb.shape[0], feature_emb.shape[2])  # a = self.gat(h, g)

            else:
                a = self.gat[i](h, g, feature_emb.shape[0], feature_emb.shape[2])  # a = self.gat[i](h, g)
            if self.gru is not None:   # 门控循环单元 (Gated Recurrent Unit，GRU) 是一种常用的GRNN（门控循环神经网络）
                a = a.view(-1, self.embedding_dim)
                h = h.view(-1, self.embedding_dim)
                h = self.gru(h, a)
                h = h.view(-1, self.num_fields, self.embedding_dim)
            else:
                h = a + h
            if self.use_residual:
                h += feature_emb
        return h


# class GAT(nn.Module):
#     def __init__(self, n_feat, n_hid,  n_heads, dropout=0.5, alpha=0.01):
#         """Dense version of GAT
#         n_heads 表示有几个GAL层，最后进行拼接在一起，类似self-attention
#         从不同的子空间进行抽取特征。
#         """
#         super(GAT, self).__init__()
#         self.dropout = dropout
#
#         # 定义multi-head的图注意力层
#         self.attentions = [GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True)
#                            for _ in range(n_heads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)  # 加入pytorch的Module模块
#         # 输出层，也通过图注意力层来实现，可实现分类、预测等功能
#         self.out_att = GraphAttentionLayer(n_hid * n_heads, n_hid * n_heads, dropout=dropout, alpha=alpha, concat=False)
#
#     def forward(self, x, adj):
#         x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=2)  # 将每个head得到的表示进行拼接
#         x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
#         x = F.elu(self.out_att(x, adj))  # 输出并激活
#         return F.log_softmax(x, dim=1)  # log_softmax速度变快，保持数值稳定


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    图注意力层
    """

    def __init__(self, in_features, out_features, dropout=0.5, alpha=0.01, concat=True):
        # 这里要设定一下batch_size
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features  # 节点表示向量的输入特征维度
        self.out_features = out_features  # 节点表示向量的输出特征维度
        self.dropout = dropout  # dropout参数
        self.alpha = alpha  # leakyrelu激活的参数
        self.concat = concat  # 如果为true, 再进行elu激活



        # 定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj, batch_size, out_features):
        """
        inp: input_fea [N, in_features]  in_features表示节点的输入特征向量元素个数
        adj: 图的邻接矩阵 维度[N, N] 非零即一，数据结构基本知识
        """
        # 定义可训练参数，即论文中的W和a
        W = nn.Parameter(torch.zeros(size=(batch_size, inp.shape[2], inp.shape[2])))  # out_features
        nn.init.xavier_uniform_(W.data, gain=1.414)  # xavier初始化
        a = nn.Parameter(torch.zeros(size=(int(out_features/2)*inp.shape[1]*inp.shape[1], 1)))
        nn.init.xavier_uniform_(a.data, gain=1.414)  # xavier初始化
        # 令变量都在gpu上进行运算
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        inp = inp.to(device)
        W = W.to(device)
        a = a.to(device)
        adj = adj.to(device)

        # h = torch.matmul(inp, W)  # [N, out_features]or[batch_size, N, out_features]
        h = inp
        N = h.size()[1]  # N 图的节点数

        # a_input = torch.cat([h.repeat(1, 1, N).view(-1, N * N, self.out_features), h.repeat(1, N, 1)], dim=-1).view(-1, N, N, 2*self.out_features)
        # # [batch_size, N, N, 2*out_features]
        a_input = torch.cat([h.repeat(1, 1, N).view(-1, N * N, self.out_features), h.repeat(1, N, 1)], dim=-1)#.view(-1,
                                                                                                                    #N,
                                                                                                                    #N,
                                                                                                                    #2*self.out_features)
        W = W.view(-1, self.out_features*2, int((self.out_features/2)))
        a_input = torch.matmul(a_input, W).view(-1, N, N*int(self.out_features/2), 1)
        a_input = a_input.to(device)
        # e = self.leakyrelu(torch.matmul(a_input, a).squeeze(3))
        # # squeeze: [batch_size, N, N, 1] => [batch_size, N, N] 图注意力的相关系数（未归一化）
        a = a.view(-1, 24)
        e = torch.matmul(self.leakyrelu(a_input.squeeze(3)), a)

        zero_vec = -1e12 * torch.ones_like(e)  # 将没有连接的边置为负无穷
        attention = torch.where(adj > 0, e, zero_vec)  # [batch_size, N, N]
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        attention = F.softmax(attention, dim=1)  # softmax形状保持不变 [batch_size, N, N]，得到归一化的注意力权重！
        attention = F.dropout(attention, self.dropout, training=self.training)  # dropout，防止过拟合
        h_prime = torch.matmul(attention, h)
        # [batch_size, N, N].[batch_size, N, out_features] => [batch_size, N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    # def __repr__(self):
    #     return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# class GAT_NET(torch.nn.Module):
#     def __init__(self, features, hidden, classes, heads=4):
#         super(GAT_NET, self).__init__()
#         self.gat1 = GATConv(features, hidden, heads=4)  # 定义GAT层，使用多头注意力机制
#         self.gat2 = GATConv(hidden*heads, classes)  # 因为多头注意力是将向量拼接，所以维度乘以头数。
#
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#
#         x = self.gat1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.gat2(x, edge_index)
#
#         return F.log_softmax(x, dim=1)