# -*- coding: utf-8 -*-
"""
Encoders for KGFusion-DTA

合并自: drug_encoder.py + protein_encoder.py + kg_encoder.py

Drug Encoders:
  - PositionalEncoding       — 正弦位置编码
  - SMILESTransformerEncoder — Drug 1D 序列编码器 (Task 1)
  - DrugMGCN                 — Drug 2D 多跳图卷积编码器 (Task 2)

Protein Encoders:
  - ProteinSeqEncoder        — Protein 1D 序列编码器 (Task 3)
  - VirtualNode              — 虚拟节点模块
  - ProteinGraphEncoder      — Protein 2D 接触图编码器 (Task 4)

KG Encoder:
  - KGEncoder                — KG 嵌入 CNN 编码器 (Task 5)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch_sparse
from torch_geometric.nn import GCNConv, GINConv, global_max_pool as gmp, global_mean_pool
from torch_geometric.utils import to_dense_adj
from torch_scatter import scatter_mean


# ============================================================
# Drug 1D — SMILES Transformer Encoder (Task 1)
# ============================================================

class PositionalEncoding(nn.Module):
    """
    正弦位置编码，为序列中每个位置注入位置信息。
    替代 transformer.py 中 Encoder 的可学习 position_embedding。
    """
    def __init__(self, d_model=128, max_len=85, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SMILESTransformerEncoder(nn.Module):
    """
    Drug 1D 序列编码器: SMILES → Transformer Encoder → GlobalMaxPool → 512d

    对比 transformer.py 的 Encoder:
      - transformer.py Encoder: 可学习位置嵌入, 手写 TransformerBlock, 输出 trg_vocab_size 维
      - 本模块: 正弦位置编码, PyTorch 原生 TransformerEncoderLayer, 输出 512d
      - transformer.py Encoder 当前未被 DAT3 使用, 本模块将正式启用 Transformer 编码 SMILES

    词表: 复用 utils.py Smiles 类的 63 字符表
      - 有效字符 → 0~62
      - padding → 63 (padding_idx)
    """
    def __init__(self, vocab_size=63, max_len=85, d_model=128, nhead=8,
                 num_layers=2, dim_feedforward=512, output_dim=512, dropout=0.1):
        super(SMILESTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.pad_idx = vocab_size  # 63

        # 1) SMILES 词表嵌入
        self.token_embedding = nn.Embedding(
            vocab_size + 1, d_model, padding_idx=self.pad_idx
        )
        # 2) 正弦位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        # 3) Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        # 4) 投影层: d_model → output_dim
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, smiles_tokens):
        """
        Args:
            smiles_tokens: (batch, m) — 整数编码, 0~62 有效, 63 为 padding

        Returns:
            (batch, output_dim) — 药物序列特征 f_seq^D
        """
        # 生成 padding mask: True 表示需要被屏蔽的位置
        src_key_padding_mask = (smiles_tokens == self.pad_idx)  # (B, m)

        # 嵌入 + 缩放 + 位置编码
        x = self.token_embedding(smiles_tokens) * math.sqrt(self.d_model)  # (B, m, d_model)
        x = self.pos_encoder(x)                                             # (B, m, d_model)

        # Transformer 编码
        x = self.transformer_encoder(
            x, src_key_padding_mask=src_key_padding_mask
        )  # (B, m, d_model)

        # GlobalMaxPool: 屏蔽 padding 位后取最大值
        mask = ~src_key_padding_mask  # (B, m), True 表示有效位置
        mask = mask.unsqueeze(-1)     # (B, m, 1)
        x = x.masked_fill(~mask, float('-inf'))
        x = x.max(dim=1)[0]          # (B, d_model)

        # 投影到 output_dim
        x = self.fc_out(x)           # (B, output_dim)
        return x


# ============================================================
# Drug 2D — MGCN 多跳图卷积编码器 (Task 2)
# ============================================================

class DrugMGCN(nn.Module):
    """
    Drug 2D 多跳 GCN 编码器: 分子图 → 多尺度 GCN(1/2/3-hop) → MaxPool → 128d

    改造自 DAT.py DAT3 的 Drug-2D 分支 (L44-119):
      (1) 删除 VAE 增强: num_feature_xd 从 156→78 (原始原子特征维度)
      (2) 每条路径使用独立参数 (DAT3 中 3 条路径共享 Conv1/2/3)
      (3) 删除 Attention 加权 (DAT3: xd_att*0.5 + xd*0.5), 融合由后续 FusionAttention 负责
      (4) concat_dim: 1092→546, FC: (1092→1024→128) → (546→512→128)

    输入: 药物分子图 (node_features, edge_index, batch) — 来自 process_smiles.smile_to_graph
    输出: f_graph^D ∈ R^{batch × 128}
    """
    def __init__(self, num_feature_xd=78, output_dim=128, dropout=0.2):
        super(DrugMGCN, self).__init__()

        # ===== 1-hop 路径: 3 层 GCNConv =====
        self.conv1_1 = GCNConv(num_feature_xd, num_feature_xd)
        self.conv1_2 = GCNConv(num_feature_xd, num_feature_xd * 2)
        self.conv1_3 = GCNConv(num_feature_xd * 2, num_feature_xd * 4)

        # ===== 2-hop 路径: 2 层 GCNConv =====
        self.conv2_1 = GCNConv(num_feature_xd, num_feature_xd)
        self.conv2_2 = GCNConv(num_feature_xd, num_feature_xd * 2)

        # ===== 3-hop 路径: 1 层 GCNConv =====
        self.conv3_1 = GCNConv(num_feature_xd, num_feature_xd)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # concat 维度: 312 + 156 + 78 = 546
        concat_dim = num_feature_xd * 4 + num_feature_xd * 2 + num_feature_xd  # 546
        self.fc_g1 = nn.Linear(concat_dim, 512)
        self.fc_g2 = nn.Linear(512, output_dim)  # → 128

    def forward(self, x, edge_index, batch):
        """
        Args:
            x:          (total_nodes, 78) — 批次中所有分子的原子特征 (拼接大图)
            edge_index: (2, total_edges)  — 批次中所有边 (已做偏移)
            batch:      (total_nodes,)    — 每个节点所属的样本索引

        Returns:
            (B, output_dim) — 药物图特征 f_graph^D
        """
        # --- 计算多跳邻接 ---
        adj = to_dense_adj(edge_index)
        num_nodes = adj.shape[1]

        # A² = A × A (2-hop 邻居)
        edge_index_sq, _ = torch_sparse.spspmm(
            edge_index, None, edge_index, None,
            num_nodes, num_nodes, num_nodes, coalesced=True
        )
        # A³ = A² × A (3-hop 邻居)
        edge_index_cb, _ = torch_sparse.spspmm(
            edge_index_sq, None, edge_index, None,
            num_nodes, num_nodes, num_nodes, coalesced=True
        )

        # --- 1-hop 路径 ---
        h1 = self.relu(self.conv1_1(x, edge_index))          # (N, 78)
        h1 = self.relu(self.conv1_2(h1, edge_index))         # (N, 156)
        h1 = self.relu(self.conv1_3(h1, edge_index))         # (N, 312)

        # --- 2-hop 路径 ---
        h2 = self.relu(self.conv2_1(x, edge_index_sq))       # (N, 78)
        h2 = self.relu(self.conv2_2(h2, edge_index_sq))      # (N, 156)

        # --- 3-hop 路径 ---
        h3 = self.relu(self.conv3_1(x, edge_index_cb))       # (N, 78)

        # --- 拼接 + 池化 ---
        concat = torch.cat([h1, h2, h3], dim=1)              # (N, 546)
        xd = gmp(concat, batch)                               # (B, 546)

        # --- FC 层 ---
        xd = self.relu(self.fc_g1(xd))                       # (B, 512)
        xd = self.dropout(xd)
        xd = self.fc_g2(xd)                                  # (B, 128)
        xd = self.dropout(xd)

        return xd


# ============================================================
# Protein 1D — ESM + BiLSTM Encoder (Task 3)
# ============================================================

class ProteinSeqEncoder(nn.Module):
    """
    Protein 1D 序列编码器: ESM嵌入 → FC降维 → BiLSTM → MaxPool → 512d

    改造自 DAT.py DAT3 的 Protein-1D 分支:
      DAT3: LSTM(128,128) + view+FC(33M参数) → 128d
      改造: LSTM(256,256) + MaxPool → 512d
    """
    def __init__(self, esm_dim=1280, rnn_hidden=256, rnn_layers=2,
                 output_dim=512, dropout=0.2):
        super(ProteinSeqEncoder, self).__init__()

        self.input_fc = nn.Linear(esm_dim, rnn_hidden)

        self.bilstm = nn.LSTM(
            input_size=rnn_hidden,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.fc_out = nn.Linear(rnn_hidden * 2, output_dim)  # 双向拼接: 256*2=512

    def forward(self, prot_esm):
        """
        Args:
            prot_esm: (batch, L, 1280) — ESM 预训练蛋白质嵌入

        Returns:
            (batch, output_dim) — 蛋白质序列特征 f_seq^P
        """
        x = self.input_fc(prot_esm)
        x, _ = self.bilstm(x)
        x = x.max(dim=1)[0]
        x = self.fc_out(x)
        return x


# ============================================================
# Protein 2D — GINConv + VirtualNode Graph Encoder (Task 4)
# ============================================================

class VirtualNode(nn.Module):
    """
    虚拟节点: 在每一层 GNN 之前, 将虚拟节点特征广播给所有真实节点,
    并在之后收集所有真实节点特征的均值来更新虚拟节点。

    DAT3 中无此机制 — DAT3 使用 Attention_p 做多头加权聚合。
    本模块替代 Attention_p, 提供更标准的全局信息传递方式。
    """
    def __init__(self, hidden_dim):
        super(VirtualNode, self).__init__()
        self.vn_embedding = nn.Embedding(1, hidden_dim)
        nn.init.constant_(self.vn_embedding.weight.data, 0)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

    def forward(self, x, batch):
        """
        Args:
            x: (N, hidden_dim) — 所有节点特征
            batch: (N,) — 每个节点所属的图索引

        Returns:
            vn_feat: (B, hidden_dim) — 每个图的虚拟节点特征(更新后)
        """
        batch_size = batch.max().item() + 1
        vn = self.vn_embedding(torch.zeros(batch_size, dtype=torch.long, device=x.device))

        x = x + vn[batch]

        vn_agg = scatter_mean(x, batch, dim=0, dim_size=batch_size)

        vn = vn + self.mlp(vn_agg)

        return x, vn


class ProteinGraphEncoder(nn.Module):
    """
    Protein 2D 接触图编码器: 蛋白质图 → GINConv → VirtualNode → global_mean_pool → 128d

    改造自 DAT.py DAT3 的 Protein-2D 分支:
      DAT3: GraphConvolution(2层) + Attention_p → 128d
      改造: GINConv(5层) + VirtualNode + global_mean_pool → 128d
    """
    def __init__(self, input_dim=1280, hidden_dim=128, num_layers=5,
                 output_dim=128, dropout=0.2):
        super(ProteinGraphEncoder, self).__init__()

        self.num_layers = num_layers

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.gin_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.gin_layers.append(GINConv(mlp, train_eps=True))
            self.bn_layers.append(nn.BatchNorm1d(hidden_dim))

        self.virtual_node = VirtualNode(hidden_dim)

        self.dropout = nn.Dropout(dropout)

        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        """
        Args:
            x:          (N, 1280) — 批次中所有蛋白质节点的 ESM 特征
            edge_index: (2, E)    — 批次中所有边 (稀疏 PyG 格式)
            batch:      (N,)      — 每个节点所属的样本索引

        Returns:
            (B, output_dim) — 蛋白质图特征 f_graph^P
        """
        x = self.input_proj(x)

        for i in range(self.num_layers):
            h = self.gin_layers[i](x, edge_index)
            h = self.bn_layers[i](h)
            h = F.relu(h)
            h = self.dropout(h)
            x = x + h

        x, vn = self.virtual_node(x, batch)

        x = global_mean_pool(x, batch)

        x = self.fc_out(x)
        return x


# ============================================================
# KG 嵌入 CNN 编码器 (Task 5)
# ============================================================

class KGEncoder(nn.Module):
    """
    KG 嵌入 CNN 编码器: 预训练 KG 实体嵌入 → 多层 1D CNN → AdaptiveMaxPool → FC → 512d

    改造自 KGE-FUSION/models3.py 中 KGE_UNIT 的 hf_dti_cnn 部分:
      (1) 输入: 800d (drug+protein 拼接) → 400d (单个实体)
      (2) Drug 和 Protein 各用一个独立的 KGEncoder 实例
      (3) BN 改为 __init__ 中预定义
      (4) 输出: AdaptiveMaxPool1d+FC→512d
    """
    def __init__(self, input_dim=400, output_dim=512):
        super(KGEncoder, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, output_dim)

    def forward(self, kg_emb):
        """
        Args:
            kg_emb: (batch, 400) — 预训练 KG 实体嵌入

        Returns:
            (batch, output_dim) — KG 编码特征 h_KG
        """
        x = kg_emb.unsqueeze(1)

        x = F.relu(self.bn1(self.conv1(x)))  # (B, 32, 400)
        x = F.relu(self.bn2(self.conv2(x)))  # (B, 64, 400)
        x = F.relu(self.bn3(self.conv3(x)))  # (B, 128, 400)

        x = self.global_pool(x).squeeze(-1)  # (B, 128)

        x = self.fc(x)  # (B, 512)
        return x
