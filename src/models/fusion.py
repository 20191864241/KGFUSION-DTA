# -*- coding: utf-8 -*-
"""
Fusion, Contrastive Learning, and Prediction Modules for KGFusion-DTA

合并自: fusion.py (原) + prediction.py + layers.py 中的 MultiHeadCrossAttention

Attention:
  - MultiHeadCrossAttention  — 多头交叉注意力 (Task 6a, 原在 layers.py)
  - MultiHeadLinearAttention  — 多头线性注意力特征精炼 (Task 6c, 原在 prediction.py)

Fusion:
  - FusionAttention           — 两阶段交叉注意力融合 (Task 6b)
  - InfoNCELoss               — 对比学习损失 (Task 6b)

Prediction:
  - PredictionModule          — 亲和力预测头 (Task 6c)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# MultiHeadCrossAttention — 多头交叉注意力 (Task 6a)
# ============================================================

class MultiHeadCrossAttention(nn.Module):
    """
    多头交叉注意力: X_a (Query) attend to X_b (Key/Value)

    论文公式: CrossAttn(X_a, X_b) = softmax(QK^T / √d_k) V

    与 LinkAttention 的关系:
      - LinkAttention: 自注意力 (Q=K=V 来自同一输入), 用于 DAT3/KGE_UNIT
      - MultiHeadCrossAttention: 交叉注意力 (Q 来自 X_a, K/V 来自 X_b),
        用于 KGFusionDTA 的 FusionAttention 融合模块
    """
    def __init__(self, d_model=512, nhead=8, dropout=0.1):
        super(MultiHeadCrossAttention, self).__init__()
        assert d_model % nhead == 0, f"d_model({d_model}) must be divisible by nhead({nhead})"

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        # Q 来自 x_a, K/V 来自 x_b
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model)

        # LayerNorm + Dropout (残差连接)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_a, x_b):
        """
        Args:
            x_a: (batch, d_model) — Query 来源 (如 f_seq 或 F_sg)
            x_b: (batch, d_model) — Key/Value 来源 (如 f_graph 或 h_KG)

        Returns:
            (batch, d_model) — 交叉注意力输出 + 残差连接
        """
        B = x_a.size(0)

        # 线性投影 + reshape 为多头
        Q = self.W_q(x_a).view(B, 1, self.nhead, self.d_k).transpose(1, 2)  # (B, nhead, 1, d_k)
        K = self.W_k(x_b).view(B, 1, self.nhead, self.d_k).transpose(1, 2)  # (B, nhead, 1, d_k)
        V = self.W_v(x_b).view(B, 1, self.nhead, self.d_k).transpose(1, 2)  # (B, nhead, 1, d_k)

        # 注意力分数: (B, nhead, 1, 1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和: (B, nhead, 1, d_k)
        attn_output = torch.matmul(attn_weights, V)

        # 拼接多头: (B, nhead, 1, d_k) → (B, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, self.d_model)

        # 输出投影 + 残差 + LayerNorm
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        output = self.layer_norm(output + x_a)

        return output


# ============================================================
# FusionAttention — 两阶段交叉注意力融合 (Task 6b)
# ============================================================

class FusionAttention(nn.Module):
    """
    两阶段 CrossAttention 融合模块:

    替代 DAT3 的简单加权融合:
      DAT3: drug = xd_att * 0.5 + xd * 0.5 — 固定权重加权

    本模块:
      阶段1: F_sg = CrossAttn(f_seq, f_graph) — 序列↔图对齐融合
      阶段2: F_fused = CrossAttn(F_sg, h_KG)  — 注入 KG 信息

    Drug 和 Protein 各用一个独立的 FusionAttention 实例。
    """
    def __init__(self, seq_dim=512, graph_dim=128, kg_dim=512,
                 fused_dim=512, nhead=8, dropout=0.1):
        super(FusionAttention, self).__init__()

        # --- 维度对齐 ---
        self.seq_proj = nn.Linear(seq_dim, fused_dim)
        self.graph_proj = nn.Linear(graph_dim, fused_dim)
        self.kg_proj = nn.Linear(kg_dim, fused_dim)

        # --- 阶段1: seq ↔ graph 交叉注意力 ---
        self.cross_attn_sg = MultiHeadCrossAttention(
            d_model=fused_dim, nhead=nhead, dropout=dropout
        )

        # --- 阶段2: F_sg ↔ KG 交叉注意力 ---
        self.cross_attn_kg = MultiHeadCrossAttention(
            d_model=fused_dim, nhead=nhead, dropout=dropout
        )

    def forward(self, f_seq, f_graph, h_kg):
        """
        Args:
            f_seq:   (batch, seq_dim)   — 序列编码特征
            f_graph: (batch, graph_dim) — 图编码特征
            h_kg:    (batch, kg_dim)    — KG 编码特征

        Returns:
            f_fused: (batch, fused_dim) — 融合后特征 F_fused
            f_sg:    (batch, fused_dim) — 阶段1中间特征 F_sg (用于对比学习)
        """
        # 维度对齐
        f_seq_proj = self.seq_proj(f_seq)       # (B, fused_dim)
        f_graph_proj = self.graph_proj(f_graph) # (B, fused_dim)
        h_kg_proj = self.kg_proj(h_kg)          # (B, fused_dim)

        # 阶段1: 序列 attend to 图
        f_sg = self.cross_attn_sg(f_seq_proj, f_graph_proj)  # (B, fused_dim)

        # 阶段2: 对齐特征 attend to KG
        f_fused = self.cross_attn_kg(f_sg, h_kg_proj)        # (B, fused_dim)

        return f_fused, f_sg


# ============================================================
# InfoNCELoss — 对比学习损失 (Task 6b)
# ============================================================

class InfoNCELoss(nn.Module):
    """
    InfoNCE 对比学习损失 (DAT3 中无此部分, 为 KGFusionDTA 新增)

    两级对比:
      Level 1: (f_seq ↔ f_graph)  — 同一实体的 1D 和 2D 表示应相似
      Level 2: (F_sg ↔ h_KG)     — 融合后的结构特征应与 KG 嵌入对齐

    L_CL = -log( exp(sim(z_i, z_j+) / τ) / Σ_k exp(sim(z_i, z_k) / τ) )
    """
    def __init__(self, temperature=0.07, projection_dim=256,
                 seq_dim=512, graph_dim=128, sg_dim=512, kg_dim=512):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

        # 投影头: 将不同维度的特征映射到统一的对比空间
        self.proj_seq = nn.Sequential(
            nn.Linear(seq_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        self.proj_graph = nn.Sequential(
            nn.Linear(graph_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        self.proj_sg = nn.Sequential(
            nn.Linear(sg_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        self.proj_kg = nn.Sequential(
            nn.Linear(kg_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def _infonce(self, z_a, z_b):
        """
        计算 InfoNCE 损失: 同一样本的 (z_a[i], z_b[i]) 为正对,
        batch 内其他样本为负对。
        """
        z_a = F.normalize(z_a, dim=-1)
        z_b = F.normalize(z_b, dim=-1)

        logits = torch.matmul(z_a, z_b.T) / self.temperature

        B = z_a.size(0)
        labels = torch.arange(B, device=z_a.device)

        loss_ab = F.cross_entropy(logits, labels)
        loss_ba = F.cross_entropy(logits.T, labels)

        return (loss_ab + loss_ba) / 2.0

    def forward(self, f_seq_d, f_graph_d, f_sg_d, h_kg_d,
                f_seq_p, f_graph_p, f_sg_p, h_kg_p):
        """
        Args:
            f_seq_d/f_graph_d/f_sg_d/h_kg_d:   Drug 的四路特征
            f_seq_p/f_graph_p/f_sg_p/h_kg_p:   Protein 的四路特征

        Returns:
            cl_loss: scalar — 总对比学习损失
        """
        # Level 1: seq ↔ graph (Drug + Protein)
        z_seq_d = self.proj_seq(f_seq_d)
        z_graph_d = self.proj_graph(f_graph_d)
        loss_sg_d = self._infonce(z_seq_d, z_graph_d)

        z_seq_p = self.proj_seq(f_seq_p)
        z_graph_p = self.proj_graph(f_graph_p)
        loss_sg_p = self._infonce(z_seq_p, z_graph_p)

        # Level 2: F_sg ↔ KG (Drug + Protein)
        z_sg_d = self.proj_sg(f_sg_d)
        z_kg_d = self.proj_kg(h_kg_d)
        loss_kg_d = self._infonce(z_sg_d, z_kg_d)

        z_sg_p = self.proj_sg(f_sg_p)
        z_kg_p = self.proj_kg(h_kg_p)
        loss_kg_p = self._infonce(z_sg_p, z_kg_p)

        # 总损失: 四项均等加权
        cl_loss = (loss_sg_d + loss_sg_p + loss_kg_d + loss_kg_p) / 4.0

        return cl_loss


# ============================================================
# MultiHeadLinearAttention — 多头线性注意力 (Task 6c)
# ============================================================

class MultiHeadLinearAttention(nn.Module):
    """
    多头线性注意力: 对特征做加权精炼 → 单向量

    DAT3 中无此模块。用于对融合后的 Drug/Protein 特征做自适应加权聚合,
    增加模型对关键特征维度的选择能力。
    """
    def __init__(self, input_dim=512, n_heads=4, dropout=0.1):
        super(MultiHeadLinearAttention, self).__init__()
        assert input_dim % n_heads == 0, \
            f"input_dim({input_dim}) must be divisible by n_heads({n_heads})"

        self.input_dim = input_dim
        self.n_heads = n_heads
        self.head_dim = input_dim // n_heads

        self.W_q = nn.Linear(input_dim, input_dim)
        self.W_k = nn.Linear(input_dim, input_dim)

        self.out_proj = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        """
        Args:
            x: (batch, input_dim) — 输入特征向量

        Returns:
            (batch, input_dim) — 加权精炼后的特征
        """
        B = x.size(0)

        Q = self.W_q(x).view(B, self.n_heads, self.head_dim)
        K = self.W_k(x).view(B, self.n_heads, self.head_dim)

        attn_scores = (Q * K) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        x_reshaped = x.view(B, self.n_heads, self.head_dim)
        out = attn_weights * x_reshaped

        out = out.contiguous().view(B, self.input_dim)
        out = self.out_proj(out)
        out = self.dropout(out)

        out = self.layer_norm(out + x)
        return out


# ============================================================
# PredictionModule — 亲和力预测头 (Task 6c)
# ============================================================

class PredictionModule(nn.Module):
    """
    亲和力预测模块: concat(F_fused^D, F_fused^P) → DNN → 亲和力值

    改造自 DAT.py DAT3 的 FC 预测头:
      DAT3: [256→512→256→1], 无 BN
      改造: [1024→512→256→1], 加 BN + 可选注意力精炼
    """
    def __init__(self, drug_fused_dim=512, prot_fused_dim=512,
                 hidden_dims=None, dropout=0.2, use_attention=True):
        super(PredictionModule, self).__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256]

        input_dim = drug_fused_dim + prot_fused_dim  # 1024

        # 可选: 多头线性注意力精炼
        self.use_attention = use_attention
        if use_attention:
            self.drug_attn = MultiHeadLinearAttention(drug_fused_dim, n_heads=4)
            self.prot_attn = MultiHeadLinearAttention(prot_fused_dim, n_heads=4)

        # DNN 预测头
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.dnn = nn.Sequential(*layers)

    def forward(self, f_fused_drug, f_fused_prot):
        """
        Args:
            f_fused_drug: (batch, drug_fused_dim) — Drug 融合特征
            f_fused_prot: (batch, prot_fused_dim) — Protein 融合特征

        Returns:
            pred: (batch,) — 预测的亲和力值
        """
        if self.use_attention:
            f_fused_drug = self.drug_attn(f_fused_drug)
            f_fused_prot = self.prot_attn(f_fused_prot)

        xc = torch.cat([f_fused_drug, f_fused_prot], dim=1)

        pred = self.dnn(xc).squeeze(-1)

        return pred
