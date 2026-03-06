# -*- coding: utf-8 -*-
"""
KGFusion-DTA 主模型

Task 7a: KGFusionDTA — 替代 DAT3

改造自 DAT.py 中的 DAT3 类 (原 226 行):
  - 删除: DAT3, get_augmented_features, GraphConvolution, Attention, Attention_p
  - 新建: KGFusionDTA — 整合所有子模块

DAT3 原始结构:
  Drug-2D (GCN+VAE) + Protein-1D (BiLSTM) + Protein-2D (GraphConvolution+Attention_p)
  → 固定权重融合 (xd_att*0.5+xd*0.5) → FC[256→1024→512→1]

KGFusionDTA 新结构:
  Drug-1D (Transformer) + Drug-2D (MGCN) + Drug-KG (CNN)
  Protein-1D (BiLSTM) + Protein-2D (GINConv+VN) + Protein-KG (CNN)
  → FusionAttention (两阶段CrossAttn) → PredictionModule (DNN[1024→512→256→1])
  + InfoNCE 对比学习
"""

import torch
import torch.nn as nn

from src.models.encoders import (
    SMILESTransformerEncoder, DrugMGCN,
    ProteinSeqEncoder, ProteinGraphEncoder,
    KGEncoder
)
from src.models.fusion import FusionAttention, InfoNCELoss, PredictionModule


class KGFusionDTA(nn.Module):
    """
    KGFusion-DTA: 知识图谱增强的药物-靶标亲和力预测模型

    替代 DAT3 (DAT.py L23-154), 整合六路编码 + 两阶段融合 + 对比学习

    对比 DAT3.__init__ 签名:
      DAT3(embedding_dim, rnn_dim, hidden_dim, graph_dim, dropout_rate,
           alpha, n_heads, ...) — 大量散列参数
      KGFusionDTA(config) — 统一配置字典

    对比 DAT3.forward 签名:
      DAT3: (protein, smiles, node_drug, edge_drug, batch, vae, node_proteins, edge_proteins)
        - protein: list of (1024, 1280) ESM 嵌入
        - smiles: list of MACCS 指纹 (已废弃)
        - vae: VAE 模型 (已废弃)
      KGFusionDTA: (smiles_tokens, drug_x, drug_edge_index, drug_batch,
                    prot_esm, prot_x, prot_edge_index, prot_batch,
                    drug_kg_emb, prot_kg_emb)
        - 去掉 smiles(MACCS)/vae
        - 新增 smiles_tokens/prot_edge_index/prot_batch/drug_kg_emb/prot_kg_emb
    """
    def __init__(self, config=None):
        super(KGFusionDTA, self).__init__()

        if config is None:
            config = {}

        # --- 从 DAT3 继承并改造的编码器 ---

        # Drug 2D: 改造自 DAT3 Drug-2D (L44-51, L87-119)
        # DAT3: GCNConv(156→156/312/624) + VAE + Attention
        # 改造: GCNConv(78→78/156/312) 独立参数, 无 VAE, 无 Attention
        self.drug_2d = DrugMGCN(
            num_feature_xd=config.get('num_feature_xd', 78),
            output_dim=config.get('drug_graph_dim', 128),
            dropout=config.get('dropout', 0.2)
        )

        # Protein 1D: 改造自 DAT3 Protein-1D (L53-64, L122-128)
        # DAT3: LSTM(128,128) + view+FC(33M参数) → 128d
        # 改造: LSTM(256,256) + MaxPool → 512d
        self.prot_1d = ProteinSeqEncoder(
            esm_dim=config.get('esm_dim', 1280),
            rnn_hidden=config.get('rnn_hidden', 256),
            rnn_layers=config.get('rnn_layers', 2),
            output_dim=config.get('prot_seq_dim', 512),
            dropout=config.get('dropout', 0.2)
        )

        # Protein 2D: 改造自 DAT3 Protein-2D (L66-72, L130-139)
        # DAT3: GraphConvolution(2层) + Attention_p → 128d
        # 改造: GINConv(5层) + VirtualNode + global_mean_pool → 128d
        self.prot_2d = ProteinGraphEncoder(
            input_dim=config.get('esm_dim', 1280),
            hidden_dim=config.get('prot_graph_hidden', 128),
            num_layers=config.get('prot_graph_layers', 5),
            output_dim=config.get('prot_graph_dim', 128),
            dropout=config.get('dropout', 0.2)
        )

        # --- 新增编码器 (DAT3 中无) ---

        # Drug 1D: SMILES Transformer (参考 transformer.py, 新启用)
        self.drug_1d = SMILESTransformerEncoder(
            vocab_size=config.get('smiles_vocab_size', 63),
            max_len=config.get('max_smiles_len', 85),
            d_model=config.get('transformer_d_model', 128),
            nhead=config.get('transformer_nhead', 8),
            num_layers=config.get('transformer_layers', 2),
            dim_feedforward=config.get('transformer_ff_dim', 512),
            output_dim=config.get('drug_seq_dim', 512),
            dropout=config.get('dropout', 0.1)
        )

        # Drug KG: CNN 编码器 (来自 KGE-FUSION/models3.py)
        self.drug_kg = KGEncoder(
            input_dim=config.get('kg_emb_dim', 400),
            output_dim=config.get('kg_output_dim', 512)
        )

        # Protein KG: CNN 编码器 (来自 KGE-FUSION/models3.py, 独立实例)
        self.prot_kg = KGEncoder(
            input_dim=config.get('kg_emb_dim', 400),
            output_dim=config.get('kg_output_dim', 512)
        )

        # --- 融合模块 (替代 DAT3 的 0.5 固定权重加权) ---

        # Drug 融合: CrossAttn(seq↔graph) → CrossAttn(F_sg↔KG)
        self.drug_fusion = FusionAttention(
            seq_dim=config.get('drug_seq_dim', 512),
            graph_dim=config.get('drug_graph_dim', 128),
            kg_dim=config.get('kg_output_dim', 512),
            fused_dim=config.get('fused_dim', 512),
            nhead=config.get('fusion_nhead', 8),
            dropout=config.get('dropout', 0.1)
        )

        # Protein 融合: CrossAttn(seq↔graph) → CrossAttn(F_sg↔KG)
        self.prot_fusion = FusionAttention(
            seq_dim=config.get('prot_seq_dim', 512),
            graph_dim=config.get('prot_graph_dim', 128),
            kg_dim=config.get('kg_output_dim', 512),
            fused_dim=config.get('fused_dim', 512),
            nhead=config.get('fusion_nhead', 8),
            dropout=config.get('dropout', 0.1)
        )

        # 对比学习损失 (DAT3 中无)
        self.cl_loss_fn = InfoNCELoss(
            temperature=config.get('temperature', 0.07),
            projection_dim=config.get('cl_projection_dim', 256),
            seq_dim=config.get('drug_seq_dim', 512),
            graph_dim=config.get('drug_graph_dim', 128),
            sg_dim=config.get('fused_dim', 512),
            kg_dim=config.get('kg_output_dim', 512)
        )

        # --- 预测模块 (替代 DAT3 的 FC[256→1024→512→1]) ---
        self.predictor = PredictionModule(
            drug_fused_dim=config.get('fused_dim', 512),
            prot_fused_dim=config.get('fused_dim', 512),
            hidden_dims=config.get('predictor_hidden_dims', [512, 256]),
            dropout=config.get('dropout', 0.2),
            use_attention=config.get('use_prediction_attention', True)
        )

    def forward(self, smiles_tokens, drug_x, drug_edge_index, drug_batch,
                prot_esm, prot_x, prot_edge_index, prot_batch,
                drug_kg_emb, prot_kg_emb):
        """
        Args:
            smiles_tokens:   (B, 85)           — SMILES 整数编码 (新增, DAT3 中无)
            drug_x:          (total_drug_nodes, 78) — 批次中所有药物原子特征
            drug_edge_index: (2, total_drug_edges)  — 批次中所有药物边
            drug_batch:      (total_drug_nodes,)    — 药物节点→样本映射
            prot_esm:        (B, L, 1280)      — ESM 蛋白质嵌入 (DAT3: list → 本模块: stacked tensor)
            prot_x:          (total_prot_nodes, 1280) — 批次中蛋白质节点特征 (PyG 稀疏)
            prot_edge_index: (2, total_prot_edges)    — 批次中蛋白质边 (PyG 稀疏)
            prot_batch:      (total_prot_nodes,)      — 蛋白质节点→样本映射
            drug_kg_emb:     (B, 400)          — Drug KG 嵌入 (新增, 来自 DistMult)
            prot_kg_emb:     (B, 400)          — Protein KG 嵌入 (新增, 来自 DistMult)

        Returns:
            y_pred:  (B,)   — 预测亲和力值
            cl_loss: scalar — 对比学习损失

        对比 DAT3.forward (L87-154):
          DAT3:
            X_list = get_augmented_features(node_drug, vae)      # VAE 增强 → 已删除
            X_list = cat(X_list, node_drug) → 156d               # → 改为直接 78d
            [Drug-2D GCN → xd → Attention → xd_att]             # → DrugMGCN(无Attention)
            [Protein-1D BiLSTM → view+FC → target]               # → ProteinSeqEncoder(MaxPool)
            [Protein-2D GraphConvolution → Attention_p → node_att] # → ProteinGraphEncoder(GINConv+VN)
            drug = xd_att*0.5 + xd*0.5                          # → FusionAttention
            protein = node_att*0.5 + target*0.5                  # → FusionAttention
            xc = cat(drug, protein) → FC → out                  # → PredictionModule
            return d_block, out                                  # → return y_pred, cl_loss
        """
        # ===== 六路编码 =====

        # Drug 1D: SMILES → Transformer → 512d (DAT3 中无此路)
        f_seq_d = self.drug_1d(smiles_tokens)                           # (B, 512)

        # Drug 2D: 分子图 → MGCN → 128d (改造自 DAT3 Drug-2D)
        f_graph_d = self.drug_2d(drug_x, drug_edge_index, drug_batch)   # (B, 128)

        # Protein 1D: ESM → BiLSTM → 512d (改造自 DAT3 Protein-1D)
        f_seq_p = self.prot_1d(prot_esm)                                # (B, 512)

        # Protein 2D: 接触图 → GINConv+VN → 128d (改造自 DAT3 Protein-2D)
        f_graph_p = self.prot_2d(prot_x, prot_edge_index, prot_batch)   # (B, 128)

        # Drug KG: KG 嵌入 → CNN → 512d (新增, 来自 KGE-FUSION)
        h_kg_d = self.drug_kg(drug_kg_emb)                              # (B, 512)

        # Protein KG: KG 嵌入 → CNN → 512d (新增, 来自 KGE-FUSION)
        h_kg_p = self.prot_kg(prot_kg_emb)                              # (B, 512)

        # ===== 两阶段融合 (替代 DAT3 的 `drug = xd_att*0.5 + xd*0.5`) =====
        f_fused_d, f_sg_d = self.drug_fusion(f_seq_d, f_graph_d, h_kg_d)  # (B, 512), (B, 512)
        f_fused_p, f_sg_p = self.prot_fusion(f_seq_p, f_graph_p, h_kg_p)  # (B, 512), (B, 512)

        # ===== 预测 (替代 DAT3 的 FC[256→1024→512→1]) =====
        y_pred = self.predictor(f_fused_d, f_fused_p)  # (B,)

        # ===== 对比学习损失 (DAT3 中无此部分) =====
        cl_loss = self.cl_loss_fn(
            f_seq_d, f_graph_d, f_sg_d, h_kg_d,
            f_seq_p, f_graph_p, f_sg_p, h_kg_p
        )

        return y_pred, cl_loss
