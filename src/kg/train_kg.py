# -*- coding: utf-8 -*-
"""
KG Embedding Pre-training for KGFusion-DTA

Task 5b: KG 嵌入预训练脚本
  来源: 改造自 KGE-FUSION/train2.py L260-302 的 PyKEEN pipeline 部分
  功能: 使用 DistMult 在合并的 KG 三元组上训练, 提取实体嵌入

改造说明:
  train2.py 原始:
    - 模型: ConvE (KGE_NAME = 'conve')
    - 数据: 单个 kg_all.txt, 按 fold 拆分训练/测试
    - 嵌入维度: 400
    - 训练: pipeline(training=..., testing=..., model='conve', ...)
    - 提取: result.entity_representations[0](indices=None) → 全部实体嵌入
    - 使用: get_features_hf() 拼接 (drug+protein) → 800d

  本模块改造:
    - 模型: ConvE → DistMult (更轻量, 适合大规模 KG)
    - 数据: 合并三个 KG (Hetionet + BioKG + Yamanishi08) 的三元组
    - 提取: 分离返回 drug → 400d, protein → 400d (不拼接)
    - 封装为独立函数, 支持跳过已训练的模型
"""

import os
import torch
import numpy as np


def train_distmult(triples_path, save_dir, embedding_dim=400, epochs=150,
                   batch_size=1024, device=None):
    """
    使用 PyKEEN 训练 DistMult KG 嵌入模型。

    来源: KGE-FUSION/train2.py L282-292
      原始代码:
        result = pipeline(
            training=training,
            testing=testing,
            model='conve',                                    # ← 改为 DistMult
            device=torch.device("cuda" if ... else "cpu"),
            model_kwargs=dict(embedding_dim=400),
            training_kwargs=dict(num_epochs=150, batch_size=1024)
        )
        result.save_to_directory(save_path)

    Args:
        triples_path: str — 合并后的 KG 三元组文件路径 (TSV: head\\trelation\\ttail)
        save_dir: str — 模型保存目录
        embedding_dim: int — 实体/关系嵌入维度 (train2.py 默认 400)
        epochs: int — 训练轮数 (train2.py 默认 150)
        batch_size: int — 批大小 (train2.py 默认 1024)
        device: torch.device — 计算设备

    Returns:
        result — PyKEEN PipelineResult 对象
    """
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training = TriplesFactory.from_path(triples_path)

    # train2.py 原始: model='conve'
    # 改造: model='DistMult' — 更轻量, 训练更快
    result = pipeline(
        training=training,
        model='DistMult',
        device=device,
        model_kwargs=dict(embedding_dim=embedding_dim),
        training_kwargs=dict(num_epochs=epochs, batch_size=batch_size),
    )

    os.makedirs(save_dir, exist_ok=True)
    result.save_to_directory(save_dir)
    print(f"KG model saved to {save_dir}")

    return result


def load_kg_model(model_path, device=None):
    """
    加载已训练的 KG 模型。

    来源: KGE-FUSION/train2.py L297-298
      原始代码:
        result = torch.load(model_path, map_location=...)
        en_re = result.entity_representations[0](indices=None)

    Args:
        model_path: str — 已训练模型的 .pkl 文件路径
        device: torch.device

    Returns:
        result — 加载的 PyKEEN 模型结果对象
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.load(model_path, map_location=device)
    return result


def extract_entity_embeddings(result):
    """
    从训练好的 KG 模型中提取所有实体嵌入。

    来源: KGE-FUSION/train2.py L299
      原始代码:
        en_re = result.entity_representations[0](indices=None)

    Args:
        result — PyKEEN 模型结果对象 (PipelineResult 或加载的 trained_model.pkl)

    Returns:
        embeddings: torch.Tensor — (num_entities, embedding_dim) 所有实体嵌入
        entity_to_id: dict — 实体名称 → 实体索引的映射
    """
    # train2.py L299: en_re = result.entity_representations[0](indices=None)
    embeddings = result.entity_representations[0](indices=None)
    entity_to_id = result.training.entity_to_id
    return embeddings, entity_to_id


def extract_drug_protein_embeddings(result, drug_ids, protein_ids):
    """
    从 KG 嵌入中分离提取 drug 和 protein 的嵌入。

    来源: 改造自 KGE-FUSION/train2.py 的 get_features_hf() (L220-232)
      原始: get_features_hf 拼接 (drug+protein) → 800d
      改造: 分离返回 drug → 400d, protein → 400d

    train2.py get_features_hf 原始:
      for i in range(length):
          head1 = data['head'][i]
          head2 = data['tail'][i]
          idx1 = ent_id[head1]
          idx2 = ent_id[head2]
          f1 = embed[idx1]    # drug 嵌入
          f2 = embed[idx2]    # protein 嵌入
          f = concat(f1, f2)  # → 800d

    本函数改造:
      分别收集 drug 和 protein 的嵌入, 不做拼接

    Args:
        result — PyKEEN 模型结果对象
        drug_ids: list[str] — drug 在 KG 中的实体名称列表
        protein_ids: list[str] — protein 在 KG 中的实体名称列表

    Returns:
        drug_embeddings: np.ndarray — (num_drugs, embedding_dim)
        protein_embeddings: np.ndarray — (num_proteins, embedding_dim)
    """
    embeddings, entity_to_id = extract_entity_embeddings(result)
    embeddings_np = embeddings.cpu().detach().numpy()

    drug_embs = []
    for drug_id in drug_ids:
        if drug_id in entity_to_id:
            idx = entity_to_id[drug_id]
            drug_embs.append(embeddings_np[idx])
        else:
            # 未在 KG 中找到的实体, 使用零向量
            drug_embs.append(np.zeros(embeddings_np.shape[1], dtype=np.float32))
    drug_embeddings = np.array(drug_embs)

    prot_embs = []
    for prot_id in protein_ids:
        if prot_id in entity_to_id:
            idx = entity_to_id[prot_id]
            prot_embs.append(embeddings_np[idx])
        else:
            prot_embs.append(np.zeros(embeddings_np.shape[1], dtype=np.float32))
    protein_embeddings = np.array(prot_embs)

    return drug_embeddings, protein_embeddings


def train_or_load(triples_path, save_dir, embedding_dim=400, epochs=150,
                  batch_size=1024, device=None):
    """
    训练或加载 KG 嵌入模型 (带缓存机制)。

    来源: KGE-FUSION/train2.py L280-298 的训练/加载逻辑
      原始代码:
        model_path = os.path.join(save_path, 'trained_model.pkl')
        if not os.path.exists(model_path):
            result = pipeline(...)
            result.save_to_directory(save_path)
        result = torch.load(model_path, ...)

    Args:
        triples_path: str — KG 三元组文件路径
        save_dir: str — 模型保存目录
        其他参数同 train_distmult

    Returns:
        result — PyKEEN 模型结果对象
    """
    model_path = os.path.join(save_dir, 'trained_model.pkl')

    if not os.path.exists(model_path):
        print(f"KG model not found at {model_path}, training...")
        result = train_distmult(triples_path, save_dir, embedding_dim,
                                epochs, batch_size, device)
    else:
        print(f"Loading existing KG model from {model_path}")
        result = load_kg_model(model_path, device)

    return result
