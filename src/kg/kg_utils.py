# -*- coding: utf-8 -*-
"""
KG Data Utilities for KGFusion-DTA

Task 5c: KG 数据加载、合并与实体映射
  来源: 改造自 KGE-FUSION/train2.py L76-126 + train_BioKG.py L76-127 的数据加载逻辑
  功能: 加载并合并多个 KG, 建立 drug/protein → KG 实体的映射

改造说明:
  train2.py 原始:
    - 加载单个 kg_all.txt (TSV), 按 DTI 关系拆分
    - drug/protein ID 从 mapping/drug.txt, mapping/protein.txt 读取
    - ID 格式: "Drug::0", "Protein::0" (带前缀)
    - 结构特征: morganfp.txt + pro_ctd.txt → PCA → MinMaxScaler

  train_BioKG.py 原始:
    - 加载 BioKG 数据集的 dti.csv
    - 其余逻辑与 train2.py 完全相同

  本模块改造:
    - 支持加载并合并多个 KG 来源 (Hetionet, BioKG, Yamanishi08)
    - 统一实体 ID 格式
    - 建立 Davis/KIBA 数据集中 drug/protein 到 KG 实体的映射
    - 去除 PCA/MinMaxScaler 等结构特征处理 (不需要, KGFusionDTA 使用 ESM/分子图)
"""

import os
import pandas as pd
import numpy as np


def load_kg_triples(kg_path, sep='\t', columns=None):
    """
    加载单个 KG 三元组文件。

    来源: KGE-FUSION/train2.py L76-77
      原始代码:
        kg1 = pd.read_csv(os.path.join(DATASET_PATH, 'kg_all.txt'), delimiter='\\t', header=None)
        kg1.columns = ['head', 'relation', 'tail']

    Args:
        kg_path: str — 三元组文件路径
        sep: str — 分隔符 (train2.py 用 '\\t', train_BioKG.py 用 ',')
        columns: list — 列名, 默认 ['head', 'relation', 'tail']

    Returns:
        pd.DataFrame — (head, relation, tail) 三元组
    """
    if columns is None:
        columns = ['head', 'relation', 'tail']

    df = pd.read_csv(kg_path, sep=sep, header=None)
    df.columns = columns[:len(df.columns)]
    return df[['head', 'relation', 'tail']]


def remove_dti_from_kg(kg_df, dti_relation='drug_target_interaction'):
    """
    从 KG 中移除 DTI 关系三元组 (防止数据泄露)。

    来源: KGE-FUSION/train2.py L79-83
      原始代码:
        dt_08 = kg1[kg1['relation'] == 'drug_target_interaction']
        kg = pd.concat([kg1, dt_08]).drop_duplicates(
            subset=['head', 'relation', 'tail'], keep=False)

    Args:
        kg_df: pd.DataFrame — 完整 KG
        dti_relation: str — DTI 关系名称

    Returns:
        kg_clean: pd.DataFrame — 去除 DTI 后的 KG
        dti_triples: pd.DataFrame — 被移除的 DTI 三元组
    """
    dti_triples = kg_df[kg_df['relation'] == dti_relation]
    # train2.py 用 concat + drop_duplicates(keep=False) 实现差集
    kg_clean = pd.concat([kg_df, dti_triples]).drop_duplicates(
        subset=['head', 'relation', 'tail'], keep=False
    )
    kg_clean.index = range(len(kg_clean))
    return kg_clean, dti_triples


def merge_kgs(*kg_dfs):
    """
    合并多个 KG 三元组 DataFrame, 去重。

    来源: 新增 (train2.py 只处理单个 KG)
    KGFusionDTA 需要合并 Hetionet + BioKG + Yamanishi08

    Args:
        *kg_dfs: pd.DataFrame — 多个 KG 三元组 DataFrame

    Returns:
        pd.DataFrame — 合并去重后的三元组
    """
    merged = pd.concat(kg_dfs, ignore_index=True)
    merged = merged.drop_duplicates(subset=['head', 'relation', 'tail'])
    merged.index = range(len(merged))
    return merged


def save_triples_tsv(kg_df, output_path):
    """
    将三元组保存为 TSV 文件 (PyKEEN TriplesFactory.from_path 要求)。

    来源: train2.py 中注释掉的
      data.to_csv(data_path+'train_kg_'+str(i+1)+'.csv', header=None, index=False, sep='\\t')

    Args:
        kg_df: pd.DataFrame — (head, relation, tail) 三元组
        output_path: str — 输出文件路径
    """
    kg_df[['head', 'relation', 'tail']].to_csv(
        output_path, sep='\t', header=False, index=False
    )
    print(f"Saved {len(kg_df)} triples to {output_path}")


def load_entity_ids(mapping_path, prefix):
    """
    从映射文件加载实体 ID 列表。

    来源: KGE-FUSION/train2.py L96-109
      原始代码:
        fp_id_list = []
        fpid_path = './data/luo/mapping/drug.txt'
        fp_drugid = open(fpid_path, 'r')
        for idx, line in enumerate(fp_drugid.readlines()):
            fp_id_list.append("Drug::" + str(idx))

    Args:
        mapping_path: str — 实体映射文件路径 (每行一个实体原始名称)
        prefix: str — 实体 ID 前缀, 如 "Drug::" 或 "Protein::"

    Returns:
        list[str] — 实体 ID 列表, 格式为 "{prefix}{index}"
    """
    entity_ids = []
    with open(mapping_path, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            entity_ids.append(f"{prefix}{idx}")
    return entity_ids


def build_entity_mapping(drug_smiles_list, protein_id_list,
                         drug_mapping_path=None, protein_mapping_path=None):
    """
    建立 Davis/KIBA 数据集中的 drug/protein 到 KG 实体 ID 的映射。

    来源: KGE-FUSION/train2.py L96-125 的映射构建逻辑
      原始代码:
        fp_id_list.append("Drug::" + str(idx))      # drug → KG ID
        pro_id_list.append("Protein::" + str(idx))   # protein → KG ID
        fp_df = concat([fp_id, drug_features])        # 合并 ID 和特征
        prodes_df = concat([pro_id, protein_features])

    本函数改造:
      - 不再做 PCA/MinMaxScaler 特征处理 (KGFusionDTA 使用 ESM/分子图)
      - 只建立 drug/protein 原始标识 → KG 实体 ID 的映射

    Args:
        drug_smiles_list: list[str] — Davis/KIBA 中的 drug SMILES 列表
        protein_id_list: list[str] — Davis/KIBA 中的 protein ID 列表
        drug_mapping_path: str — (可选) drug mapping 文件路径
        protein_mapping_path: str — (可选) protein mapping 文件路径

    Returns:
        drug_to_kg_id: dict — {drug_smiles: "Drug::idx"}
        protein_to_kg_id: dict — {protein_id: "Protein::idx"}
    """
    drug_to_kg_id = {}
    for idx, smiles in enumerate(drug_smiles_list):
        drug_to_kg_id[smiles] = f"Drug::{idx}"

    protein_to_kg_id = {}
    for idx, pid in enumerate(protein_id_list):
        protein_to_kg_id[pid] = f"Protein::{idx}"

    return drug_to_kg_id, protein_to_kg_id


def load_and_merge_kgs(hetionet_path=None, biokg_path=None, yamanishi_path=None,
                       output_path=None):
    """
    加载并合并三个 KG 数据源。

    来源: 新增 (参考 train2.py 和 train_BioKG.py 的数据加载方式)
      train2.py:      kg_all.txt (TSV, Hetionet)
      train_BioKG.py: dti.csv (CSV, BioKG)

    Args:
        hetionet_path: str — Hetionet KG 文件路径 (TSV)
        biokg_path: str — BioKG 文件路径 (CSV)
        yamanishi_path: str — Yamanishi08 KG 文件路径 (TSV)
        output_path: str — (可选) 合并后三元组保存路径

    Returns:
        merged_kg: pd.DataFrame — 合并去重后的三元组 (不含 DTI 关系)
    """
    kgs = []

    if hetionet_path and os.path.exists(hetionet_path):
        # train2.py L76: pd.read_csv(kg_all.txt, delimiter='\t', header=None)
        kg = load_kg_triples(hetionet_path, sep='\t')
        kg, _ = remove_dti_from_kg(kg)
        kgs.append(kg)
        print(f"Loaded Hetionet: {len(kg)} triples")

    if biokg_path and os.path.exists(biokg_path):
        # train_BioKG.py L76: pd.read_csv(dti.csv, delimiter=',')
        kg = load_kg_triples(biokg_path, sep=',')
        kg, _ = remove_dti_from_kg(kg)
        kgs.append(kg)
        print(f"Loaded BioKG: {len(kg)} triples")

    if yamanishi_path and os.path.exists(yamanishi_path):
        kg = load_kg_triples(yamanishi_path, sep='\t')
        kg, _ = remove_dti_from_kg(kg)
        kgs.append(kg)
        print(f"Loaded Yamanishi08: {len(kg)} triples")

    if not kgs:
        raise ValueError("No KG data files found. Provide at least one KG path.")

    merged_kg = merge_kgs(*kgs)
    print(f"Merged KG: {len(merged_kg)} triples (deduplicated)")

    if output_path:
        save_triples_tsv(merged_kg, output_path)

    return merged_kg
