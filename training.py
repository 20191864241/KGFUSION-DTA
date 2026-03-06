"""
KGFusion-DTA Training Script

Task 7c: 改造自原始 training.py (230 行)

改造对照表:
  | 原始 training.py         | 改造后                                    | 来源            |
  |--------------------------|-------------------------------------------|-----------------|
  | argparse (epochs/lr/...) | 新增 --lambda-cl, --temperature, --kg-emb | KGE-FUSION      |
  | VAE 初始化               | 删除                                      | —               |
  | DTAData MACCS 指纹       | smiles_tokens                             | Task 1          |
  | DAT3() 模型初始化        | KGFusionDTA(config)                       | Task 7a         |
  | collate 返回 8 元素      | collate 返回 11 元素                      | Task 7b         |
  | forward(protein, smiles, ..., vae, ...) | forward(smiles_tokens, ...) | Task 7a  |
  | loss = MSE(out, affinity)| total_loss = MSE + λ * CL                 | Task 6b         |
  | 评估 (CI only)           | 新增 MSE, r_m² 指标                       | KGE-FUSION/utils|
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from src.getdata import getdata_from_csv
from src.utils import DrugTargetDataset, collate, AminoAcid, ci, mse, rmse, pearson, spearman, rm_squared
from src.models.DAT import KGFusionDTA

# --- argparse (改造自原始 L1-31) ---
# 新增: --lambda-cl, --temperature, --kg-emb-path
# 删除: VAE 相关参数 (encoder_layer_sizes, decoder_layer_sizes, latent_size)
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default=True, help='Disables CUDA training.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--batchsize', type=int, default=128, help='Number of batch_size')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate.')
parser.add_argument('--pretrain', action='store_false', help='protein pretrained or not')
parser.add_argument('--dataset', default='davis', help='dataset: davis or kiba')
parser.add_argument('--training-dataset-path', default='data/davis_train.csv', help='training dataset path')
parser.add_argument('--testing-dataset-path', default='data/davis_test.csv', help='testing dataset path')

# --- 新增参数 (来自 KGE-FUSION/train2.py 的 config + 论文设计) ---
parser.add_argument('--lambda-cl', type=float, default=0.1,
                    help='对比学习损失权重 λ (DAT3 中无此参数)')
parser.add_argument('--temperature', type=float, default=0.07,
                    help='InfoNCE 温度 τ (DAT3 中无此参数)')
parser.add_argument('--kg-emb-path', type=str, default=None,
                    help='KG 嵌入 npz 文件路径 (DAT3 中无此参数)')
parser.add_argument('--kg-model-path', type=str, default='saved_models/distmult/',
                    help='KG 模型路径, 用于可选预训练 (来自 KGE-FUSION/train2.py)')

args = parser.parse_args()
dataset = args.dataset
use_cuda = args.cuda and torch.cuda.is_available()

batch_size = args.batchsize
epochs = args.epochs
lr = args.lr
weight_decay = args.weight_decay

is_pretrain = args.pretrain

Alphabet = AminoAcid()

training_dataset_address = args.training_dataset_path
testing_dataset_address = args.testing_dataset_path

# --- 删除 VAE 初始化 (原始 L57-61) ---
# 原始:
#   vae = VAE(encoder_layer_sizes=..., latent_size=..., decoder_layer_sizes=...).cuda()
# 改造: 不再需要 VAE

# --- 可选: KG 嵌入预训练 (新增, 来自 KGE-FUSION/train2.py L283-302) ---
if args.kg_emb_path is None:
    kg_emb_path = f'data/kg/{dataset}_kg_embeddings.npz'
else:
    kg_emb_path = args.kg_emb_path

if not os.path.exists(kg_emb_path):
    print(f"[警告] KG 嵌入文件 {kg_emb_path} 不存在, 将使用零向量占位")
    print("       如需生成, 请先运行: python -m src.kg.train_kg")

# ---加载训练数据 (改造自原始 L63-81) ---
# 改造: DrugTargetDataset 新增 kg_emb_path 参数
if is_pretrain:
    train_drug, train_protein, train_affinity, pid = getdata_from_csv(training_dataset_address, maxlen=1536)
else:
    train_drug, train_protein, train_affinity = getdata_from_csv(training_dataset_address, maxlen=1024)
    train_protein = [x.encode('utf-8').upper() for x in train_protein]
    train_protein = [torch.from_numpy(Alphabet.encode(x)).long() for x in train_protein]
train_affinity = torch.from_numpy(np.array(train_affinity)).float()

# 创建 Dataset (改造: 新增 kg_emb_path)
dataset_train = DrugTargetDataset(
    train_drug, train_protein, train_affinity, pid,
    is_target_pretrain=is_pretrain, self_link=False,
    dataset=dataset, kg_emb_path=kg_emb_path
)
dataloader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate
)

# ---加载测试数据 (改造自原始 L83-97) ---
if is_pretrain:
    test_drug, test_protein, test_affinity, pid = getdata_from_csv(testing_dataset_address, maxlen=1536)
else:
    test_drug, test_protein, test_affinity = getdata_from_csv(testing_dataset_address, maxlen=1024)
    test_protein = [x.encode('utf-8').upper() for x in test_protein]
    test_protein = [torch.from_numpy(Alphabet.encode(x)).long() for x in test_protein]
test_affinity = torch.from_numpy(np.array(test_affinity)).float()

dataset_test = DrugTargetDataset(
    test_drug, test_protein, test_affinity, pid,
    is_target_pretrain=is_pretrain, self_link=False,
    dataset=dataset, kg_emb_path=kg_emb_path
)
dataloader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate
)

# ---加载模型 (改造自原始 L99-101) ---
# 原始: model = DAT3(embedding_dim, rnn_dim, hidden_dim, graph_dim, dropout, alpha, n_heads, ...)
# 改造: model = KGFusionDTA(config)
config = {
    'dropout': args.dropout,
    'temperature': args.temperature,
    # 编码器默认参数在 KGFusionDTA.__init__ 中设置
}
model = KGFusionDTA(config)

if use_cuda:
    model.cuda()

# 优化器 (保留)
params = [p for p in model.parameters() if p.requires_grad]
optim = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

# 损失函数
# 原始: criterion = nn.MSELoss()
# 改造: MSE + λ * CL (CL 由模型内部计算)
criterion = nn.MSELoss()

train_epoch_size = len(train_drug)
test_epoch_size = len(test_drug)
print('--- KGFusionDTA model --- ')

best_ci = 0
best_mse = 100000

# ---进行训练 (改造自原始 L126-230) ---
for epoch in range(epochs):
    # ===== 训练阶段 =====
    model.train()
    b = 0
    total_loss = []
    total_ci = []

    # 改造: collate 返回 11 元素 (原 8 元素)
    for (prot_esm, smiles_tokens, affinity,
         drug_nodes, drug_edges, drug_batch,
         prot_nodes, prot_edge_index, prot_batch,
         drug_kg, prot_kg) in dataloader_train:

        if use_cuda:
            prot_esm = prot_esm.cuda()
            smiles_tokens = smiles_tokens.cuda()
            affinity = affinity.cuda()
            drug_nodes = drug_nodes.cuda()
            drug_edges = drug_edges.cuda()
            drug_batch = drug_batch.cuda()
            prot_nodes = prot_nodes.cuda()
            prot_edge_index = prot_edge_index.cuda()
            prot_batch = prot_batch.cuda()
            drug_kg = drug_kg.cuda()
            prot_kg = prot_kg.cuda()

        # forward (改造自原始 L146)
        # 原始: _, out = model(protein, smiles, node_drug, edge_drug, batch, vae, node_proteins, edge_proteins)
        # 改造: y_pred, cl_loss = model(smiles_tokens, ...)
        y_pred, cl_loss = model(
            smiles_tokens, drug_nodes, drug_edges, drug_batch,
            prot_esm, prot_nodes, prot_edge_index, prot_batch,
            drug_kg, prot_kg
        )

        # 损失 (改造自原始 L147)
        # 原始: loss = criterion(out, affinity)
        # 改造: total_loss = MSE + λ * CL
        mse_loss = criterion(y_pred, affinity)
        loss = mse_loss + args.lambda_cl * cl_loss

        optim.zero_grad()
        loss.backward()
        optim.step()

        out_cpu = y_pred.cpu().detach()
        affinity_cpu = affinity.cpu().detach()
        loss_cpu = loss.cpu().detach().item()
        c_index = ci(affinity_cpu.numpy(), out_cpu.numpy())

        b = b + batch_size
        total_loss.append(loss_cpu)
        total_ci.append(c_index)
        print('# [{}/{}] training {:.1%} loss={:.5f} (mse={:.5f}, cl={:.5f}), ci={:.5f}'.format(
            epoch + 1, epochs, b / train_epoch_size,
            loss_cpu, mse_loss.item(), cl_loss.item(), c_index
        ), end='\r')

    print('\ntotal_loss={:.5f}, total_ci={:.5f}'.format(np.mean(total_loss), np.mean(total_ci)))

    # ===== 评估阶段 (改造自原始 L171-226) =====
    model.eval()
    b = 0
    total_loss_eval = []
    total_ci_eval = []
    total_pred = torch.Tensor()
    total_label = torch.Tensor()

    with torch.no_grad():
        for (prot_esm, smiles_tokens, affinity,
             drug_nodes, drug_edges, drug_batch,
             prot_nodes, prot_edge_index, prot_batch,
             drug_kg, prot_kg) in dataloader_test:

            if use_cuda:
                prot_esm = prot_esm.cuda()
                smiles_tokens = smiles_tokens.cuda()
                affinity = affinity.cuda()
                drug_nodes = drug_nodes.cuda()
                drug_edges = drug_edges.cuda()
                drug_batch = drug_batch.cuda()
                prot_nodes = prot_nodes.cuda()
                prot_edge_index = prot_edge_index.cuda()
                prot_batch = prot_batch.cuda()
                drug_kg = drug_kg.cuda()
                prot_kg = prot_kg.cuda()

            y_pred, cl_loss = model(
                smiles_tokens, drug_nodes, drug_edges, drug_batch,
                prot_esm, prot_nodes, prot_edge_index, prot_batch,
                drug_kg, prot_kg
            )

            loss = criterion(y_pred, affinity)

            out_cpu = y_pred.cpu()
            affinity_cpu = affinity.cpu()
            loss_cpu = loss.cpu().detach().item()
            c_index = ci(affinity_cpu.detach().numpy(), out_cpu.detach().numpy())

            b = b + batch_size
            total_loss_eval.append(loss_cpu)
            total_ci_eval.append(c_index)
            total_pred = torch.cat((total_pred, out_cpu), 0)
            total_label = torch.cat((total_label, affinity_cpu), 0)

            print('# [{}/{}] testing {:.1%} loss={:.5f}, ci={:.5f}'.format(
                epoch + 1, epochs, b / test_epoch_size, loss_cpu, c_index
            ), end='\r')

    # 全局评估指标 (改造: 新增 MSE, RMSE, Pearson, Spearman, r_m²)
    y_true = total_label.detach().numpy().flatten()
    y_pred_np = total_pred.detach().numpy().flatten()

    all_ci = ci(y_true, y_pred_np)
    all_mse = mse(y_true, y_pred_np)
    all_rmse = rmse(y_true, y_pred_np)
    all_pearson = pearson(y_true, y_pred_np)
    all_spearman = spearman(y_true, y_pred_np)
    all_rm2 = rm_squared(y_true, y_pred_np)

    # 原始: print('total_loss={:.5f}, total_ci={:.5f}'.format(...))
    # 改造: 输出完整指标
    print('\n[Epoch {}] Eval | loss={:.5f}, CI={:.5f}, MSE={:.5f}, RMSE={:.5f}, '
          'Pearson={:.5f}, Spearman={:.5f}, r_m²={:.5f}'.format(
        epoch + 1, np.mean(total_loss_eval), all_ci, all_mse, all_rmse,
        all_pearson, all_spearman, all_rm2
    ))

    # 保存最佳模型 (保留原逻辑, 路径改名)
    save_path = f'saved_models/KGFusionDTA_best_{dataset}.pkl'
    if all_ci > best_ci:
        best_ci = all_ci
        model.cpu()
        save_dict = {
            'model': model.state_dict(),
            'optim': optim.state_dict(),
            'ci': best_ci,
            'mse': all_mse,
            'epoch': epoch + 1
        }
        os.makedirs('saved_models', exist_ok=True)
        torch.save(save_dict, save_path)
        print(f'  => Best model saved (CI={best_ci:.5f})')
        if use_cuda:
            model.cuda()
