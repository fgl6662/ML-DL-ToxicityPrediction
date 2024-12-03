import matplotlib
matplotlib.use('TkAgg')
import os
from rdkit import Chem
import dgl
import csv
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dgllife.model import attentivefp_predictor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score, matthews_corrcoef, f1_score, cohen_kappa_score
import pandas as pd
import matplotlib.pyplot as plt
from dgllife.utils import mol_to_bigraph, atom_type_one_hot, atom_degree_one_hot, atom_formal_charge, \
    atom_num_radical_electrons, atom_hybridization_one_hot, atom_total_num_H_one_hot, one_hot_encoding, \
    ConcatFeaturizer, BaseAtomFeaturizer, BaseBondFeaturizer
from functools import partial

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

spec = "combined"
# 定义原子和键的特征化函数
def chirality(atom):
    try:
        return one_hot_encoding(atom.GetProp('_CIPCode'), ['R', 'S']) + \
               [atom.HasProp('_ChiralityPossible')]
    except:
        return [False, False] + [atom.HasProp('_ChiralityPossible')]

# 定义原子特征化函数
atom_featurizer = BaseAtomFeaturizer(
    {'hv': ConcatFeaturizer([
        partial(atom_type_one_hot, allowable_set=[
            'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At'],
            encode_unknown=True),
        partial(atom_degree_one_hot, allowable_set=list(range(6))),
        atom_formal_charge, atom_num_radical_electrons,
        partial(atom_hybridization_one_hot, encode_unknown=True),
        lambda atom: [0],
        atom_total_num_H_one_hot, chirality
    ])}
)

bond_featurizer = BaseBondFeaturizer({
    'he': lambda bond: [0 for _ in range(10)]
})

def collate_molgraphs(data):
    assert len(data[0]) in [3, 4], \
        'Expect the tuple to be of length 3 or 4, got {:d}'.format(len(data[0]))
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
        masks = None
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)
    return smiles, bg, labels, masks

def train_one_epoch(model, data_loader, loss_criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, masks = batch_data
        bg, labels, masks = bg.to(device), labels.to(device), masks.to(device)

        prediction = model(bg, bg.ndata['hv'], bg.edata['he'])
        loss = (loss_criterion(prediction, labels) * (masks != 0).float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(data_loader)

def evaluate_model(model, data_loader, device):
    model.eval()
    all_pred = []
    labels = []
    masks = []
    with torch.no_grad():
        for batch_data in data_loader:
            smiles, bg, batch_labels, batch_masks = batch_data
            bg = bg.to(device)
            batch_labels, batch_masks = batch_labels.to(device), batch_masks.to(device)
            pred = model(bg, bg.ndata['hv'], bg.edata['he'])
            pred = torch.sigmoid(pred)
            all_pred.append(pred.cpu().numpy())
            labels.append(batch_labels.cpu().numpy())
            masks.append(batch_masks.cpu().numpy())

    res = np.vstack(all_pred)
    labels = np.concatenate(labels)
    masks = np.concatenate(masks)
    return res, labels, masks

def run_a_train_epoch(model, train_loader, test_loader,loss_criterion, optimizer, device, epoch, results, best_metrics):
    train_loss = train_one_epoch(model, train_loader, loss_criterion, optimizer, device)
    res1, labels, masks = evaluate_model(model, test_loader, device)
    labels = labels[masks == 1]
    res = res1[masks == 1]

    predicted_labels = np.round(res)
    metrics = {
        'ACC': accuracy_score(labels, predicted_labels),
        'BA': balanced_accuracy_score(labels, predicted_labels),
        'ROC_AUC': roc_auc_score(labels, res),
        'PR_AUC': average_precision_score(labels, res),
        'MCC': matthews_corrcoef(labels, predicted_labels),
        'F1': f1_score(labels, predicted_labels),
        'CK': cohen_kappa_score(labels, predicted_labels)
    }
    print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Metrics: {metrics}')
    results.append(metrics)
    if metrics['ROC_AUC'] > best_metrics['ROC_AUC']:
        best_metrics.update(metrics)
        # 保存模型
        model_path = f"./result/model/{spec}/gnn.pth"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
    return train_loss, best_metrics

def data_train_test(n_epochs):
    best_metrics = {'ACC': 0, 'BA': 0, 'ROC_AUC': 0, 'PR_AUC': 0, 'MCC': 0, 'F1': 0, 'CK': 0}
    df_train = pd.read_csv(f"../../../data/data_kmean/{spec}/data_train.csv")
    df_test = pd.read_csv(f"../../../data/data_kmean/{spec}/data_test.csv")

    train_smi = df_train['SMILES'].tolist()
    train_class = torch.tensor(df_train['class'].astype(float).values).reshape(-1, 1)

    test_smi = df_test['SMILES'].tolist()
    test_class = torch.tensor(df_test['class'].astype(float).values).reshape(-1, 1)

    train_mols = [Chem.MolFromSmiles(smile) for smile in train_smi]
    test_mols = [Chem.MolFromSmiles(smile) for smile in test_smi]

    good_train_mols = [mol for mol in train_mols if mol is not None]
    good_test_mols = [mol for mol in test_mols if mol is not None]

    assert len(good_train_mols) == len(train_class), 'Number of molecules and classes must match.'
    assert len(good_test_mols) == len(test_class), 'Number of molecules and classes must match.'

    train_graph = [mol_to_bigraph(mol, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer) for mol in good_train_mols]
    test_graph = [mol_to_bigraph(mol, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer) for mol in good_test_mols]

    model = attentivefp_predictor.AttentiveFPPredictor(node_feat_size=39, edge_feat_size=10, num_layers=2,
                                                     num_timesteps=2,
                                                     graph_feat_size=200, n_tasks=4, dropout=0.2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    train_loader = DataLoader(dataset=list(zip(good_train_mols, train_graph, train_class)), batch_size=10,
                              collate_fn=collate_molgraphs)
    test_loader = DataLoader(dataset=list(zip(good_test_mols, test_graph, test_class)), batch_size=10,
                             collate_fn=collate_molgraphs)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=10 ** (-2.5), weight_decay=10 ** (-5.0))

    results = []
    for e in range(n_epochs):
        run_a_train_epoch(model,train_loader,test_loader, loss_fn, optimizer, device, e, results, best_metrics)
    # 绘制AUC-ROC曲线并保存图像
    plot_path = f"./result/{spec}/plot/"
    os.makedirs(plot_path, exist_ok=True)
    plt.plot(range(n_epochs), [metrics['ROC_AUC'] for metrics in results])
    plt.xlabel('Epochs')
    plt.ylabel('AUC-ROC')
    plt.title(f'AUC-ROC for {spec}')
    plt.savefig(f'{plot_path}auc_roc_plot.png')  # 保存图像
    plt.close()  # 关闭图像以释放内存
    # 保存最好的指标到CSV文件
    csv_path = f"./result/{spec}/best_metrics.csv"
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['ACC', 'BA', 'ROC_AUC', 'PR_AUC', 'MCC', 'F1', 'CK']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(best_metrics)

data_train_test(n_epochs=100)