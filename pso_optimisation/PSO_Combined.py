import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
import pandas as pd
import numpy as np
import random
import os
import dgl
from pyswarm.pyswarm import pso

from utils import printPerformance, extract_weight, get_optimasation_function
from imblearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dgllife.model import AttentiveFPPredictor
from dgllife.utils import mol_to_bigraph, atom_type_one_hot, atom_degree_one_hot, atom_formal_charge, \
    atom_num_radical_electrons, atom_hybridization_one_hot, atom_total_num_H_one_hot, one_hot_encoding, \
    ConcatFeaturizer, BaseAtomFeaturizer, BaseBondFeaturizer
from functools import partial
from rdkit import Chem
from sklearn.feature_selection import VarianceThreshold

threshold = (.95 * (1 - .95))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
spec = "combined"
# 定义原子和键的特征化函数
def chirality(atom):
    try:
        return one_hot_encoding(atom.GetProp('_CIPCode'), ['R', 'S']) + \
               [atom.HasProp('_ChiralityPossible')]
    except:
        return [False, False] + [atom.HasProp('_ChiralityPossible')]

atom_featurizer = BaseAtomFeaturizer(
    {'hv': ConcatFeaturizer([
        partial(atom_type_one_hot, allowable_set=[
            'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At'],
            encode_unknown=True),
        partial(atom_degree_one_hot, allowable_set=list(range(6))),
        atom_formal_charge, atom_num_radical_electrons,
        partial(atom_hybridization_one_hot, encode_unknown=True),
        lambda atom: [0],  # A placeholder for aromatic information,
        atom_total_num_H_one_hot, chirality
    ])}
)

bond_featurizer = BaseBondFeaturizer({
    'he': lambda bond: [0 for _ in range(10)]
})

def collate_molgraphs(data):
    # 合并分子图数据
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

# 定义模型
my_svm_mol2vec = make_pipeline(VarianceThreshold(), SVC(random_state=42,C=10,gamma=0.001, probability=True))
my_xgb_mol2vec = make_pipeline(VarianceThreshold(), XGBClassifier(random_state=42, n_estimators=50, max_depth=6, colsample_bytree=0.4, learning_rate=0.1))
my_knn_mol2vec = make_pipeline(VarianceThreshold(), KNeighborsClassifier(n_neighbors=11))
my_gnn_graph = AttentiveFPPredictor(node_feat_size=39, edge_feat_size=10, num_layers=2, num_timesteps=2,
                                    graph_feat_size=200, n_tasks=1, dropout=0.2)

train_mol2vec = np.load(f"../../data/featurised_data/{spec}/mol2vec_train.npy", allow_pickle=True)
train_label = np.load(f"../../data/featurised_data/{spec}/label_train.npy", allow_pickle=True)

test_mol2vec = np.load(f"../../data/featurised_data/{spec}/mol2vec_test.npy", allow_pickle=True)
test_label = np.load(f"../../data/featurised_data/{spec}/label_test.npy", allow_pickle=True)

val_mol2vec = np.load(f"../../data/featurised_data/{spec}/mol2vec_val.npy", allow_pickle=True)
val_label = np.load(f"../../data/featurised_data/{spec}/label_val.npy", allow_pickle=True)

train_gnn = pd.read_csv(f"../../data/data_kmean/{spec}/data_train.csv")
val_gnn = pd.read_csv(f"../../data/data_kmean/{spec}/data_val.csv")
test_gnn = pd.read_csv(f"../../data/data_kmean/{spec}/data_test.csv")

train_smi = train_gnn['SMILES'].tolist()
train_class = torch.tensor(train_gnn['class'].astype(float).values).reshape(-1, 1)
test_smi = test_gnn['SMILES'].tolist()
test_class = torch.tensor(test_gnn['class'].astype(float).values).reshape(-1, 1)
val_smi = val_gnn['SMILES'].tolist()
val_class = torch.tensor(val_gnn['class'].astype(float).values).reshape(-1, 1)

train_mols = [Chem.MolFromSmiles(smile) for smile in train_smi]
test_mols = [Chem.MolFromSmiles(smile) for smile in test_smi]
val_mols = [Chem.MolFromSmiles(smile) for smile in val_smi]

good_train_mols = [mol for mol in train_mols if mol is not None]
good_test_mols = [mol for mol in test_mols if mol is not None]
good_val_mols = [mol for mol in val_mols if mol is not None]

assert len(good_train_mols) == len(train_class), 'Number of molecules and classes must match.'
assert len(good_test_mols) == len(test_class), 'Number of molecules and classes must match.'
assert len(good_val_mols) == len(val_class), 'Number of molecules and classes must match.'

train_graphs = [mol_to_bigraph(Chem.MolFromSmiles(smiles), node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer) for smiles in train_gnn['SMILES']]
val_graphs = [mol_to_bigraph(Chem.MolFromSmiles(smiles), node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer) for smiles in val_gnn['SMILES']]
test_graphs = [mol_to_bigraph(Chem.MolFromSmiles(smiles), node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer) for smiles in test_gnn['SMILES']]

train_loader = DataLoader(dataset=list(zip(good_train_mols, train_graphs, train_class)), batch_size=10, collate_fn=collate_molgraphs)
val_loader   = DataLoader(dataset=list(zip(good_val_mols, val_graphs, val_class)), batch_size=10, collate_fn=collate_molgraphs)
test_loader  = DataLoader(dataset=list(zip(good_test_mols, test_graphs, test_class)), batch_size=10, collate_fn=collate_molgraphs)

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = my_gnn_graph.to(device)
model_path = f"../../model_tuning/AttentiveFP/result/model/combined/gnn.pth"
model.load_state_dict(torch.load(model_path, map_location=device))

test_res, test_labels, masks = evaluate_model(model, test_loader, device)
test_pred_gnn_graph = test_res[masks == 1]
test_pred_svm_mol2vec = my_svm_mol2vec.fit(train_mol2vec, train_label).predict_proba(test_mol2vec)[:, 1]
test_pred_xgb_mol2vec = my_xgb_mol2vec.fit(train_mol2vec, train_label).predict_proba(test_mol2vec)[:, 1]
test_pred_knn_mol2vec = my_knn_mol2vec.fit(train_mol2vec, train_label).predict_proba(test_mol2vec)[:, 1]

test_pred_list = [test_pred_svm_mol2vec, test_pred_xgb_mol2vec, test_pred_knn_mol2vec, test_pred_gnn_graph]
auc_optimization_func = get_optimasation_function(test_pred_list, test_label)

lb, ub = [0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.5, 0.5]
seed_range = np.arange(0, 10)
weight_list = []
for s in seed_range:
    np.random.seed(int(s))
    random.seed(int(s))
    optx, fopt = pso(auc_optimization_func, lb, ub, swarmsize=100, seed=s, maxiter=50)
    weight_list.append(extract_weight(optx))
    print(f"Round {s + 1}: Completed")

w1_list, w2_list, w3_list, w4_list = [], [], [], []
for weight in weight_list:
    w1_list.append(weight[0])
    w2_list.append(weight[1])
    w3_list.append(weight[2])
    w4_list.append(weight[3])

w1_avage_norm = np.round(np.mean(w1_list), 4)
w2_avage_norm = np.round(np.mean(w2_list), 4)
w3_avage_norm = np.round(np.mean(w3_list), 4)
w4_avage_norm = np.round(np.mean(w4_list), 4)

# 使用各个模型进行预测
test_pred_svm_mol2vec = my_svm_mol2vec.fit(train_mol2vec, train_label).predict_proba(test_mol2vec)[:, 1]
test_pred_xgb_mol2vec = my_xgb_mol2vec.fit(train_mol2vec, train_label).predict_proba(test_mol2vec)[:, 1]
test_pred_knn_mol2vec = my_knn_mol2vec.fit(train_mol2vec, train_label).predict_proba(test_mol2vec)[:, 1]
test_res, test_labels, masks = evaluate_model(model, test_loader, device)
test_pred_gnn_graph = test_res[masks == 1]

# 计算各个模型的 AUC-ROC 分数
test_roc_auc_svm_mol2vec = roc_auc_score(test_label, test_pred_svm_mol2vec)
test_roc_auc_xgb_mol2vec = roc_auc_score(test_label, test_pred_xgb_mol2vec)
test_roc_auc_knn_mol2vec = roc_auc_score(test_label, test_pred_knn_mol2vec)
test_roc_auc_gnn_graph = roc_auc_score(test_label, test_pred_gnn_graph)

# 合并所有模型的预测结果
test_pred_top4ensemble = (test_pred_svm_mol2vec * w1_avage_norm + test_pred_xgb_mol2vec * w2_avage_norm + test_pred_knn_mol2vec * w3_avage_norm + test_pred_gnn_graph * w4_avage_norm)

# 计算集成模型的 AUC-ROC 分数
test_roc_auc_top4ensemble = roc_auc_score(test_label, test_pred_top4ensemble)

# 导出结果
test_roc_list = [test_roc_auc_svm_mol2vec, test_roc_auc_xgb_mol2vec, test_roc_auc_knn_mol2vec, test_roc_auc_gnn_graph, test_roc_auc_top4ensemble]
weight_list = [w1_avage_norm, w2_avage_norm, w3_avage_norm, w4_avage_norm, 1]
fea_list = ['svm_mol2vec', 'xgb_mol2vec', 'knn_mol2vec', 'gnn_graph', 'top4ensemble']
weight_path = f"../../PSOresults/Combined/weights"
if not os.path.isdir(weight_path):
    os.makedirs(weight_path)


pred_path = f"../../PSOresults/Combined/pred/top4"
if not os.path.isdir(pred_path):
    os.makedirs(pred_path)
pd.DataFrame(zip(test_roc_list, weight_list), index=fea_list, columns=['ROC-AUC', 'Weight']).to_csv(f"{weight_path}/roc_auc_test_top4ensemble.csv", index=None)
pd.DataFrame(zip(test_pred_top4ensemble, test_label), columns=["predicted_prob", "true_class"]).to_csv(f"{pred_path}/y_prob_test_top4ensemble.csv", index=None)

models = ['svm_mol2vec', 'xgb_mol2vec', 'knn_mol2vec', 'gnn_graph', 'top4ensemble']
preds = [test_pred_svm_mol2vec, test_pred_xgb_mol2vec, test_pred_knn_mol2vec, test_pred_gnn_graph, test_pred_top4ensemble]
auc_scores = [test_roc_auc_svm_mol2vec, test_roc_auc_xgb_mol2vec, test_roc_auc_knn_mol2vec, test_roc_auc_gnn_graph, test_roc_auc_top4ensemble]

plt.figure()
for i, model in enumerate(models):
    fpr, tpr, _ = roc_curve(test_label, preds[i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model} (area = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC curves of XRA-ens and its base classifiers on the Combined Dataset')
plt.legend(loc="lower right")
plt.show()

