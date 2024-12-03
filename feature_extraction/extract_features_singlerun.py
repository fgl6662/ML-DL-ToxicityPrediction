# 导入库
import pandas as pd
import numpy as np
import os
from featurizer.Mol2vec.getMol2vec import extract_Mol2Vec
from featurizer.rdkitMD.getRdkitMD import extract_rdkitMD
from featurizer.MorganFP.getMorganFP import extract_MorganFP

# 物种与数据集定义
spec = "combined"
datasets = ['train', 'val', 'test']


for dataset in datasets:
    # 加载数据
    data = pd.read_csv(f"../data/data_kmean/{spec}/data_{dataset}.csv")['SMILES'].to_list()
    label = pd.read_csv(f"../data/data_kmean/{spec}/data_{dataset}.csv")['class'].values

    # 检查并创建保存特征的路径
    PATH = f"../data/singlerun/featurised_data/{spec}"
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    # 提取特征
    rdkit_md = extract_rdkitMD(data).iloc[:,1:].values
    mf1024 = extract_MorganFP(data, bit_type=1024).iloc[:,1:].values
    mf2048 = extract_MorganFP(data, bit_type=2048).iloc[:,1:].values
    mol2vec = extract_Mol2Vec(data)

    # 保存特征
    np.save(f"{PATH}/rdkit_md_{dataset}.npy", rdkit_md)
    np.save(f"{PATH}/fp1024_{dataset}.npy", mf1024)
    np.save(f"{PATH}/fp2048_{dataset}.npy", mf2048)
    np.save(f"{PATH}/mol2vec_{dataset}.npy", mol2vec)
    np.save(f"{PATH}/label_{dataset}.npy", label)