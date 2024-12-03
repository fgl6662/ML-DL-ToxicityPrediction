
import pandas as pd
import os

# 定义结果组织
models = ['knn', 'svm', 'rf', 'xgb']
species = ['combined']

for spec in species:
    result_path = f'./result/{spec}/'
    os.makedirs(result_path, exist_ok=True)

    for model in models:
        # 读取数据
        metrics_df = pd.read_csv(f"./md_model_tuning/result/{spec}/{model}/{model.upper()}_md.csv")['Metrics']
        rdkitmd_df = pd.read_csv(f"./md_model_tuning/result/{spec}/{model}/{model.upper()}_md.csv")['md']
        mf_1024_df = pd.read_csv(f"./mf_model_tuning/result/{spec}/{model}/{model.upper()}_mf_1024.csv")['mf_1024']
        mf_2048_df = pd.read_csv(f"./mf_model_tuning/result/{spec}/{model}/{model.upper()}_mf_2048.csv")['mf_2048']
        mol2vec_df = pd.read_csv(f"./mol2vec_moldel_tuning/result/{spec}/{model}/{model.upper()}_mol2vec.csv")['mol2vec']

        # 拼接数据
        concat_df = pd.concat([metrics_df, rdkitmd_df, mf_1024_df, mf_2048_df, mol2vec_df], axis=1)

        # 保存到对应物种的目录下
        concat_df.to_csv(f"{result_path}{model.upper()}.csv",index=False)