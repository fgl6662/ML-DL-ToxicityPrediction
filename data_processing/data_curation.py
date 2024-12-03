# Import libraries
import os
import pandas as pd 
from chemtonic.curation.refinement import refineComplete

#============================================================
# Load active/inactive compounds
spec = "96h LC50"

active_compounds = pd.read_csv(f"../data/data_clean/{spec}/{spec}_active.csv")['canonicalsmiles'].tolist()
inactive_compounds = pd.read_csv(f"../data/data_clean/{spec}/{spec}_inactive.csv")['canonicalsmiles'].tolist()

# Fullly refine
refined_actives = refineComplete(active_compounds)
refined_inactives = refineComplete(inactive_compounds)

# Remove conflicted labels
active_labels = [1] * len(refined_actives)
inactive_labels = [0] * len(refined_inactives)

refined_compounds = refined_actives['SMILES'].tolist() + refined_inactives['SMILES'].tolist()
refined_labels = [1] * len(refined_actives) + [0] * len(refined_inactives)

df = pd.DataFrame(zip(refined_compounds, refined_labels), columns=['SMILES', 'class'])
df = df.drop_duplicates(subset=['SMILES'], keep=False)
save_path = "../data/data_curation/{spec}/".format(spec=spec)
if not os.path.exists(save_path):
    os.makedirs(save_path)
df.to_csv(f"{save_path}{spec}_anticancerNP.csv", index=False)

# Check number of compounds per class
num_active   = len(df[df['class']==1]) #367
num_inactive = len(df[df['class']==0]) #630
