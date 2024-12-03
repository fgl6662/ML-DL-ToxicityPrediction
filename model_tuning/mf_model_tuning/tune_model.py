# Import Library
import numpy as np
from tuning_func import knn_tuning, svm_tuning, xgb_tuning, rf_tuning
import time
spec = "combined"
fea = "2048"
#--------------------------------------------------------------------#
# TRAINING MODEL                                                     #
#--------------------------------------------------------------------#
#Set up parameters
my_n_neighbors = np.arange(3,16,2)
knn_para = [my_n_neighbors]

my_C     = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
my_gamma = [0.001, 0.01, 0.1, 1, 10]
svm_para = [my_C, my_gamma]

my_n_estimators_xgb     = 50
my_learning_rate_xgb    = [0.001, 0.01, 0.1]
my_max_depth_xgb        = np.arange(3, 9)
my_colsample_bytree_xgb = np.arange(0.2, 0.95, 0.1)
xgb_para = [my_n_estimators_xgb, my_learning_rate_xgb, my_max_depth_xgb, my_colsample_bytree_xgb]

my_n_estimators_rf      = 50
my_max_depth_rf         = np.arange(3, 9)
my_max_features_rf      = np.arange(0.2, 0.95, 0.1)
my_min_samples_split_rf = np.arange(2, 6)
rf_para = [my_n_estimators_rf, my_max_depth_rf, my_max_features_rf, my_min_samples_split_rf]

#Define parameter
# 特征比特长度列表

X_data = np.load(f'../../../data/featurised_data/{spec}/fp{fea}_train.npy', allow_pickle=True)
y_data = np.load(f'../../../data/featurised_data/{spec}/label_train.npy')
test_train = np.load(f'../../../data/featurised_data/{spec}/fp{fea}_test.npy', allow_pickle=True)
test_label = np.load(f'../../../data/featurised_data/{spec}/label_test.npy')

# Training
start_knn = time.time()
knn_tuning(X_data, y_data, test_train, test_label, tag="mf_{}".format(fea), para=knn_para)
end_knn = time.time()
print("KNN Time:{} seconds".format(end_knn - start_knn))
# -----------------------------
start_svm = time.time()
svm_tuning(X_data, y_data, test_train, test_label, tag="mf_{}".format(fea), para=svm_para)
end_svm = time.time()
print("SVM Time:{} seconds".format(end_svm - start_svm))
# -----------------------------
start_xgb = time.time()
xgb_tuning(X_data, y_data, test_train, test_label, tag="mf_{}".format(fea), para=xgb_para)
end_xgb = time.time()
print("XGB Time:{} seconds".format(end_xgb - start_xgb))
# -----------------------------
start_rf = time.time()
rf_tuning(X_data, y_data, test_train, test_label, tag="mf_{}".format(fea), para=rf_para)
end_rf = time.time()
print("RF Time:{} seconds".format(end_rf - start_rf))



