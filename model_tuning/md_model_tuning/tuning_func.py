# Import Library
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import os
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from utils import printPerformance
spec = "combined"
#--------------------------------------------------------------------#
# Remove Low-variance Features
from sklearn.feature_selection import VarianceThreshold
threshold = (.95 * (1 - .95))

# Normalize Features
from sklearn.preprocessing import MinMaxScaler

#--------------------------------------------------------------------#
# TUNING KNN MODEL                                                   #
#--------------------------------------------------------------------#
def knn_tuning(X_train, y_train,test_train,test_label, scoring = 'roc_auc', nfold=5, tag=None, seed=0, para=None):
    X_train_, y_train_ = X_train, y_train
    test_train_, test_label_ = test_train, test_label
    if tag is not None:
        tag_ = tag
    seed_     = seed
    scoring_  = scoring
    nfold_    = nfold
    concat_df = pd.DataFrame(['AUC-ROC', 'AUC-PR','ACC', 'BA', 'MCC', 
                              'SN/RE', 'SP', 'PR', 'F1', 'CK', 
                              'cross_validated',
                              'best_n_neighbors'], columns= ["Metrics"]) 
    #=====================================#
    # Set Up Parameter
    if para == None:
        my_n_neighbors = np.arange(3,21)
    else:
        my_n_neighbors = para[0]
    #=====================================# 
    my_classifier = make_pipeline(MinMaxScaler(),
                                  VarianceThreshold(threshold),
                                  KNeighborsClassifier())
    #=====================================#
    # GridsearchCV
    my_parameters_grid = {'n_neighbors': my_n_neighbors}
    my_new_parameters_grid = {'kneighborsclassifier__' + key: my_parameters_grid[key] for key in my_parameters_grid}
    grid_cv = GridSearchCV(my_classifier, 
                           my_new_parameters_grid, 
                           scoring=scoring_,
                           n_jobs= -1,
                           cv = StratifiedKFold(n_splits=nfold_, shuffle=True, random_state=42),
                           return_train_score=True)
    grid_cv.fit(X_train_, y_train_)
    #=====================================#
    # Create Regressor uing Best Parameters (Use only one option at each run)
    best_n_neighbors = grid_cv.best_params_['kneighborsclassifier__n_neighbors']
    cross_validated  = grid_cv.best_score_      
    best_para_set    = [cross_validated, best_n_neighbors]
    #=====================================#
    my_best_classifier = make_pipeline(MinMaxScaler(), 
                                       VarianceThreshold(threshold),
                                       KNeighborsClassifier(n_neighbors=best_n_neighbors))
    #=====================================#
    # Testing on train data
    my_best_classifier.fit(X_train_, y_train_)
    y_pred = my_best_classifier.predict(X_train_)
    y_prob = my_best_classifier.predict_proba(test_train_)[::,1]
    #=====================================#
    pred_path = f'./pred/{spec}/knn/'
    os.makedirs(pred_path, exist_ok=True)
    df = pd.DataFrame(y_prob, columns=['Predicted Probability'])
    file_name = f'y_prob_KNN_{tag_}.csv'
    df.to_csv(os.path.join(pred_path, file_name), index=False)
    #=====================================#
    # Evaluation
    x         = printPerformance(test_label_, y_prob)
    print(x)
    x_list    = list(x)
    x_list    = x_list + best_para_set
    new_df    = pd.DataFrame(x_list, columns = [tag_])
    concat_df = pd.concat([concat_df, new_df], axis=1)
    result_path = f'./result/{spec}/knn/'
    os.makedirs(result_path, exist_ok=True)
    file_path = os.path.join(result_path, f'KNN_{tag_}.csv')
    concat_df.to_csv(file_path, index=False)

    para_path = f'./para/{spec}/knn/'
    os.makedirs(para_path, exist_ok=True)
    best_para = pd.DataFrame(best_para_set, columns=['Best_para'])
    best_para.to_csv(os.path.join(para_path, 'Best_para.csv'), index=False)

#--------------------------------------------------------------------#
# TUNING SVM MODEL                                                   #
#--------------------------------------------------------------------#
def svm_tuning(X_train, y_train,test_train,test_label, scoring = 'roc_auc', nfold=5, tag=None, seed=0, para=None):
    X_train_, y_train_ = X_train, y_train
    test_train_, test_label_ = test_train, test_label
    if tag is not None:
        tag_ = tag
    seed_     = seed
    scoring_  = scoring
    nfold_    = nfold
    concat_df = pd.DataFrame(['AUC-ROC', 'AUC-PR','ACC', 'BA', 
                              'SN/RE', 'SP', 'PR', 'MCC', 'F1', 'CK', 
                              'cross_validated',
                              'best_C', 'best_gamma'], columns= ["Metrics"]) 
    #=====================================#
    # Set Up Parameter
    if para == None:
        my_C     = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 25, 50, 75, 100]
        my_gamma = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.6, 0.7, 0.8, 0.8, 0.9, 1]
    else:
        my_C, my_gamma = para[0], para[1]
    #=====================================# 
    my_classifier = make_pipeline(MinMaxScaler(),
                                  VarianceThreshold(threshold),
                                  SVC())
    #=====================================#
    # GridsearchCV
    my_parameters_grid = {'C': my_C, 'gamma': my_gamma}
    my_new_parameters_grid = {'svc__' + key: my_parameters_grid[key] for key in my_parameters_grid}
    grid_cv = GridSearchCV(my_classifier, 
                           my_new_parameters_grid, 
                           scoring=scoring_,
                           n_jobs= -1,
                           cv = StratifiedKFold(n_splits=nfold_, shuffle=True, random_state=42),
                           return_train_score=True)
    grid_cv.fit(X_train_, y_train_)
    #=====================================#
    # Create Regressor uing Best Parameters (Use only one option at each run)
    best_C           = grid_cv.best_params_['svc__C']
    best_gamma       = grid_cv.best_params_['svc__gamma']
    cross_validated  = grid_cv.best_score_   
    best_para_set    = [cross_validated, best_C, best_gamma]
    #=====================================#
    my_best_classifier = make_pipeline(MinMaxScaler(), 
                                       VarianceThreshold(threshold),
                                       SVC(C=best_C, gamma=best_gamma, probability=True))
    #=====================================#
    # Testing on train data
    my_best_classifier.fit(X_train_, y_train_)
    y_pred = my_best_classifier.predict(X_train_)
    y_prob = my_best_classifier.predict_proba(test_train_)[::,1]
    #=====================================#
    pred_path = f'./pred/{spec}/svm/'
    os.makedirs(pred_path, exist_ok=True)
    df = pd.DataFrame(y_prob, columns=['Predicted Probability'])
    file_name = f'y_prob_SVM_{tag_}.csv'
    df.to_csv(os.path.join(pred_path, file_name), index=False)
    #=====================================#
    # Evaluation
    x         = printPerformance(test_label_, y_prob)
    print(x)
    x_list    = list(x)
    x_list    = x_list + best_para_set
    new_df    = pd.DataFrame(x_list, columns = [tag_])
    concat_df = pd.concat([concat_df, new_df], axis=1)
    result_path = f'./result/{spec}/svm/'
    os.makedirs(result_path, exist_ok=True)
    file_path = os.path.join(result_path, f'SVM_{tag_}.csv')
    concat_df.to_csv(file_path, index=False)

    para_path = f'./para/{spec}/svm/'
    os.makedirs(para_path, exist_ok=True)
    best_para = pd.DataFrame(best_para_set, columns=['Best_para'])
    best_para.to_csv(os.path.join(para_path, 'Best_para.csv'), index=False)
    
#--------------------------------------------------------------------#
# TUNING XGB MODEL                                                   #
#--------------------------------------------------------------------#
def xgb_tuning(X_train, y_train,test_train,test_label, scoring = 'roc_auc', nfold=5, tag=None, seed=0, para=None):
    X_train_, y_train_ = X_train, y_train
    test_train_, test_label_ = test_train, test_label
    if tag is not None:
        tag_ = tag
    seed_     = seed
    scoring_  = scoring
    nfold_    = nfold
    concat_df = pd.DataFrame(['AUC-ROC', 'AUC-PR','ACC', 'BA', 
                              'SN/RE', 'SP', 'PR', 'MCC', 'F1', 'CK', 
                              'cross_validated',
                              'best_max_depth', 'best_colsample_bytree', 'best_learning_rate'], columns= ["Metrics"]) 
    #=====================================#
    # Set Up Parameter
    if para == None:        
        my_n_estimators     = 200
        my_learning_rate    = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1]
        my_max_depth        = np.arange(3, 10)
        my_colsample_bytree = np.arange(0.2, 0.95, 0.05)
    else:
        my_n_estimators, my_learning_rate, my_max_depth, my_colsample_bytree = para[0], para[1], para[2], para[3]
    #=====================================# 
    my_classifier = make_pipeline(MinMaxScaler(),
                                  VarianceThreshold(threshold),
                                  XGBClassifier(random_state=42, n_estimators=my_n_estimators))
    #=====================================#
    # GridsearchCV
    my_parameters_grid = {'max_depth': my_max_depth, 'learning_rate': my_learning_rate, 'colsample_bytree': my_colsample_bytree}
    my_new_parameters_grid = {'xgbclassifier__' + key: my_parameters_grid[key] for key in my_parameters_grid}
    grid_cv = GridSearchCV(my_classifier, 
                           my_new_parameters_grid, 
                           scoring=scoring_,
                           n_jobs= -1,
                           cv = StratifiedKFold(n_splits=nfold_, shuffle=True, random_state=42),
                           return_train_score=True)
    grid_cv.fit(X_train_, y_train_)
    #=====================================#
    # Create Regressor uing Best Parameters (Use only one option at each run)
    best_max_depth        = grid_cv.best_params_['xgbclassifier__max_depth']
    best_colsample_bytree = grid_cv.best_params_['xgbclassifier__colsample_bytree']
    best_learning_rate    = grid_cv.best_params_['xgbclassifier__learning_rate']
    cross_validated       = grid_cv.best_score_  
    best_para_set         = [cross_validated, best_max_depth, best_colsample_bytree, best_learning_rate]
    #=====================================#
    my_best_classifier = make_pipeline(MinMaxScaler(), 
                                       VarianceThreshold(threshold),
                                       XGBClassifier(random_state=42, 
                                                     n_estimators     = my_n_estimators,
                                                     max_depth        = best_max_depth,
                                                     colsample_bytree = best_colsample_bytree,
                                                     learning_rate    = best_learning_rate))
    #=====================================#
    # Testing on train data
    my_best_classifier.fit(X_train_, y_train_)
    y_pred = my_best_classifier.predict(X_train_)
    y_prob = my_best_classifier.predict_proba(test_train_)[::,1]
    #=====================================#
    pred_path = f'./pred/{spec}/xgb/'
    os.makedirs(pred_path, exist_ok=True)
    df = pd.DataFrame(y_prob, columns=['Predicted Probability'])
    file_name = f'y_prob_XGB_{tag_}.csv'
    df.to_csv(os.path.join(pred_path, file_name), index=False)
    #=====================================#
    # Evaluation
    x         = printPerformance(test_label_, y_prob)
    print(x)
    x_list    = list(x)
    x_list    = x_list + best_para_set
    new_df    = pd.DataFrame(x_list, columns = [tag_])
    concat_df = pd.concat([concat_df, new_df], axis=1)
    result_path = f'./result/{spec}/xgb/'
    os.makedirs(result_path, exist_ok=True)
    file_path = os.path.join(result_path, f'XGB_{tag_}.csv')
    concat_df.to_csv(file_path, index=False)

    para_path = f'./para/{spec}/xgb/'
    os.makedirs(para_path, exist_ok=True)
    best_para = pd.DataFrame(best_para_set, columns=['Best_para'])
    best_para.to_csv(os.path.join(para_path, 'Best_para.csv'), index=False)
#--------------------------------------------------------------------#
# TUNING RF MODEL                                                    #
#--------------------------------------------------------------------#
def rf_tuning(X_train, y_train, test_train,test_label,scoring = 'roc_auc', nfold=5, tag=None, seed=0, para=None):
    X_train_, y_train_ = X_train, y_train
    test_train_, test_label_ = test_train, test_label
    if tag is not None:
        tag_ = tag
    seed_     = seed
    scoring_  = scoring
    nfold_    = nfold
    concat_df = pd.DataFrame(['AUC-ROC', 'AUC-PR','ACC', 'BA', 
                              'SN/RE', 'SP', 'PR', 'MCC', 'F1', 'CK', 
                              'cross_validated',
                              'best_max_depth', 'best_max_features', 'best_min_samples_split'], columns= ["Metrics"]) 
    #=====================================#
    # Set Up Parameter
    if para == None:
        my_n_estimators      = 200
        my_max_depth         = np.arange(2, 10)
        my_max_features      = np.arange(0.2, 0.95, 0.05)
        my_min_samples_split = np.arange(2, 10)
    else:
        my_n_estimators, my_max_depth, my_max_features, my_min_samples_split = para[0], para[1], para[2], para[3]
    #=====================================# 
    my_classifier = make_pipeline(MinMaxScaler(),
                                  VarianceThreshold(threshold),
                                  RandomForestClassifier(random_state=42, n_estimators=my_n_estimators))
    #=====================================#
    # GridsearchCV
    my_parameters_grid = {'max_depth': my_max_depth, 'max_features': my_max_features, 'min_samples_split': my_min_samples_split}
    my_new_parameters_grid = {'randomforestclassifier__' + key: my_parameters_grid[key] for key in my_parameters_grid}
    grid_cv = GridSearchCV(my_classifier, 
                           my_new_parameters_grid, 
                           scoring=scoring_,
                           n_jobs=-1,
                           cv = StratifiedKFold(n_splits=nfold_, shuffle=True, random_state=42), 
                           return_train_score=True)
    grid_cv.fit(X_train_, y_train_)
    #=====================================#
    # Create Regressor uing Best Parameters (Use only one option at each run)
    best_max_depth         = grid_cv.best_params_['randomforestclassifier__max_depth']
    best_max_features      = grid_cv.best_params_['randomforestclassifier__max_features']
    best_min_samples_split = grid_cv.best_params_['randomforestclassifier__min_samples_split']
    cross_validated        = grid_cv.best_score_  
    best_para_set          = [cross_validated, best_max_depth, best_max_features, best_min_samples_split]
    #=====================================#
    my_best_classifier = make_pipeline(MinMaxScaler(), 
                                       VarianceThreshold(threshold),
                                       RandomForestClassifier(random_state      = 42,
                                                              n_estimators      = my_n_estimators, 
                                                              max_depth         = best_max_depth,
                                                              max_features      = best_max_features,
                                                              min_samples_split = best_min_samples_split))  
    #=====================================#
    # Testing on train data
    my_best_classifier.fit(X_train_, y_train_)
    y_pred = my_best_classifier.predict(X_train_)
    y_prob = my_best_classifier.predict_proba(test_train_)[::,1]
    #=====================================#
    pred_path = f'./pred/{spec}/rf/'
    os.makedirs(pred_path, exist_ok=True)
    df = pd.DataFrame(y_prob, columns=['Predicted Probability'])
    file_name = f'y_prob_RF_{tag_}.csv'
    df.to_csv(os.path.join(pred_path, file_name), index=False)
    #=====================================#
    # Evaluation
    x         = printPerformance(test_label_, y_prob)
    print(x)
    x_list    = list(x)
    x_list    = x_list + best_para_set
    new_df    = pd.DataFrame(x_list, columns = [tag_])
    concat_df = pd.concat([concat_df, new_df], axis=1)
    result_path = f'./result/{spec}/rf/'
    os.makedirs(result_path, exist_ok=True)
    file_path = os.path.join(result_path, f'RF_{tag_}.csv')
    concat_df.to_csv(file_path, index=False)

    para_path = f'./para/{spec}/rf/'
    os.makedirs(para_path, exist_ok=True)
    best_para = pd.DataFrame(best_para_set, columns=['Best_para'])
    best_para.to_csv(os.path.join(para_path, 'Best_para.csv'), index=False)
