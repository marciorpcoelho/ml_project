import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn import tree, linear_model, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier

df_original_path = 'DatasetML.csv'
df_before_dummies_path = 'DatasetML_processed_before_dummies.csv'
df_processed_path = 'DatasetML_processed.csv'
models_path = 'models/'
plots_path = 'plots/'

k, gridsearch_score = 10, 'recall'

classification_models = {
    'dt': [tree.DecisionTreeClassifier, [{'min_samples_leaf': [3, 5, 7, 9, 10, 15, 20, 30], 'max_depth': [3, 5, 6], 'class_weight': ['balanced']}]],
    'rf': [RandomForestClassifier, [{'n_estimators': [3, 5, 10, 25, 50, 100, 200, 500], 'max_depth': [3, 5, 10, 20], 'class_weight': ['balanced']}]],
    'lr': [linear_model.LogisticRegression, [{'C': np.logspace(-2, 2, 20), 'solver': ['liblinear']}]],
    'knn': [KNeighborsClassifier, [{'n_neighbors': np.arange(1, 50, 1)}]],
    'svm': [svm.SVC, [{'C': np.logspace(-2, 2, 10)}, {'probability': [True]}]],
    'ab': [AdaBoostClassifier, [{'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}]],
    'gc': [GradientBoostingClassifier, [{'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}]],
    'xgb': [xgb.XGBClassifier, [{'objective': ['binary:logistic'], 'booster': ['gbtree'], 'max_depth': [5, 10, 20, 50, 100], 'reg_alpha': [1e-5, 1e-2, 0.1, 1, 10, 100], 'reg_lambda': [1, 2, 5]}]],
    'lgb': [lgb.LGBMClassifier, [{'num_leaves': [10, 15, 30, 50, 100], 'n_estimators': [25, 50, 100, 200, 500], 'objective': ['binary'], 'metric': ['auc'], 'reg_alpha': [1e-5, 1e-2, 0.1, 1, 10, 100], 'reg_lambda': [1, 2, 5]}]],
}

c2_dict = {
    'Z33/Z30/Z31': ['Z33', 'Z30', 'Z31'],
}

c4_dict = {
    'Z62/Z63/Z64/Z65': ['Z62', 'Z63', 'Z64', 'Z65'],
}

c5_1_dict = {
    'Z91/Z94': ['Z91', 'Z94'],
}

c9_dict = {
    'Z151/Z153': ['Z151', 'Z153'],
}

c10_dict = {
    'Z171/Z172/Z174': ['Z171', 'Z172', 'Z174'],
}
