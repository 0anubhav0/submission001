import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.figure_factory import create_distplot
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report, confusion_matrix, fbeta_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from collections import Counter

data = pd.read_csv('german_credit_data.csv')
data.drop(["Unnamed: 0"], axis=1, inplace=True)

categoric_vars_list = ["Sex", "Job", "Housing", "Saving accounts", "Checking account", "Purpose", "Risk"]
df = data.copy()
df = pd.get_dummies(df, columns=categoric_vars_list[:-1], drop_first=True)
df['y'] = df['Risk'].apply(lambda x: 0 if x == 'good' else 1)
y = df["y"]
df.drop(["Risk", "y"], axis=1, inplace=True)
X = df

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

classifierMap = {}

log_reg = LogisticRegression(solver='liblinear', random_state=42)
param_grid = {
    'C': [0.01, 0.1, 1, 1.5, 2, 3, 4, 5, 10, 100],
    'penalty': ['l1', 'l2']
}
grid_search = GridSearchCV(log_reg, param_grid, scoring='roc_auc', cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_lr = grid_search.best_estimator_
classifierMap["lr"] = best_lr
y_pred_lr = best_lr.predict(X_test)
y_pred_prob_lr = best_lr.predict_proba(X_test)[:, 1]

dt = DecisionTreeClassifier(random_state=42)
param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search_dt = GridSearchCV(dt, param_grid, scoring='roc_auc', cv=5, verbose=1, n_jobs=-1)
grid_search_dt.fit(X_train, y_train)
best_dt = grid_search_dt.best_estimator_
classifierMap["dt"] = best_dt
y_pred_dt = best_dt.predict(X_test)
y_pred_prob_dt = best_dt.predict_proba(X_test)[:, 1]

rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search_rf = GridSearchCV(rf, param_grid, scoring='roc_auc', cv=5, verbose=1, n_jobs=-1)
grid_search_rf.fit(X_train, y_train)
best_rf = grid_search_rf.best_estimator_
classifierMap["rf"] = best_rf
y_pred_rf = best_rf.predict(X_test)
y_pred_prob_rf = best_rf.predict_proba(X_test)[:, 1]

svm = SVC(probability=True, random_state=42)
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}
grid_search_svm = GridSearchCV(svm, param_grid, scoring='roc_auc', cv=5, verbose=1, n_jobs=-1)
grid_search_svm.fit(X_train, y_train)
best_svm = grid_search_svm.best_estimator_
classifierMap["svm"] = best_svm
y_pred_svm = best_svm.predict(X_test)
y_pred_prob_svm = best_svm.predict_proba(X_test)[:, 1]

best_model = None
max_auc = 0.0

def bestFinder(y_test, y_pred_prob, model_name):
    global best_model, max_auc
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    print(f"AUC for {model_name}: {roc_auc:.4f}")
    if roc_auc > max_auc:
        max_auc = roc_auc
        best_model = classifierMap[model_name]

bestFinder(y_test, y_pred_prob_dt, "dt")
bestFinder(y_test, y_pred_prob_rf, "rf")
bestFinder(y_test, y_pred_prob_lr, "lr")
bestFinder(y_test, y_pred_prob_svm, "svm")

if best_model:
    joblib.dump(best_model, 'best_model.pkl')
    print("Best model saved as 'best_model.pkl'")
else:
    print("No model selected as best.")
