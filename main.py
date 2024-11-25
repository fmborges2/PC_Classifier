import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import numpy as np
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from ocpc_py import MultiClassPC
import warnings
warnings.filterwarnings('ignore')


file_path = '/home/rafael/orange_dataset/data/Sensorial todos tratamento_with_parameters.xlsx'
data = pd.read_excel(file_path)

label_counts = data['Intenção de compra'].value_counts().sort_index()
print(label_counts)
data = data.dropna(axis=1)
colunas_com_infinitos = data.columns[data.isin([np.inf, -np.inf]).any()]

data = data.drop(columns=colunas_com_infinitos)
data
y = data['Intenção de compra']
X = data.drop(columns=['Intenção de compra'], inplace=True)
X = data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

oversampling = True

if oversampling:
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    print("Contagens após o oversampling:")
    print(pd.Series(y_train).value_counts().sort_index())


def run_model(model):

    scores = cross_val_score(model, X_train, y_train, cv=10)
    mean_score = scores.mean()
    std_dev_score = np.std(scores)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    y_pred_proba = model.predict_proba(X_test)

    # Calcular a AUC usando a abordagem One-vs-Rest (OvR)
    auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)

    print(f'AUC no conjunto de teste: {auc_score:.2f}')
    print(f'Validação Cruzada (10 folds) - Acurácia média: {mean_score:.2f}')
    print(f'Acurácia no conjunto de teste: {test_accuracy:.2f}')

    plt.figure(figsize=(5, 5))
    sns.boxplot(scores)
    plt.title('Boxplot of Cross Validation Accuracies')
    plt.xlabel('Scores')
    plt.show()

    mean_score = round(mean_score, 4)
    std_dev_score = round(std_dev_score, 4)
    auc_score = round(auc_score, 4)
    recall = round(recall.mean(), 4)
    precision = round(precision.mean(), 4)
    f1 = round(f1.mean(), 4)
    test_accuracy = round(test_accuracy, 4)
    return scores, mean_score, std_dev_score, auc_score, recall, precision, f1, test_accuracy


model = MultiClassPC(k_max=10, alfa=0.1, lamda=0.5, buffer=200)

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = []
auc_scores = []

for train_index, val_index in kf.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train.iloc[train_index].to_numpy(), X_train.iloc[val_index].to_numpy()
    y_train_fold, y_val_fold = y_train.iloc[train_index].to_numpy(), y_train.iloc[val_index].to_numpy()

    model.fit(X_train_fold, y_train_fold)

    y_val_pred = model.predict(X_val_fold)

    accuracy = accuracy_score(y_val_fold, y_val_pred)
    scores.append(accuracy)

    y_val_proba = model.predict_proba(X_val_fold)
    auc = roc_auc_score(pd.get_dummies(y_val_fold), y_val_proba, multi_class='ovr')
    auc_scores.append(auc)

model.fit(X_train.to_numpy(), y_train.to_numpy())

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

test_auc = roc_auc_score(pd.get_dummies(y_test), y_pred_proba, multi_class='ovr')

test_accuracy = accuracy_score(y_test, y_pred)

auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)

print(f'Validação Cruzada (10 folds) - Acurácia média: {np.mean(scores):.2f}')
print(f'AUC média na validação: {np.mean(auc_scores):.2f}')
print(f'Acurácia no conjunto de teste: {test_accuracy:.2f}')
print(f'AUC no conjunto de teste: {test_auc:.2f}')

plt.figure(figsize=(10, 6))
sns.boxplot(scores)
plt.title('Boxplot dos Scores de Validação Cruzada')
plt.xlabel('Acurácia')
plt.legend()
plt.show()

scores_ocpc = scores
mean_score_ocpc = round(np.mean(scores), 4)
std_dev_score_ocpc = round(np.std(scores), 4)
auc_score_ocpc = round(auc_score, 4)
recall_ocpc = round(recall.mean(), 4)
precision_ocpc = round(precision.mean(), 4)
f1_ocpc = round(f1.mean(), 4)
test_accuracy_ocpc = round(test_accuracy, 4)