import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import ensemble, metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc, accuracy_score
from sklearn.preprocessing import label_binarize
import seaborn as sns

# Selecting the specified columns
#feature_table = pd.read_csv("new_data - Sheet1.csv")
#labels = pd.read_csv("final_dataset_labels - Sheet1.csv")
feature_table = pd.read_csv("no_aug_dataset - Sheet1.csv")
labels = pd.read_csv("no_aug_labels - Sheet1.csv")

classes = ['Rock', 'Paper', 'Scissors', 'Rest']
selected_columns = [0,1,2,3,4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14,15,16, 17, 18, 19]
features = feature_table.iloc[:, selected_columns]
labels = labels.iloc[:, 0]

# Convert to PyTorch tensors
features_tensor = torch.tensor(features.values, dtype=torch.float32)
labels_tensor = torch.tensor(labels.values, dtype=torch.long)

# Splitting the dataset into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(
#    features_tensor, labels_tensor, train_size=5770, test_size=2000, stratify=labels_tensor, random_state=42
#)
X_train, X_test, y_train, y_test = train_test_split(
    features_tensor, labels_tensor, train_size=3389, test_size=498, stratify=labels_tensor, random_state=42
)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Classifiers to be used
classifiers = {
    "AdaBoostClassifier": ensemble.AdaBoostClassifier(),
    "BaggingClassifier": ensemble.BaggingClassifier(),
    "ExtraTreesClassifier": ensemble.ExtraTreesClassifier(),
    "GradientBoostingClassifier": ensemble.GradientBoostingClassifier(),
    "HistGradientBoostingClassifier": ensemble.HistGradientBoostingClassifier(),
    "RandomForestClassifier (Optimized) n=491, Max_leaf = 807": ensemble.RandomForestClassifier(n_estimators=491, max_leaf_nodes=807, random_state=42),
    "RandomForestClassifier n=200, Max_leaf = 400": ensemble.RandomForestClassifier(n_estimators=200, max_leaf_nodes=400 , random_state=42),

    "RandomForestClassifier (default) n=100, Max_leaf = None": ensemble.RandomForestClassifier(random_state=42)
}

# For storing f1 scores
f1_scores = []
results = []
# K-Fold Cross-validation
kf = KFold(n_splits=5)

for name, clf in classifiers.items():
    f1_cv_scores = []
    accuracy_cv_scores = []
    
    for train_index, val_index in kf.split(X_train):
        X_train_kf, X_val_kf = X_train[train_index], X_train[val_index]
        y_train_kf, y_val_kf = y_train[train_index], y_train[val_index]

        clf.fit(X_train_kf, y_train_kf)
        y_pred_kf = clf.predict(X_val_kf)

        # Calculate F1 score and accuracy
        f1 = f1_score(y_val_kf, y_pred_kf, average='weighted')
        accuracy = accuracy_score(y_val_kf, y_pred_kf)
        f1_cv_scores.append(f1)
        accuracy_cv_scores.append(accuracy)


    # Average F1 score and accuracy from CV
    avg_f1_cv = np.mean(f1_cv_scores)
    avg_accuracy_cv = np.mean(accuracy_cv_scores)
    results.append((name+"Cross_val", avg_f1_cv, avg_accuracy_cv))
    # Resubstitution F1 score and accuracy
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    f1 = f1_score(y_train, y_pred_train, average='weighted')
    accuracy = accuracy_score(y_train, y_pred_train)
    results.append((name+"Resub", f1, accuracy))

    # ROC Curve
    y_train_bin = label_binarize(y_train, classes=np.unique(labels_tensor))
    y_pred_train_bin = label_binarize(y_pred_train, classes=np.unique(labels_tensor))
    fpr, tpr, _ = roc_curve(y_train_bin.ravel(), y_pred_train_bin.ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title(f'ROC Curve for {name}Resubstitution')
    plt.legend(loc='lower right')
    plt.savefig(f'data/roc_curve_{name}_resub.png')

            # Confusion Matrix
    cm = confusion_matrix(y_train, y_pred_train)
    plt.figure(figsize=(10, 7))


    # Plotting using seaborn for better visualization
    sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='icefire')
    tick_marks = np.arange(len(classes)) # classes = ['Rock', 'Paper', 'Scissors', 'Rest']
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    plt.title(f'Confusion Matrix for {name} Resubstitution')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
        
    plt.savefig(f'data/conf_matrix_{name}_Resub.png')




    # Training the model on full training data
    clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test)

    # Test F1 score and accuracy
    f1_test = f1_score(y_test, y_pred_test, average='weighted')
    accuracy_test = accuracy_score(y_test, y_pred_test)
    results.append((name+"Test", f1_test, accuracy_test))
    # ROC Curve
    y_test_bin = label_binarize(y_test, classes=np.unique(labels_tensor))
    y_pred_test_bin = label_binarize(y_pred_test, classes=np.unique(labels_tensor))
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_pred_test_bin.ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title(f'ROC Curve for {name} Test')
    plt.legend(loc='lower right')
    plt.savefig(f'data/roc_curve_{name}_test.png')

            # Confusion Matrix
    cm = confusion_matrix(y_val_kf, y_pred_kf)
    plt.figure(figsize=(10, 7))


    # Plotting using seaborn for better visualization
    sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='icefire')
    tick_marks = np.arange(len(classes)) # classes = ['Rock', 'Paper', 'Scissors', 'Rest']
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    plt.title(f'Confusion Matrix for {name} Test')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
        
    plt.savefig(f'data/conf_matrix_{name}_test.png')

# Save F1 scores to a CSV
results_df = pd.DataFrame(results, columns=['Classifier', 'F1 Score', 'Accuracy'])
results_df.to_csv('data/results.csv', index=False)