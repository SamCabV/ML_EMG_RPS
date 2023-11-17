import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, accuracy_score
# Selecting the specified columns
feature_table = pd.read_csv("new_data - Sheet1.csv")
print("FUCK")
labels = pd.read_csv("final_dataset_labels - Sheet1.csv")

selected_columns = [0,1,2,3,4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14,15,16, 17, 18, 19]
features = feature_table.iloc[:, selected_columns]
labels = labels.iloc[:, 0]

# Convert to PyTorch tensors
features_tensor = torch.tensor(features.values, dtype=torch.float32)
labels_tensor = torch.tensor(labels.values, dtype=torch.long)

# Splitting the dataset into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(
    features_tensor, labels_tensor, train_size=7770, test_size=7, stratify=labels_tensor, random_state=42
)


model = RandomForestClassifier(n_estimators=491, max_leaf_nodes=807, max_features=3, random_state=42)

# K-Fold Cross Validation
kf = KFold(n_splits=5)
fold_results = []

for train_index, val_index in kf.split(train_features):
    # Indexing tensors directly
    X_train, X_val = train_features[train_index], train_features[val_index]
    y_train, y_val = train_labels[train_index], train_labels[val_index]

    # Convert to numpy arrays for sklearn compatibility
    X_train_np, y_train_np = X_train.numpy(), y_train.numpy()
    X_val_np, y_val_np = X_val.numpy(), y_val.numpy()

    model.fit(X_train_np, y_train_np)
    val_predictions = model.predict(X_val_np)
    fold_accuracy = accuracy_score(y_val_np, val_predictions)
    fold_results.append(fold_accuracy)

# Testing
test_predictions = model.predict(test_features)
test_accuracy = accuracy_score(test_labels, test_predictions)

# Confusion Matrix
conf_mat = confusion_matrix(test_labels, test_predictions)

# Display Results
print("Confusion Matrix:\n", conf_mat)
print("Test Accuracy:", test_accuracy)
print("Average Fold Accuracy:", np.mean(fold_results))

# Model Saving (Optional)
import joblib
model_file_name = 'my_random_forest_model.joblib'
joblib.dump(model, model_file_name)
print(f"Model saved as {model_file_name}")