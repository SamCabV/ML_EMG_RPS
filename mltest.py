import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import pandas as pd

# Selecting the specified columns
feature_table = pd.read_csv("new_data - Sheet1.csv")
labels = pd.read_csv("final_dataset_labels - Sheet1.csv")

selected_columns = [4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19]
features = feature_table.iloc[:, selected_columns]
labels = labels.iloc[:, 0]

# Convert to PyTorch tensors
features_tensor = torch.tensor(features.values, dtype=torch.float32)
labels_tensor = torch.tensor(labels.values, dtype=torch.long)

# Splitting the dataset into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(
    features_tensor, labels_tensor, train_size=4376, test_size=1458, stratify=labels_tensor, random_state=42
)


# Neural network model with modifications
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_sigmoid_stack = nn.Sequential(
            nn.Linear(12, 297),  # 12 features input, hidden layer with 297 neurons
            nn.Sigmoid(),
            nn.Linear(297, 4)    # Output layer with 4 class labels
        )

    def forward(self, x):
        logits = self.linear_sigmoid_stack(x)
        return logits

# Initialize the neural network
model = NeuralNetwork()

# Loss function and optimizer with weight decay for regularization
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5.332349054522264e-06)

# K-Fold Cross Validation
kf = KFold(n_splits=5)
fold_results = []

for train_index, val_index in kf.split(train_features):
    X_train, X_val = train_features[train_index], train_features[val_index]
    y_train, y_val = train_labels[train_index], train_labels[val_index]

    # Convert to dataloaders
    train_data = TensorDataset(X_train, y_train)
    val_data = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=4, shuffle=False)

    # Training
    for epoch in range(1000):  # Training for 1000 epochs to match MATLAB's iteration limit
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    fold_results.append(100 * correct / total)

# Testing
test_data = TensorDataset(test_features, test_labels)
test_loader = DataLoader(test_data, batch_size=4, shuffle=False)
model.eval()
test_preds = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        test_preds.extend(predicted.numpy())

# Confusion Matrix and Accuracy
conf_mat = confusion_matrix(test_labels, test_preds)
test_accuracy = accuracy_score(test_labels, test_preds)
fold_accuracy = np.mean(fold_results)

print(conf_mat)
model_file_name = 'my_model.pth'  # You can choose your own file name
torch.save(model.state_dict(), model_file_name)

print(f"Model saved as {model_file_name}")

