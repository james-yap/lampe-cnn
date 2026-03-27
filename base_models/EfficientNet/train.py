import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Subset
import numpy as np

from model import EfficientNet
from process_data_k_fold import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

k_folds = 2
num_epochs = 15

# initialising StratifiedGroupKFold with shuffling
sgkf = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=42)

# Lists to store results for each fold
fold_accuracies = []
best_val_accuracy_overall = 0.0
best_model_state_dict = None
# lists to store training and validation losses for the best fold (the highest validation accuracy)
best_fold_train_losses = []
best_fold_val_losses = []

print(f"Starting K-Fold Cross Validation with {k_folds} folds...")

X_index = np.arange(len(train_val_dataset))
for fold, (train_ids, val_ids) in enumerate(sgkf.split(X_index, train_val_labels, groups=train_val_groups)):
    print(f"\nFOLD {fold+1}/{k_folds}")

    # Re-initialize model for each fold
    model = EfficientNet(num_classes=4, unfreeze_blocks=1).to(device)
    # Use label smoothing in the loss function to help with generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    # Only update the parameters of the last layer, lower learning rate and add weight decay for better generalization and to prevent overfitting
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-5, weight_decay=1e-5)

    # Create data loaders for current fold
    train_subset_fold = TransformedSubset(Subset(train_val_dataset, train_ids.tolist()), transform=train_transforms)
    val_subset_fold = TransformedSubset(Subset(train_val_dataset, val_ids.tolist()), transform=val_transforms)

    train_loader_fold = torch.utils.data.DataLoader(train_subset_fold, batch_size=32, shuffle=True)
    val_loader_fold = torch.utils.data.DataLoader(val_subset_fold, batch_size=32, shuffle=False)

    best_acc_fold = 0.0
    train_losses_fold = []
    val_losses_fold = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Training loop
        for images, labels in train_loader_fold:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader_fold:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total
        train_losses_fold.append(running_loss)
        val_losses_fold.append(val_loss)
        print(f"Epoch {epoch+1}: Train Loss={running_loss}: , Val Loss={val_loss}: , Val Acc={val_accuracy}: ")

        # Save the best model for this fold
        if val_accuracy > best_acc_fold:
            best_acc_fold = val_accuracy
            if val_accuracy > best_val_accuracy_overall:
                best_val_accuracy_overall = val_accuracy
                best_model_state_dict = model.state_dict()
                best_fold_train_losses = train_losses_fold.copy()
                best_fold_val_losses = val_losses_fold.copy()
                torch.save(best_model_state_dict, "best_model_overall.pth")

    fold_accuracies.append(best_acc_fold)


print(f"Average Fold Accuracy: {sum(fold_accuracies) / k_folds}: ")
print(f"Best Validation Accuracy Across all Folds: {best_val_accuracy_overall}: ")

# Loading the overall best model and evaluating on the permanent test set
if best_model_state_dict:
    model = EfficientNet(num_classes=4, unfreeze_blocks=1).to(device)
    model.load_state_dict(torch.load("best_model_overall.pth"))
    model.eval()

    correct = 0
    total = 0
    # with no gradient computing for evaluation
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total
    print(f"Permanent Test Set Accuracy with Best K-Fold Model: {test_accuracy}: ")
else:
    print("No best model found. Something went wrong during training.")



