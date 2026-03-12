import torch
import torch.nn as nn

# get the model and dataset from the respective files
from model import ResNet18
from process_data import train_loader, val_loader, test_loader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18(num_classes=4).to(device)


# defining loss and optimizer
criterion = nn.CrossEntropyLoss()
# only update the parameters of the last layer
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
# optional: learning rate scheduler
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

num_epochs = 10
# train - patterns, validation - fine tuning, test - evaluation

best_acc = 0.0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    print(epoch)
    # training loop
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # forward + backward + optimize
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()



    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = correct / total
    print(f"Epoch {epoch+1}: Train Loss={running_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.4f}")




    # saving the best model
    if val_accuracy > best_acc:
        best_acc = val_accuracy
        torch.save(model.state_dict(), "best_model.pth")



# test-evaluation

# Load the best model and evaluate on the test set
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

correct = 0
total = 0
# No need to compute gradients during evaluation
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = correct / total
print("Test Accuracy:", test_accuracy)