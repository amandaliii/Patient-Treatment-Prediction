# pytorch tools for tensors, dataset handling, and neural network layers
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
# to calculate precision, recall, and f1 score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# evaluate model on validation set with metrics calculation
def evaluate_model(model, dataloader, loss_fn, device):
    # set model to evaluation mode disables dropout etc.
    model.eval()
    total_loss = 0
    # collect predictions
    all_preds = []
    # collect true targets
    all_targets = []
    # disable gradient calculations for efficiency
    with torch.no_grad():
        for inputs, targets in dataloader:
            # move data to device (CPU/GPU)
            inputs, targets = inputs.to(device), targets.to(device)
            # forward pass
            output = model(inputs)
            # calculate loss for the batch
            loss = loss_fn(output, targets)
            # calculate total loss
            total_loss += loss.item()
            # get predicted indices
            preds = output.argmax(dim=-1).cpu().numpy()
            # append to list
            all_preds.extend(preds)
            # append ground truth
            all_targets.extend(targets.cpu().numpy())
    # average loss over all batches
    avg_loss = total_loss / len(dataloader)
    # calculate accuracy, precision, recall and F1 (macro averaged, ignoring divide by zero warnings)
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    return avg_loss, accuracy, precision, recall, f1

# train the model with train/validation split and class weights for imbalanced classes
def train_model(model, train_loader, val_loader, class_weights, epochs, lr=1e-4):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
    model = model.to(device)  # Move model to device
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))  # Use weighted cross-entropy loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # Adam optimizer with weight decay for regularization
    # list to collect training/validation stats
    metrics = []

    for epoch in range(epochs):
        # set model to training mode
        model.train()
        total_train_loss = 0
        for inputs, targets in train_loader:
            # move batch to device
            inputs, targets = inputs.to(device), targets.to(device)
            # reset gradients
            optimizer.zero_grad()
            # forward pass
            output = model(inputs)
            # calculate loss
            loss = loss_fn(output, targets)
            # backpropagation
            loss.backward()
            # gradient clipping to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # update parameters
            optimizer.step()
            # accumulate loss for reporting
            total_train_loss += loss.item()
        # average loss per batch
        avg_train_loss = total_train_loss / len(train_loader)

        # validation evaluation
        avg_val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, loss_fn, device)
        # save metrics for this epoch; 'Category' will be assigned later in main code loop
        metrics.append({
            "Epoch": epoch + 1,
            "Category": None,
            "Train_Loss": avg_train_loss,
            "Val_Loss": avg_val_loss,
            "Val_Accuracy": val_accuracy,
            "Val_Precision": val_precision,
            "Val_Recall": val_recall,
            "Val_F1": val_f1
        })
        # print progress info
        print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"Val Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
    return metrics
