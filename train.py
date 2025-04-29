import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import get_data_loaders, get_kfold_data_loaders
from model import DeepfakeDetector, FocalLoss
import torchvision.transforms as transforms
import copy
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import numpy as np

def evaluate(model, loader, device, return_probs=False):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if return_probs:
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
    acc = 100 * correct / total
    if return_probs:
        auc = roc_auc_score(all_labels, all_probs)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, np.round(all_probs), average='binary')
        return acc, auc, precision, recall, f1
    return acc

def train_model(use_kfold=False, k_folds=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepfakeDetector().to(device)
    criterion = FocalLoss(gamma=2.0, alpha=0.25)
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)
    warmup_epochs = 5
    warmup_lr = 1e-5

    if use_kfold:
        data_loaders = get_kfold_data_loaders('data/real', 'data/fake', batch_size=64, k_folds=k_folds)
    else:
        train_loader, val_loader = get_data_loaders('data/real', 'data/fake', batch_size=64)

    num_epochs = 100
    best_val_acc = 0.0
    patience = 10
    epochs_no_improve = 0
    best_model_state = None

    if use_kfold:
        fold_results = []
        for fold, (train_loader, val_loader) in enumerate(data_loaders):
            print(f'\nFold {fold+1}/{k_folds}')
            model = DeepfakeDetector().to(device)
            optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-5)
            scheduler = CosineAnnealingLR(optimizer, T_max=50)
            best_val_acc = 0.0
            epochs_no_improve = 0
            best_model_state = None

            for epoch in range(num_epochs):
                # Learning rate warmup
                if epoch < warmup_epochs:
                    lr = warmup_lr + (0.0003 - warmup_lr) * (epoch / warmup_epochs)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                train_acc = 100 * correct / total

                val_acc, val_auc, val_precision, val_recall, val_f1 = evaluate(model, val_loader, device, return_probs=True)
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, '
                      f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Val AUC: {val_auc:.4f}, '
                      f'Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}')

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = copy.deepcopy(model.state_dict())
                    torch.save(best_model_state, f'models/saved_model_fold{fold+1}.pth')
                    print(f'Saved model for fold {fold+1} with validation accuracy: {val_acc:.2f}%')
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f'Early stopping triggered after {epoch+1} epochs')
                        break

                if epoch >= warmup_epochs:
                    scheduler.step()

            model.load_state_dict(best_model_state)
            fold_results.append((best_val_acc, val_auc, val_precision, val_recall, val_f1))
        avg_acc = np.mean([res[0] for res in fold_results])
        print(f'\nK-Fold Results: Avg Acc: {avg_acc:.2f}%, Avg AUC: {np.mean([res[1] for res in fold_results]):.4f}')
        # Save the best model from the fold with highest accuracy
        best_fold = np.argmax([res[0] for res in fold_results]) + 1
        best_model_state = torch.load(f'models/saved_model_fold{best_fold}.pth')
        torch.save(best_model_state, 'models/saved_model.pth')
    else:
        for epoch in range(num_epochs):
            if epoch < warmup_epochs:
                lr = warmup_lr + (0.0003 - warmup_lr) * (epoch / warmup_epochs)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            train_acc = 100 * correct / total

            val_acc, val_auc, val_precision, val_recall, val_f1 = evaluate(model, val_loader, device, return_probs=True)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, '
                  f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Val AUC: {val_auc:.4f}, '
                  f'Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
                torch.save(best_model_state, 'models/saved_model.pth')
                print(f'Saved model with validation accuracy: {val_acc:.2f}%')
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    break

            if epoch >= warmup_epochs:
                scheduler.step()

        model.load_state_dict(best_model_state)
        print(f'Training complete. Best validation accuracy: {best_val_acc:.2f}%')

        # Evaluate on test set if available
        try:
            test_loader = get_data_loaders('data/test_real', 'data/test_fake', batch_size=64)[1]
            test_acc, test_auc, test_precision, test_recall, test_f1 = evaluate(model, test_loader, device, return_probs=True)
            print(f'Test Acc: {test_acc:.2f}%, Test AUC: {test_auc:.4f}, '
                  f'Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}')
        except FileNotFoundError:
            print("Test set directories not found. Skipping test evaluation.")

if __name__ == '__main__':
    train_model(use_kfold=False)  # Set to True for k-fold cross-validation