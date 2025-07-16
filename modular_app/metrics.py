from torchmetrics import Accuracy, F1Score, Precision, Recall

def set_metrics(num_classes, task="multiclass", device="cpu"):
    accuracy_score = Accuracy(task=task, num_classes=num_classes).to(device)
    f1_score = F1Score(task=task, num_classes=num_classes).to(device)
    precision = Precision(task=task, num_classes=num_classes).to(device)
    recall = Recall(task=task, num_classes=num_classes).to(device)
    return accuracy_score, f1_score, precision, recall

def classification_report(num_classes, y_true, y_pred, task="multiclass", device="cpu"):
    accuracy_score, f1_score, precision, recall = set_metrics(num_classes, task="multiclass", device="cpu")
    report = {
        "Accuracy":     accuracy_score(y_pred, y_true).item(),
        "Precision":    precision(y_pred, y_true).item(),
        "Recall":       recall(y_pred, y_true).item(),
        "F1-Score":     f1_score(y_pred, y_true).item()
        }
    return report