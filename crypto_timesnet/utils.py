import torch
import numpy as np
import random
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import os


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_class_weights(train_dataset):
    """计算类别权重（对抗不平衡）"""
    labels = []
    for i in range(len(train_dataset)):
        labels.append(train_dataset[i]['label'].item())
    
    labels = np.array(labels)
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    # 逆频率权重
    weights = torch.zeros(3)
    for label, count in zip(unique, counts):
        weights[label] = total / (3 * count)
    
    # 归一化
    weights = weights / weights.sum() * 3
    
    print(f"\n【类别权重】")
    for i, w in enumerate(weights):
        print(f"  Label {i}: {w:.4f}")
    
    return weights


def evaluate_model(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            x_enc = batch['x_enc'].to(device)
            x_mark_enc = batch['x_mark_enc'].to(device)
            labels = batch['label'].squeeze().to(device)
            
            outputs = model(x_enc, x_mark_enc)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    # 计算各类别召回率
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    recalls = []
    for i in range(3):
        if cm[i].sum() > 0:
            recalls.append(cm[i, i] / cm[i].sum())
        else:
            recalls.append(0.0)
    
    return {
        'loss': avg_loss,
        'macro_f1': macro_f1,
        'recalls': recalls,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels
    }


def print_evaluation_report(results, phase='Validation'):
    """打印评估报告"""
    print(f"\n【{phase} Report】")
    print(f"  Loss: {results['loss']:.4f}")
    print(f"  Macro F1: {results['macro_f1']:.4f}")
    print(f"  Recall Label 0: {results['recalls'][0]:.2%}")
    print(f"  Recall Label 1: {results['recalls'][1]:.2%}")
    print(f"  Recall Label 2: {results['recalls'][2]:.2%}")
    
    print(f"\n  Confusion Matrix:")
    print(results['confusion_matrix'])


def check_survival_conditions(results, config):
    """检查生存条件"""
    min_recall = config.train.min_recall_minority
    
    recall_1 = results['recalls'][1]
    recall_2 = results['recalls'][2]
    
    survival = recall_1 >= min_recall and recall_2 >= min_recall
    
    return survival, (recall_1, recall_2)


def save_checkpoint(model, optimizer, epoch, best_f1, config, filename='checkpoint.pth'):
    """保存检查点"""
    os.makedirs(config.train.save_dir, exist_ok=True)
    filepath = os.path.join(config.train.save_dir, filename)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_f1': best_f1,
        'config': config
    }, filepath)
    
    print(f"  ✓ Checkpoint saved: {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    """加载检查点"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['best_f1']
