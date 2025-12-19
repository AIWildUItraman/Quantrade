"""
【ORACLE 生存训练系统】
虚拟货币放量信号检测 - TimesNet
"""

import torch
import torch.nn as nn
import os
import sys
from tqdm import tqdm

from config import get_config
from data_processor import DataProcessor
from model import TimesNetModel, FocalLoss
from utils import (
    set_seed, calculate_class_weights, evaluate_model,
    print_evaluation_report, check_survival_conditions,
    save_checkpoint, load_checkpoint
)
from inference_full import run_inference


class Trainer:
    """
    【训练控制器】
    """
    def __init__(self, config):
        self.config = config
        set_seed(config.seed)
        
        # 设备
        self.device = torch.device(config.train.device)
        print(f"\n【系统初始化】")
        print(f"  设备: {self.device}")
        print(f"  随机种子: {config.seed}")
        
        # 数据处理
        self.processor = DataProcessor(config)
        train_df, val_df, test_df = self.processor.load_and_split()
        self.train_loader, self.val_loader, self.test_loader, self.train_dataset = \
            self.processor.create_dataloaders(train_df, val_df, test_df)
        
        # 模型
        print(f"\n【模型构建】")
        print(f"  架构: TimesNet")
        print(f"  输入特征数: {config.model.enc_in}")
        print(f"  序列长度: {config.model.seq_len}")
        print(f"  模型维度: {config.model.d_model}")
        print(f"  TimesBlock层数: {config.model.e_layers}")
        
        self.model = TimesNetModel(config.model).to(self.device)
        
        # 损失函数
        if config.train.use_class_weights:
            class_weights = calculate_class_weights(self.train_dataset).to(self.device)
        else:
            class_weights = None
        
        self.criterion = FocalLoss(
            alpha=class_weights,
            gamma=config.train.focal_gamma
        )
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay
        )
        
        # 学习率调度器
        if config.train.scheduler_type == 'cosine':
            # Use smooth cosine decay without warm restarts to avoid LR jumps that destabilize metrics
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.train.epochs,
                eta_min=config.train.learning_rate * 0.05
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        
        # 训练状态
        self.best_f1 = 0
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'val_recall_1': [],
            'val_recall_2': []
        }
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        for batch_idx, batch in enumerate(pbar):
            x_enc = batch['x_enc'].to(self.device)
            x_mark_enc = batch['x_mark_enc'].to(self.device)
            labels = batch['label'].squeeze().to(self.device)
            
            # 前向传播
            outputs = self.model(x_enc, x_mark_enc)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.train.grad_clip
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            if batch_idx % self.config.train.log_interval == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def train(self):
        """完整训练流程"""
        print(f"\n{'='*60}")
        print("【开始训练】")
        print(f"{'='*60}")
        
        for epoch in range(self.config.train.epochs):
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_results = evaluate_model(
                self.model, self.val_loader,
                self.criterion, self.device
            )
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_results['loss'])
            self.history['val_f1'].append(val_results['macro_f1'])
            self.history['val_recall_1'].append(val_results['recalls'][1])
            self.history['val_recall_2'].append(val_results['recalls'][2])
            
            # 打印结果
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.config.train.epochs}")
            print(f"{'='*60}")
            print(f"  Train Loss: {train_loss:.4f}")
            print_evaluation_report(val_results, phase='Validation')
            
            # 检查生存条件
            survival, (recall_1, recall_2) = check_survival_conditions(
                val_results, self.config
            )
            
            if not survival and epoch > 10:
                print(f"\n⚠️  警告: 少数类召回率过低!")
                print(f"   Label 1 Recall: {recall_1:.2%}")
                print(f"   Label 2 Recall: {recall_2:.2%}")
            
            # 保存最佳模型
            current_f1 = val_results['macro_f1']
            if current_f1 > self.best_f1:
                self.best_f1 = current_f1
                save_checkpoint(
                    self.model, self.optimizer, epoch,
                    self.best_f1, self.config,
                    filename='best_model.pth'
                )
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # 早停
            if self.patience_counter >= self.config.train.patience:
                print(f"\n【Early Stopping】在第 {epoch+1} 轮触发")
                print(f"  Best Macro F1: {self.best_f1:.4f}")
                break
            
            # 学习率调整
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"  Learning Rate: {current_lr:.6f}")
        
        print(f"\n{'='*60}")
        print("【训练完成】")
        print(f"  Best Validation F1: {self.best_f1:.4f}")
        print(f"{'='*60}")
    
    def test(self):
        """测试最佳模型"""
        print(f"\n{'='*60}")
        print("【最终测试】")
        print(f"{'='*60}")
        
        # 加载最佳模型
        checkpoint_path = os.path.join(
            self.config.train.save_dir,
            'best_model.pth'
        )
        epoch, best_f1 = load_checkpoint(checkpoint_path, self.model)
        print(f"  加载模型: Epoch {epoch}, Best F1 {best_f1:.4f}")
        
        # 测试评估
        test_results = evaluate_model(
            self.model, self.test_loader,
            self.criterion, self.device
        )
        
        print_evaluation_report(test_results, phase='Test')
        
        # 最终生存判决
        print(f"\n{'='*60}")
        print("【生死判决】")
        print(f"{'='*60}")
        
        survival, (recall_1, recall_2) = check_survival_conditions(
            test_results, self.config
        )
        
        if survival:
            print("✓✓✓ 生存条件满足 - 模型可以部署")
            print(f"    Label 1 Recall: {recall_1:.2%}")
            print(f"    Label 2 Recall: {recall_2:.2%}")
        else:
            print("✗✗✗ 生存条件未满足 - 需要调整")
            print(f"    Label 1 Recall: {recall_1:.2%} (需要 ≥ {self.config.train.min_recall_minority:.0%})")
            print(f"    Label 2 Recall: {recall_2:.2%} (需要 ≥ {self.config.train.min_recall_minority:.0%})")
        
        print(f"{'='*60}")
        
        # 详细分类报告
        from sklearn.metrics import classification_report
        print("\n【详细分类报告】")
        print(classification_report(
            test_results['labels'],
            test_results['predictions'],
            target_names=['No Signal (0)', 'Pump (1)', 'Dump (2)'],
            digits=4
        ))
        
        return test_results


def main():
    """主函数"""
    # 加载配置
    config = get_config()
    
    # 创建训练器
    trainer = Trainer(config)
    
    # 训练
    trainer.train()
    
    # 测试
    test_results = trainer.test()

    # 训练后对全量数据推理并写出CSV
    try:
        print("\n【全量推理】开始运行 inference_full ...")
        run_inference()
        print("【全量推理】完成，结果已保存到 checkpoints/inference_full.csv 与 checkpoints/full_with_pred.csv")
    except Exception as e:
        print(f"【全量推理】失败: {e}")
        print("可手动运行: python3 inference_full.py")
    
    print("\n【任务完成】")
    print("氧气供应恢复。祝你在市场中生存。")


if __name__ == '__main__':
    main()
