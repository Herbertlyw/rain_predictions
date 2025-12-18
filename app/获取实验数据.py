#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
获取模型训练和预测的真实数据
运行此脚本获取答辩需要的所有实验数据
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import time

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def get_model_info():
    """获取模型信息"""
    print("="*60)
    print("1. 模型信息统计")
    print("="*60)
    
    try:
        # 加载最佳模型
        checkpoint_path = 'best_model.pth'
        if not os.path.exists(checkpoint_path):
            checkpoint_path = 'predrnn_v2_best.pth'
        if not os.path.exists(checkpoint_path):
            checkpoint_path = 'checkpoints/predrnn_v2_best.pth'
            
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 模型参数量
            total_params = sum(p.numel() for p in checkpoint.values() if isinstance(p, torch.Tensor))
            print(f"✓ 总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
            
            # 训练轮数
            if 'epoch' in checkpoint:
                print(f"✓ 训练轮数: {checkpoint['epoch'] + 1}")
            
            # 损失历史
            if 'train_losses' in checkpoint:
                losses = checkpoint['train_losses']
                print(f"✓ 初始损失: {losses[0]:.6f}")
                print(f"✓ 最终损失: {losses[-1]:.6f}")
                print(f"✓ 损失下降: {(losses[0]-losses[-1])/losses[0]*100:.1f}%")
                
                # 绘制损失曲线
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(losses)+1), losses, 'b-', linewidth=2, label='训练损失')
                plt.xlabel('训练轮数 (Epoch)', fontsize=12)
                plt.ylabel('损失值 (Loss)', fontsize=12)
                plt.title('模型训练损失曲线', fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3)
                plt.legend(fontsize=11)
                
                # 标注关键点
                min_loss_idx = np.argmin(losses)
                plt.scatter(min_loss_idx+1, losses[min_loss_idx], 
                           color='red', s=100, zorder=5, label='最佳点')
                plt.annotate(f'最小损失: {losses[min_loss_idx]:.6f}',
                           xy=(min_loss_idx+1, losses[min_loss_idx]),
                           xytext=(10, 20), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                
                plt.tight_layout()
                plt.savefig('答辩材料_训练曲线.png', dpi=300, bbox_inches='tight')
                print(f"✓ 训练曲线已保存: 答辩材料_训练曲线.png")
                
                return {
                    'total_params': total_params,
                    'epochs': len(losses),
                    'initial_loss': losses[0],
                    'final_loss': losses[-1],
                    'best_loss': min(losses),
                    'improvement': (losses[0]-losses[-1])/losses[0]*100
                }
        else:
            print("❌ 未找到模型文件")
            return None
            
    except Exception as e:
        print(f"❌ 获取模型信息失败: {str(e)}")
        return None

def analyze_training_checkpoints():
    """分析训练检查点"""
    print("\n" + "="*60)
    print("2. 训练检查点分析")
    print("="*60)
    
    checkpoint_dir = 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        print("❌ checkpoints目录不存在")
        return None
        
    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) 
                         if f.startswith('predrnn_v2_epoch_') and f.endswith('.pth')])
    
    if not checkpoints:
        print("❌ 未找到检查点文件")
        return None
    
    print(f"✓ 找到 {len(checkpoints)} 个检查点")
    
    # 分析每个检查点
    checkpoint_data = []
    for ckpt_file in checkpoints:
        try:
            epoch_num = int(ckpt_file.split('_')[-1].split('.')[0])
            ckpt = torch.load(os.path.join(checkpoint_dir, ckpt_file), 
                             map_location='cpu')
            
            if 'train_losses' in ckpt:
                losses = ckpt['train_losses']
                checkpoint_data.append({
                    'epoch': epoch_num,
                    'loss': losses[-1] if losses else None,
                    'file': ckpt_file
                })
        except Exception as e:
            print(f"  ⚠ 跳过 {ckpt_file}: {str(e)}")
    
    if checkpoint_data:
        checkpoint_data.sort(key=lambda x: x['epoch'])
        print(f"\n最后5个检查点:")
        for data in checkpoint_data[-5:]:
            print(f"  Epoch {data['epoch']:2d}: 损失 = {data['loss']:.6f}")
    
    return checkpoint_data

def test_prediction_performance():
    """测试预测性能"""
    print("\n" + "="*60)
    print("3. 预测性能测试")
    print("="*60)
    
    try:
        from app import model, config
        import torch
        
        if model is None:
            print("❌ 模型未加载")
            return None
        
        # 创建测试数据
        test_input = torch.randn(1, 5, 128, 128).to(config.device)
        
        # 预热
        with torch.no_grad():
            _ = model(test_input, return_loss=False)
        
        # 测试推理时间
        num_tests = 10
        times = []
        
        print(f"正在测试推理速度 ({num_tests}次)...")
        for i in range(num_tests):
            start = time.time()
            with torch.no_grad():
                predictions, _ = model(test_input, return_loss=False)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.time()
            times.append(end - start)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"✓ 平均推理时间: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
        print(f"✓ 输出形状: {predictions.shape}")
        
        return {
            'avg_inference_time': avg_time,
            'std_inference_time': std_time,
            'output_shape': predictions.shape
        }
        
    except Exception as e:
        print(f"❌ 性能测试失败: {str(e)}")
        print("  建议: 先运行 app.py 初始化模型")
        return None

def analyze_predictions():
    """分析预测结果"""
    print("\n" + "="*60)
    print("4. 预测结果分析")
    print("="*60)
    
    predictions_dir = 'predictions'
    if not os.path.exists(predictions_dir):
        print("❌ predictions目录不存在")
        return None
    
    # 获取所有预测会话
    sessions = [d for d in os.listdir(predictions_dir) 
                if os.path.isdir(os.path.join(predictions_dir, d))]
    
    print(f"✓ 找到 {len(sessions)} 个预测会话")
    
    if not sessions:
        return None
    
    # 分析一个典型会话
    sample_session = sessions[0]
    session_dir = os.path.join(predictions_dir, sample_session)
    
    # 统计文件
    npy_files = [f for f in os.listdir(session_dir) if f.endswith('.npy')]
    png_files = [f for f in os.listdir(session_dir) if f.endswith('.png')]
    
    print(f"\n示例会话 {sample_session}:")
    print(f"  - NPY文件: {len(npy_files)} 个")
    print(f"  - PNG图像: {len(png_files)} 个")
    
    # 加载并分析预测结果
    pred_files = sorted([f for f in npy_files if f.startswith('prediction_')])
    if pred_files:
        predictions = []
        for pf in pred_files[:5]:  # 只分析前5个
            data = np.load(os.path.join(session_dir, pf))
            predictions.append(data)
        
        predictions = np.array(predictions)
        print(f"  - 预测数据形状: {predictions.shape}")
        print(f"  - 值范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
        print(f"  - 平均值: {predictions.mean():.4f}")
    
    return {
        'num_sessions': len(sessions),
        'sample_session': sample_session,
        'num_predictions': len(pred_files)
    }

def calculate_metrics():
    """计算评价指标（需要有真实数据对比）"""
    print("\n" + "="*60)
    print("5. 评价指标计算")
    print("="*60)
    
    print("⚠ 注意: 需要真实数据才能计算MSE、SSIM等指标")
    print("   如果您有测试集，请使用以下代码计算：")
    print("""
# 示例代码：
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# 假设您有真实值和预测值
y_true = np.load('ground_truth.npy')  # 真实值
y_pred = np.load('predictions.npy')    # 预测值

# 计算指标
mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())

print(f"MSE: {mse:.6f}")
print(f"MAE: {mae:.6f}")
    """)
    
    return None

def generate_report():
    """生成完整报告"""
    print("\n" + "="*60)
    print("生成实验数据报告")
    print("="*60)
    
    report = {
        'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_info': get_model_info(),
        'checkpoints': analyze_training_checkpoints(),
        'performance': test_prediction_performance(),
        'predictions': analyze_predictions(),
    }
    
    # 保存为JSON
    with open('答辩材料_实验数据.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n✓ 报告已保存: 答辩材料_实验数据.json")
    
    # 生成Markdown报告
    md_report = generate_markdown_report(report)
    with open('答辩材料_实验数据报告.md', 'w', encoding='utf-8') as f:
        f.write(md_report)
    
    print(f"✓ Markdown报告已保存: 答辩材料_实验数据报告.md")
    
    return report

def generate_markdown_report(report):
    """生成Markdown格式的报告"""
    md = "# 实验数据报告\n\n"
    md += f"**生成时间**: {report['generated_at']}\n\n"
    md += "---\n\n"
    
    # 模型信息
    if report['model_info']:
        info = report['model_info']
        md += "## 1. 模型信息\n\n"
        md += f"- **总参数量**: {info['total_params']:,} ({info['total_params']/1e6:.2f}M)\n"
        md += f"- **训练轮数**: {info['epochs']}\n"
        md += f"- **初始损失**: {info['initial_loss']:.6f}\n"
        md += f"- **最终损失**: {info['final_loss']:.6f}\n"
        md += f"- **最佳损失**: {info['best_loss']:.6f}\n"
        md += f"- **损失下降**: {info['improvement']:.1f}%\n\n"
    
    # 预测性能
    if report['performance']:
        perf = report['performance']
        md += "## 2. 预测性能\n\n"
        md += f"- **平均推理时间**: {perf['avg_inference_time']*1000:.2f} ms\n"
        md += f"- **标准差**: {perf['std_inference_time']*1000:.2f} ms\n\n"
    
    # 预测结果
    if report['predictions']:
        pred = report['predictions']
        md += "## 3. 预测结果\n\n"
        md += f"- **预测会话数**: {pred['num_sessions']}\n"
        md += f"- **预测帧数**: {pred['num_predictions']}\n\n"
    
    md += "---\n\n"
    md += "## 使用说明\n\n"
    md += "请将以上真实数据填入您的答辩PPT和技术报告中。\n\n"
    
    return md

if __name__ == '__main__':
    print("\n" + "="*60)
    print("       获取答辩实验数据脚本")
    print("="*60 + "\n")
    
    report = generate_report()
    
    print("\n" + "="*60)
    print("完成！请查看以下文件：")
    print("  1. 答辩材料_训练曲线.png")
    print("  2. 答辩材料_实验数据.json")
    print("  3. 答辩材料_实验数据报告.md")
    print("="*60 + "\n")

