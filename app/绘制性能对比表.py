import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

def draw_performance_table():
    """绘制模型性能对比表格"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 表格数据
    headers = ['模型配置', '参数量', 'MSE', 'SSIM', '训练时间/epoch']
    
    data = [
        ['Baseline ConvLSTM', '2.88M', '0.0285', '0.782', '12.3分钟'],
        ['+空间超图(c_t)', '4.44M\n(+54.6%)', '0.0233\n(↓18.2%)', '0.814\n(↑4.1%)', '14.2分钟\n(+15%)'],
        ['+时间超图(m_t)', '4.44M\n(+54.6%)', '0.0248\n(↓13.1%)', '0.802\n(↑2.5%)', '13.8分钟\n(+12%)'],
        ['双层超图(完整)', '5.89M\n(+104.8%)', '0.0194\n(↓32.0%)', '0.843\n(↑7.8%)', '15.7分钟\n(+28%)']
    ]
    
    # 颜色配置
    header_color = '#4361ee'
    baseline_color = '#e8f4f8'
    row_colors = ['#f0f9ff', '#e0f2fe', '#dbeafe']
    highlight_color = '#fef3c7'  # 高亮最佳结果
    
    # 列宽
    col_widths = [0.25, 0.18, 0.18, 0.18, 0.21]
    
    # 绘制表格标题
    title_y = 0.95
    ax.text(0.5, title_y, '表X 不同模型配置的性能对比', 
            ha='center', va='top', fontsize=20, fontweight='bold',
            transform=ax.transAxes)
    
    # 表格起始位置
    table_top = 0.85
    row_height = 0.14
    
    # 绘制表头
    x_start = 0.05
    for i, (header, width) in enumerate(zip(headers, col_widths)):
        x = x_start + sum(col_widths[:i])
        # 表头背景
        rect = patches.FancyBboxPatch(
            (x, table_top - row_height), width, row_height,
            boxstyle="round,pad=0.005",
            facecolor=header_color, edgecolor='white', linewidth=2,
            transform=ax.transAxes
        )
        ax.add_patch(rect)
        # 表头文字
        ax.text(x + width/2, table_top - row_height/2, header,
                ha='center', va='center', fontsize=13, fontweight='bold',
                color='white', transform=ax.transAxes)
    
    # 绘制数据行
    for row_idx, row_data in enumerate(data):
        y = table_top - (row_idx + 2) * row_height
        
        # 选择行颜色
        if row_idx == 0:
            bg_color = baseline_color
        elif row_idx == 3:  # 双层超图，最佳结果
            bg_color = highlight_color
        else:
            bg_color = row_colors[row_idx - 1]
        
        for col_idx, (cell, width) in enumerate(zip(row_data, col_widths)):
            x = x_start + sum(col_widths[:col_idx])
            
            # 单元格背景
            rect = patches.Rectangle(
                (x, y), width, row_height,
                facecolor=bg_color, edgecolor='#d1d5db', linewidth=1,
                transform=ax.transAxes
            )
            ax.add_patch(rect)
            
            # 单元格文字
            fontsize = 11 if '\n' in cell else 12
            fontweight = 'bold' if row_idx == 3 else 'normal'
            
            # 根据列设置对齐方式
            if col_idx == 0:
                ha = 'left'
                x_text = x + 0.02
            else:
                ha = 'center'
                x_text = x + width/2
            
            ax.text(x_text, y + row_height/2, cell,
                    ha=ha, va='center', fontsize=fontsize, 
                    fontweight=fontweight, transform=ax.transAxes)
    
    # 添加注释
    note_y = table_top - (len(data) + 2) * row_height - 0.05
    ax.text(0.05, note_y, 
            '注：括号内为相比Baseline的变化率；MSE越低越好，SSIM越高越好；黄色高亮为最佳模型',
            ha='left', va='top', fontsize=10, style='italic',
            color='#6b7280', transform=ax.transAxes)
    
    # 添加性能指标图标
    icons_y = note_y - 0.08
    # MSE图标
    ax.text(0.05, icons_y, '▼ MSE最优：', ha='left', va='center',
            fontsize=10, fontweight='bold', color='#10b981',
            transform=ax.transAxes)
    ax.text(0.15, icons_y, '0.0194 (↓32.0%)', ha='left', va='center',
            fontsize=10, color='#10b981', transform=ax.transAxes)
    
    # SSIM图标
    ax.text(0.35, icons_y, '▲ SSIM最优：', ha='left', va='center',
            fontsize=10, fontweight='bold', color='#3b82f6',
            transform=ax.transAxes)
    ax.text(0.45, icons_y, '0.843 (↑7.8%)', ha='left', va='center',
            fontsize=10, color='#3b82f6', transform=ax.transAxes)
    
    # 参数效率
    ax.text(0.65, icons_y, '⚡ 参数效率：', ha='left', va='center',
            fontsize=10, fontweight='bold', color='#f59e0b',
            transform=ax.transAxes)
    ax.text(0.75, icons_y, '10.6%/M', ha='left', va='center',
            fontsize=10, color='#f59e0b', transform=ax.transAxes)
    
    # 关闭坐标轴
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('模型性能对比表.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.3)
    print("✓ 已保存: 模型性能对比表.png")

def draw_performance_chart():
    """绘制性能对比柱状图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    models = ['Baseline\nConvLSTM', '空间超图\n(c_t)', '时间超图\n(m_t)', '双层超图\n(完整)']
    colors = ['#94a3b8', '#4cc9f0', '#f72585', '#4361ee']
    
    # 1. 参数量对比
    params = [2.88, 4.44, 4.44, 5.89]
    bars1 = ax1.bar(models, params, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    ax1.set_ylabel('参数量 (M)', fontsize=12, fontweight='bold')
    ax1.set_title('模型参数量对比', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 7)
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars1, params)):
        if i > 0:
            increase = (val - params[0]) / params[0] * 100
            label = f'{val}M\n(+{increase:.1f}%)'
        else:
            label = f'{val}M'
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. MSE对比
    mse = [0.0285, 0.0233, 0.0248, 0.0194]
    bars2 = ax2.bar(models, mse, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    ax2.set_ylabel('MSE (越低越好)', fontsize=12, fontweight='bold')
    ax2.set_title('预测误差对比', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 0.035)
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars2, mse)):
        if i > 0:
            decrease = (params[0] - val) / params[0] * 100
            label = f'{val:.4f}\n(↓{decrease:.1f}%)'
            color = '#10b981'
        else:
            label = f'{val:.4f}'
            color = 'black'
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                label, ha='center', va='bottom', fontsize=10, fontweight='bold', color=color)
    
    # 3. SSIM对比
    ssim = [0.782, 0.814, 0.802, 0.843]
    bars3 = ax3.bar(models, ssim, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    ax3.set_ylabel('SSIM (越高越好)', fontsize=12, fontweight='bold')
    ax3.set_title('结构相似度对比', fontsize=14, fontweight='bold', pad=15)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.set_ylim(0.7, 0.9)
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars3, ssim)):
        if i > 0:
            increase = (val - ssim[0]) / ssim[0] * 100
            label = f'{val:.3f}\n(↑{increase:.1f}%)'
            color = '#3b82f6'
        else:
            label = f'{val:.3f}'
            color = 'black'
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                label, ha='center', va='bottom', fontsize=10, fontweight='bold', color=color)
    
    # 4. 训练时间对比
    time = [12.3, 14.2, 13.8, 15.7]
    bars4 = ax4.bar(models, time, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    ax4.set_ylabel('训练时间 (分钟/epoch)', fontsize=12, fontweight='bold')
    ax4.set_title('训练时间对比', fontsize=14, fontweight='bold', pad=15)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    ax4.set_ylim(0, 20)
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars4, time)):
        if i > 0:
            increase = (val - time[0]) / time[0] * 100
            label = f'{val}分钟\n(+{increase:.1f}%)'
        else:
            label = f'{val}分钟'
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('模型性能全面对比', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig('模型性能对比图.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("✓ 已保存: 模型性能对比图.png")

if __name__ == '__main__':
    print("开始生成性能对比图表...")
    print("-" * 50)
    
    draw_performance_table()
    draw_performance_chart()
    
    print("-" * 50)
    print("✅ 性能对比图表生成完成！")
    print("\n生成的文件：")
    print("  1. 模型性能对比表.png - 详细数据表格")
    print("  2. 模型性能对比图.png - 可视化柱状图")



