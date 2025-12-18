import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sys

# 强制刷新输出
sys.stdout.flush()

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def draw_hypergraph_basic():
    """绘制基本超图结构"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # 创建3×3节点
    positions = []
    for i in range(3):
        for j in range(3):
            x, y = j, 2 - i
            positions.append((x, y))
    
    # 绘制节点
    for i, (x, y) in enumerate(positions):
        if i == 4:  # 中心节点
            color = '#f72585'
            size = 300
        else:
            color = '#4361ee'
            size = 200
        ax.scatter(x, y, s=size, c=color, zorder=3, edgecolors='white', linewidths=2)
        ax.text(x, y-0.3, f'v{i+1}', ha='center', fontsize=10)
    
    # 绘制超边（椭圆形包围）
    ellipse = patches.Ellipse((1, 1), 2.5, 2.5, 
                              fill=False, edgecolor='#ff9800', 
                              linewidth=3, linestyle='--', alpha=0.7)
    ax.add_patch(ellipse)
    
    # 添加标注
    ax.text(2.5, 1, '超边 e₁', fontsize=14, color='#ff9800', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(1, -0.8, '连接9个节点', ha='center', fontsize=12)
    
    ax.set_xlim(-0.5, 3)
    ax.set_ylim(-1, 3)
    ax.axis('off')
    ax.set_aspect('equal')
    
    plt.title('雷达图像的超图表示', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('超图示意图_基本结构.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("✓ 已保存: 超图示意图_基本结构.png")

def draw_hypergraph_comparison():
    """绘制卷积核与超边的对比"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：卷积核
    kernel = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
    
    im = ax1.imshow(kernel, cmap='Blues', alpha=0.6)
    ax1.set_xticks(range(3))
    ax1.set_yticks(range(3))
    
    for i in range(3):
        for j in range(3):
            ax1.text(j, i, str(kernel[i, j]), 
                    ha='center', va='center', fontsize=20, fontweight='bold')
    
    ax1.set_title('3×3 卷积核', fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    # 右图：超边表示
    positions = [(1, 2), (2, 2), (3, 2),
                (1, 1), (2, 1), (3, 1),
                (1, 0), (2, 0), (3, 0)]
    
    for i, (x, y) in enumerate(positions):
        if i == 4:
            color = '#f72585'
            size = 400
        else:
            color = '#4361ee'
            size = 250
        ax2.scatter(x, y, s=size, c=color, zorder=3, 
                   edgecolors='white', linewidths=3)
    
    # 绘制超边连接
    center = positions[4]
    for pos in positions:
        if pos != center:
            ax2.plot([center[0], pos[0]], [center[1], pos[1]], 
                    'orange', linewidth=2, alpha=0.5, zorder=1)
    
    # 超边椭圆
    ellipse = patches.Ellipse((2, 1), 2.8, 2.8,
                             fill=False, edgecolor='#ff9800',
                             linewidth=3, linestyle='--', alpha=0.8)
    ax2.add_patch(ellipse)
    
    ax2.text(4, 1, '超边', fontsize=16, color='#ff9800',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.set_xlim(0, 5)
    ax2.set_ylim(-0.5, 2.5)
    ax2.set_title('超图表示', fontsize=16, fontweight='bold')
    ax2.axis('off')
    ax2.set_aspect('equal')
    
    # 添加转换箭头
    fig.text(0.5, 0.5, '重新解释为\n↓', ha='center', va='center',
            fontsize=14, fontweight='bold', color='#4361ee')
    
    plt.suptitle('卷积核 vs 超图超边', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('超图示意图_卷积对比.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("✓ 已保存: 超图示意图_卷积对比.png")

def draw_dual_hypergraph():
    """绘制双层超图架构"""
    fig = plt.figure(figsize=(12, 10))
    
    # 顶部：输入序列
    ax_input = plt.subplot(4, 1, 1)
    for i in range(5):
        rect = patches.Rectangle((i*1.5, 0), 1, 1, 
                                 facecolor='#4361ee', alpha=0.3, 
                                 edgecolor='#4361ee', linewidth=2)
        ax_input.add_patch(rect)
        ax_input.text(i*1.5+0.5, 0.5, f't{i+1}', ha='center', va='center',
                     fontsize=14, fontweight='bold')
    ax_input.set_xlim(-0.5, 7.5)
    ax_input.set_ylim(-0.5, 1.5)
    ax_input.text(-0.3, 0.5, '输入序列', fontsize=14, fontweight='bold', va='center')
    ax_input.axis('off')
    
    # 中上：空间超图
    ax_spatial = plt.subplot(4, 1, 2)
    spatial_box = patches.FancyBboxPatch((0.5, 0), 6, 1.5,
                                         boxstyle="round,pad=0.1",
                                         facecolor='#4cc9f0', alpha=0.2,
                                         edgecolor='#4cc9f0', linewidth=3)
    ax_spatial.add_patch(spatial_box)
    
    # 绘制空间超图节点
    for i in range(5):
        for j in range(3):
            ax_spatial.scatter(1+i*1.2, 0.3+j*0.4, s=100, 
                             c='#4cc9f0', zorder=3, edgecolors='white', linewidths=1)
    
    ax_spatial.text(3.5, 1.2, '空间超图 (c_t)', fontsize=14, 
                   fontweight='bold', ha='center', color='#4cc9f0')
    ax_spatial.text(7, 0.75, '捕获：\n空间分布', fontsize=11, 
                   ha='left', va='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax_spatial.set_xlim(-0.5, 8)
    ax_spatial.set_ylim(-0.3, 1.8)
    ax_spatial.axis('off')
    
    # 中下：时间超图
    ax_temporal = plt.subplot(4, 1, 3)
    temporal_box = patches.FancyBboxPatch((0.5, 0), 6, 1.5,
                                          boxstyle="round,pad=0.1",
                                          facecolor='#f72585', alpha=0.2,
                                          edgecolor='#f72585', linewidth=3)
    ax_temporal.add_patch(temporal_box)
    
    # 绘制时间超图节点（链式）
    for i in range(5):
        ax_temporal.scatter(1+i*1.2, 0.75, s=100,
                          c='#f72585', zorder=3, edgecolors='white', linewidths=1)
        if i < 4:
            ax_temporal.arrow(1.1+i*1.2, 0.75, 0.9, 0,
                            head_width=0.15, head_length=0.1,
                            fc='#f72585', ec='#f72585', alpha=0.6)
    
    ax_temporal.text(3.5, 1.2, '时间超图 (m_t)', fontsize=14,
                    fontweight='bold', ha='center', color='#f72585')
    ax_temporal.text(7, 0.75, '捕获：\n时间演化', fontsize=11,
                    ha='left', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax_temporal.set_xlim(-0.5, 8)
    ax_temporal.set_ylim(-0.3, 1.8)
    ax_temporal.axis('off')
    
    # 底部：输出序列
    ax_output = plt.subplot(4, 1, 4)
    for i in range(5):
        rect = patches.Rectangle((i*1.5, 0), 1, 1,
                                facecolor='#4361ee', alpha=0.6,
                                edgecolor='#4361ee', linewidth=2)
        ax_output.add_patch(rect)
        ax_output.text(i*1.5+0.5, 0.5, f't{i+6}', ha='center', va='center',
                      fontsize=14, fontweight='bold', color='white')
    ax_output.set_xlim(-0.5, 7.5)
    ax_output.set_ylim(-0.5, 1.5)
    ax_output.text(-0.3, 0.5, '输出序列', fontsize=14, fontweight='bold', va='center')
    ax_output.axis('off')
    
    # 添加连接箭头
    fig.text(0.1, 0.70, '↓', ha='center', fontsize=30, color='#4361ee')
    fig.text(0.1, 0.48, '↕', ha='center', fontsize=30, color='gray')
    fig.text(0.1, 0.25, '↓', ha='center', fontsize=30, color='#4361ee')
    
    plt.suptitle('双层超图神经网络架构', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0.12, 0.03, 1, 0.96])
    plt.savefig('超图示意图_双层架构.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("✓ 已保存: 超图示意图_双层架构.png")

def draw_multiscale_hypergraph():
    """绘制多尺度超图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    scales = [3, 5, 7]
    colors = ['#4361ee', '#4cc9f0', '#7209b7']
    titles = ['小尺度\n(3×3超边)', '中尺度\n(5×5超边)', '大尺度\n(7×7超边)']
    
    for idx, (ax, scale, color, title) in enumerate(zip(axes, scales, colors, titles)):
        # 生成网格节点
        center = scale // 2
        for i in range(scale):
            for j in range(scale):
                x, y = j - center, center - i
                if i == center and j == center:
                    ax.scatter(x, y, s=300, c='#f72585', zorder=3,
                             edgecolors='white', linewidths=2)
                else:
                    ax.scatter(x, y, s=150, c=color, alpha=0.6, zorder=2,
                             edgecolors='white', linewidths=1)
        
        # 绘制超边范围（圆形）
        circle = patches.Circle((0, 0), scale/2 + 0.3,
                               fill=False, edgecolor=color,
                               linewidth=3, linestyle='--', alpha=0.8)
        ax.add_patch(circle)
        
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold', color=color)
    
    plt.suptitle('多尺度超图结构', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('超图示意图_多尺度.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("✓ 已保存: 超图示意图_多尺度.png")

if __name__ == '__main__':
    print("开始生成超图示意图...")
    print("-" * 50)
    
    draw_hypergraph_basic()
    draw_hypergraph_comparison()
    draw_dual_hypergraph()
    draw_multiscale_hypergraph()
    
    print("-" * 50)
    print("✅ 所有示意图生成完成！")
    print("\n生成的文件：")
    print("  1. 超图示意图_基本结构.png")
    print("  2. 超图示意图_卷积对比.png")
    print("  3. 超图示意图_双层架构.png")
    print("  4. 超图示意图_多尺度.png")