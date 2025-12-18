"""
计算不同模型配置的参数量
数据：500×500图像，经过patch_size=4处理后变成125×125
"""
import torch
import torch.nn as nn

# ==================== 基础LSTM Cell (Baseline) ====================
class BaselineLSTMCell(nn.Module):
    """标准LSTM作为baseline"""
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride):
        super(BaselineLSTMCell, self).__init__()
        self.num_hidden = num_hidden
        padding = filter_size // 2
        
        # 标准ConvLSTM: i, f, g, o 四个门
        self.conv_x = nn.Conv2d(in_channel, num_hidden * 4, 
                                kernel_size=filter_size, stride=stride, 
                                padding=padding, bias=True)
        self.conv_h = nn.Conv2d(num_hidden, num_hidden * 4, 
                                kernel_size=filter_size, stride=stride, 
                                padding=padding, bias=True)

class SpatialHypergraphCell(nn.Module):
    """添加空间超图 c_t"""
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride):
        super(SpatialHypergraphCell, self).__init__()
        self.num_hidden = num_hidden
        padding = filter_size // 2
        
        # x的卷积: i, f, g, o + c_t的输入门
        self.conv_x = nn.Conv2d(in_channel, num_hidden * 5, 
                                kernel_size=filter_size, stride=stride, 
                                padding=padding, bias=False)
        # h的卷积: i, f, g, o
        self.conv_h = nn.Conv2d(num_hidden, num_hidden * 4, 
                                kernel_size=filter_size, stride=stride, 
                                padding=padding, bias=False)
        # c_t的卷积: i, f, o
        self.conv_c = nn.Conv2d(num_hidden, num_hidden * 3, 
                                kernel_size=filter_size, stride=stride, 
                                padding=padding, bias=False)
        # 输出卷积
        self.conv_o = nn.Conv2d(num_hidden * 2, num_hidden, 
                                kernel_size=1, stride=1, 
                                padding=0, bias=False)
        # BatchNorm
        self.bn = nn.BatchNorm2d(num_hidden)

class TemporalHypergraphCell(nn.Module):
    """添加时间超图 m_t"""
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride):
        super(TemporalHypergraphCell, self).__init__()
        self.num_hidden = num_hidden
        padding = filter_size // 2
        
        # x的卷积: i, f, g, o + m_t的输入门
        self.conv_x = nn.Conv2d(in_channel, num_hidden * 5, 
                                kernel_size=filter_size, stride=stride, 
                                padding=padding, bias=False)
        # h的卷积: i, f, g, o
        self.conv_h = nn.Conv2d(num_hidden, num_hidden * 4, 
                                kernel_size=filter_size, stride=stride, 
                                padding=padding, bias=False)
        # m_t的卷积: i, f, g
        self.conv_m = nn.Conv2d(num_hidden, num_hidden * 3, 
                                kernel_size=filter_size, stride=stride, 
                                padding=padding, bias=False)
        # 输出卷积
        self.conv_o = nn.Conv2d(num_hidden * 2, num_hidden, 
                                kernel_size=1, stride=1, 
                                padding=0, bias=False)
        # BatchNorm
        self.bn = nn.BatchNorm2d(num_hidden)

class DualHypergraphCell(nn.Module):
    """双层超图：同时包含 c_t 和 m_t"""
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super(DualHypergraphCell, self).__init__()
        self.num_hidden = num_hidden
        padding = filter_size // 2
        
        # x的卷积: 7个输出 (i, f, g, o + i', i_m, g_m)
        self.conv_x = nn.Conv2d(in_channel, num_hidden * 7, 
                                kernel_size=filter_size, stride=stride, 
                                padding=padding, bias=False)
        # h的卷积: 4个输出 (i, f, g, o)
        self.conv_h = nn.Conv2d(num_hidden, num_hidden * 4, 
                                kernel_size=filter_size, stride=stride, 
                                padding=padding, bias=False)
        # m的卷积: 3个输出 (i, f, g)
        self.conv_m = nn.Conv2d(num_hidden, num_hidden * 3, 
                                kernel_size=filter_size, stride=stride, 
                                padding=padding, bias=False)
        # o的卷积
        self.conv_o = nn.Conv2d(num_hidden * 2, num_hidden, 
                                kernel_size=filter_size, stride=stride, 
                                padding=padding, bias=False)
        # 最后的1x1卷积
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, 
                                    kernel_size=1, stride=1, 
                                    padding=0, bias=False)
        # BatchNorm (多个)
        if layer_norm:
            self.norm_x = nn.LayerNorm([num_hidden * 7, height, width])
            self.norm_h = nn.LayerNorm([num_hidden * 4, height, width])
            self.norm_m = nn.LayerNorm([num_hidden * 3, height, width])
            self.norm_o = nn.LayerNorm([num_hidden, height, width])
            self.norm_last = nn.LayerNorm([num_hidden, height, width])
        else:
            self.norm_x = nn.BatchNorm2d(num_hidden * 7)
            self.norm_h = nn.BatchNorm2d(num_hidden * 4)
            self.norm_m = nn.BatchNorm2d(num_hidden * 3)
            self.norm_o = nn.BatchNorm2d(num_hidden)
            self.norm_last = nn.BatchNorm2d(num_hidden)

def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_full_model(cell_class, num_layers=4):
    """创建完整的多层模型"""
    # 配置参数（基于您的实际配置）
    in_channel = 1  # 单通道图像
    num_hidden = 64  # 每层64个隐藏单元
    height = 125  # 500/4 = 125 (经过patch处理)
    width = 125
    filter_size = 5
    stride = 1
    layer_norm = 0  # 使用BatchNorm
    
    cells = nn.ModuleList()
    for i in range(num_layers):
        if i == 0:
            in_ch = in_channel
        else:
            in_ch = num_hidden
        
        if cell_class == DualHypergraphCell:
            cell = cell_class(in_ch, num_hidden, height, width, 
                            filter_size, stride, layer_norm)
        else:
            cell = cell_class(in_ch, num_hidden, height, width, 
                            filter_size, stride)
        cells.append(cell)
    
    return cells

# ==================== 计算参数量 ====================
print("="*60)
print("模型参数量对比 (500×500输入, patch_size=4 → 125×125)")
print("配置: 4层, 每层64隐藏单元, 卷积核5×5")
print("="*60)

# 1. Baseline LSTM
baseline_cells = create_full_model(BaselineLSTMCell, num_layers=4)
baseline_params = sum(count_parameters(cell) for cell in baseline_cells)
print(f"\n1️⃣ Baseline LSTM (标准ConvLSTM)")
print(f"   参数量: {baseline_params:,} ({baseline_params/1e6:.2f}M)")

# 2. 添加空间超图 c_t
spatial_cells = create_full_model(SpatialHypergraphCell, num_layers=4)
spatial_params = sum(count_parameters(cell) for cell in spatial_cells)
spatial_increase = (spatial_params - baseline_params) / baseline_params * 100
print(f"\n2️⃣ +空间超图 (c_t)")
print(f"   参数量: {spatial_params:,} ({spatial_params/1e6:.2f}M)")
print(f"   相比baseline增加: +{spatial_increase:.1f}%")

# 3. 添加时间超图 m_t
temporal_cells = create_full_model(TemporalHypergraphCell, num_layers=4)
temporal_params = sum(count_parameters(cell) for cell in temporal_cells)
temporal_increase = (temporal_params - baseline_params) / baseline_params * 100
print(f"\n3️⃣ +时间超图 (m_t)")
print(f"   参数量: {temporal_params:,} ({temporal_params/1e6:.2f}M)")
print(f"   相比baseline增加: +{temporal_increase:.1f}%")

# 4. 双层超图 (完整PredRNNv2)
dual_cells = create_full_model(DualHypergraphCell, num_layers=4)
dual_params = sum(count_parameters(cell) for cell in dual_cells)
dual_increase = (dual_params - baseline_params) / baseline_params * 100
print(f"\n4️⃣ 双层超图 (完整: c_t + m_t)")
print(f"   参数量: {dual_params:,} ({dual_params/1e6:.2f}M)")
print(f"   相比baseline增加: +{dual_increase:.1f}%")

print("\n" + "="*60)
print("📊 总结表格")
print("="*60)
print(f"{'模型配置':<20} {'参数量':>12} {'相比baseline':>12}")
print("-"*60)
print(f"{'Baseline LSTM':<20} {f'{baseline_params/1e6:.2f}M':>12} {'-':>12}")
print(f"{'+空间超图(c_t)':<20} {f'{spatial_params/1e6:.2f}M':>12} {f'+{spatial_increase:.1f}%':>12}")
print(f"{'+时间超图(m_t)':<20} {f'{temporal_params/1e6:.2f}M':>12} {f'+{temporal_increase:.1f}%':>12}")
print(f"{'双层超图(完整)':<20} {f'{dual_params/1e6:.2f}M':>12} {f'+{dual_increase:.1f}%':>12}")
print("="*60)

# ==================== 详细分解（以第一层为例）====================
print("\n" + "="*60)
print("🔍 各配置第一层参数量详细分解")
print("="*60)

print("\n【Baseline LSTM】")
baseline_cell = BaselineLSTMCell(1, 64, 125, 125, 5, 1)
print(f"  conv_x: {count_parameters(baseline_cell.conv_x):,} 参数")
print(f"  conv_h: {count_parameters(baseline_cell.conv_h):,} 参数")
print(f"  总计: {count_parameters(baseline_cell):,} 参数")

print("\n【空间超图 c_t】")
spatial_cell = SpatialHypergraphCell(1, 64, 125, 125, 5, 1)
print(f"  conv_x (5 gates): {count_parameters(spatial_cell.conv_x):,} 参数")
print(f"  conv_h (4 gates): {count_parameters(spatial_cell.conv_h):,} 参数")
print(f"  conv_c (3 gates): {count_parameters(spatial_cell.conv_c):,} 参数")
print(f"  conv_o: {count_parameters(spatial_cell.conv_o):,} 参数")
print(f"  bn: {count_parameters(spatial_cell.bn):,} 参数")
print(f"  总计: {count_parameters(spatial_cell):,} 参数")

print("\n【时间超图 m_t】")
temporal_cell = TemporalHypergraphCell(1, 64, 125, 125, 5, 1)
print(f"  conv_x (5 gates): {count_parameters(temporal_cell.conv_x):,} 参数")
print(f"  conv_h (4 gates): {count_parameters(temporal_cell.conv_h):,} 参数")
print(f"  conv_m (3 gates): {count_parameters(temporal_cell.conv_m):,} 参数")
print(f"  conv_o: {count_parameters(temporal_cell.conv_o):,} 参数")
print(f"  bn: {count_parameters(temporal_cell.bn):,} 参数")
print(f"  总计: {count_parameters(temporal_cell):,} 参数")

print("\n【双层超图】")
dual_cell = DualHypergraphCell(1, 64, 125, 125, 5, 1, 0)
print(f"  conv_x (7 gates): {count_parameters(dual_cell.conv_x):,} 参数")
print(f"  conv_h (4 gates): {count_parameters(dual_cell.conv_h):,} 参数")
print(f"  conv_m (3 gates): {count_parameters(dual_cell.conv_m):,} 参数")
print(f"  conv_o: {count_parameters(dual_cell.conv_o):,} 参数")
print(f"  conv_last: {count_parameters(dual_cell.conv_last):,} 参数")
print(f"  所有BatchNorm: {count_parameters(dual_cell.norm_x) + count_parameters(dual_cell.norm_h) + count_parameters(dual_cell.norm_m) + count_parameters(dual_cell.norm_o) + count_parameters(dual_cell.norm_last):,} 参数")
print(f"  总计: {count_parameters(dual_cell):,} 参数")

print("\n✅ 计算完成！")



