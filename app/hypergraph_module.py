"""
超图神经网络模块 - 可集成到现有PredRNN模型中
用于大创结题，将ConvLSTM模型与超图神经网络概念结合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HypergraphConvolution(nn.Module):
    """
    超图卷积层
    将图像的空间邻域关系建模为超图结构
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(HypergraphConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # 节点特征变换
        self.node_transform = nn.Linear(in_channels, out_channels)
        
        # 超边特征变换
        self.hyperedge_transform = nn.Linear(in_channels * kernel_size * kernel_size, out_channels)
        
        # 注意力权重
        self.attention = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        输入: [batch, channels, height, width]
        输出: [batch, out_channels, height, width]
        
        将卷积核覆盖区域视为超边，执行超图卷积
        """
        batch, channels, height, width = x.shape
        
        # 1. 将图像展开为节点
        # [batch, channels, height, width] -> [batch, height*width, channels]
        nodes = x.view(batch, channels, -1).permute(0, 2, 1)
        
        # 2. 构建超边（使用unfold提取局部邻域）
        # 每个kernel_size×kernel_size的区域视为一个超边
        padding = self.kernel_size // 2
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, padding=padding)
        # [batch, channels*kernel_size*kernel_size, height*width]
        
        hyperedges = x_unfold.permute(0, 2, 1)
        # [batch, height*width, channels*kernel_size*kernel_size]
        
        # 3. 节点特征变换
        node_features = self.node_transform(nodes)
        # [batch, height*width, out_channels]
        
        # 4. 超边特征变换
        hyperedge_features = self.hyperedge_transform(hyperedges)
        # [batch, height*width, out_channels]
        
        # 5. 超图卷积：聚合超边信息到节点
        # 使用注意力机制
        combined = torch.cat([node_features, hyperedge_features], dim=-1)
        attention_weights = self.attention(combined)
        
        # 加权聚合
        output = node_features + attention_weights * hyperedge_features
        
        # 6. 重塑回图像形状
        output = output.permute(0, 2, 1).view(batch, self.out_channels, height, width)
        
        return output


class DynamicHypergraph(nn.Module):
    """
    动态超图构建模块
    根据输入动态构建超图结构（超边连接）
    """
    def __init__(self, channels, num_hyperedges=64):
        super(DynamicHypergraph, self).__init__()
        self.channels = channels
        self.num_hyperedges = num_hyperedges
        
        # 学习超边生成器
        self.hyperedge_generator = nn.Sequential(
            nn.Conv2d(channels, num_hyperedges, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 超边特征提取
        self.hyperedge_feature = nn.Conv2d(channels, channels, kernel_size=1)
        
    def forward(self, x):
        """
        x: [batch, channels, height, width]
        """
        batch, channels, height, width = x.shape
        num_nodes = height * width
        
        # 1. 生成关联矩阵 H: [batch, num_nodes, num_hyperedges]
        # 表示节点-超边的连接关系
        hyperedge_weights = self.hyperedge_generator(x)
        # [batch, num_hyperedges, height, width]
        
        H = hyperedge_weights.view(batch, self.num_hyperedges, -1).permute(0, 2, 1)
        # [batch, height*width, num_hyperedges]
        
        # 2. 节点特征矩阵 X: [batch, num_nodes, channels]
        X = x.view(batch, channels, -1).permute(0, 2, 1)
        
        # 3. 超图卷积：X' = D_v^(-1/2) H W D_e^(-1) H^T D_v^(-1/2) X
        # 简化版本：X' = H @ H^T @ X
        
        # 节点度矩阵
        D_v = torch.sum(H, dim=2, keepdim=True) + 1e-6  # [batch, num_nodes, 1]
        
        # 超边度矩阵
        D_e = torch.sum(H, dim=1, keepdim=True) + 1e-6  # [batch, 1, num_hyperedges]
        
        # 归一化的超图卷积
        H_normalized = H / torch.sqrt(D_v)  # [batch, num_nodes, num_hyperedges]
        H_T_normalized = H.permute(0, 2, 1) / D_e  # [batch, num_hyperedges, num_nodes]
        
        # 超图传播
        X_propagated = torch.bmm(
            torch.bmm(H_normalized, H_T_normalized), 
            X
        )  # [batch, num_nodes, channels]
        
        X_propagated = X_propagated / torch.sqrt(D_v)
        
        # 4. 重塑回图像
        output = X_propagated.permute(0, 2, 1).view(batch, channels, height, width)
        
        return output


class HypergraphEnhancedConvLSTM(nn.Module):
    """
    集成超图模块的ConvLSTM
    在原有ConvLSTM基础上添加超图卷积层
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, use_hypergraph=True):
        super(HypergraphEnhancedConvLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.use_hypergraph = use_hypergraph
        
        # 原始ConvLSTM组件
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=True
        )
        
        # 超图增强模块
        if use_hypergraph:
            self.hypergraph_conv = HypergraphConvolution(hidden_dim, hidden_dim)
            self.dynamic_hypergraph = DynamicHypergraph(hidden_dim)
            
            # 融合权重
            self.fusion = nn.Sequential(
                nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        # 标准ConvLSTM操作
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        # 超图增强
        if self.use_hypergraph:
            # 应用超图卷积
            h_hypergraph_1 = self.hypergraph_conv(h_next)
            h_hypergraph_2 = self.dynamic_hypergraph(h_next)
            
            # 融合标准LSTM输出和超图输出
            h_combined = torch.cat([h_hypergraph_1, h_hypergraph_2], dim=1)
            h_next = self.fusion(h_combined)
        
        return h_next, c_next


class SpatialHypergraphAttention(nn.Module):
    """
    空间超图注意力模块
    用于捕获长距离空间依赖关系
    """
    def __init__(self, channels, reduction=8):
        super(SpatialHypergraphAttention, self).__init__()
        self.channels = channels
        
        # 查询、键、值变换
        self.query_conv = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        
        # 超边权重
        self.hyperedge_weight = nn.Parameter(torch.randn(1))
        
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        """
        x: [batch, channels, height, width]
        """
        batch, channels, height, width = x.shape
        
        # 生成Q, K, V
        query = self.query_conv(x).view(batch, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch, -1, height * width)
        value = self.value_conv(x).view(batch, -1, height * width)
        
        # 计算注意力（模拟超图结构）
        attention = torch.bmm(query, key)  # [batch, height*width, height*width]
        attention = self.softmax(attention) * self.hyperedge_weight
        
        # 应用注意力
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch, channels, height, width)
        
        # 残差连接
        out = self.gamma * out + x
        
        return out


# ==================== 使用示例 ====================
def test_hypergraph_modules():
    """测试超图模块"""
    batch_size = 2
    channels = 32
    height, width = 64, 64
    
    x = torch.randn(batch_size, channels, height, width)
    
    # 测试超图卷积
    print("测试超图卷积...")
    hgconv = HypergraphConvolution(channels, channels)
    out1 = hgconv(x)
    print(f"输入形状: {x.shape}, 输出形状: {out1.shape}")
    
    # 测试动态超图
    print("\n测试动态超图...")
    dynamic_hg = DynamicHypergraph(channels)
    out2 = dynamic_hg(x)
    print(f"输入形状: {x.shape}, 输出形状: {out2.shape}")
    
    # 测试超图增强ConvLSTM
    print("\n测试超图增强ConvLSTM...")
    hg_lstm = HypergraphEnhancedConvLSTM(1, channels, kernel_size=3)
    input_tensor = torch.randn(batch_size, 1, height, width)
    h_init = torch.randn(batch_size, channels, height, width)
    c_init = torch.randn(batch_size, channels, height, width)
    
    h_next, c_next = hg_lstm(input_tensor, (h_init, c_init))
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出h形状: {h_next.shape}, 输出c形状: {c_next.shape}")
    
    # 测试空间超图注意力
    print("\n测试空间超图注意力...")
    attention = SpatialHypergraphAttention(channels)
    out3 = attention(x)
    print(f"输入形状: {x.shape}, 输出形状: {out3.shape}")
    
    print("\n所有模块测试通过！")


if __name__ == '__main__':
    test_hypergraph_modules()

