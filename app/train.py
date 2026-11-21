# pred_rnn_v2_optimized.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
from torch.utils.data import Dataset, DataLoader
import math
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.ndimage as ndimage
import random
import time
from torch.amp import GradScaler, autocast
import warnings
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset
# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
warnings.filterwarnings("ignore", category=FutureWarning)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== 数值稳定性增强 ====================
def apply_stability_measures(tensor, name=""):
    """应用全面的数值稳定性措施"""
    if tensor is None:
        return tensor
        
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        logger.warning(f"NaN/Inf detected in {name}, applying stability measures")
        
        # 1. 尝试用均值替换
        tensor_mean = torch.nanmean(tensor)
        if torch.isnan(tensor_mean) or torch.isinf(tensor_mean):
            tensor_mean = torch.tensor(0.0, device=tensor.device)
        
        tensor = torch.where(torch.isfinite(tensor), tensor, tensor_mean)
        
        # 2. 如果仍有问题，使用0替换
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)
    
    # 4. 限制数值范围
    tensor = torch.clamp(tensor, -50.0, 50.0)
    
    return tensor

# ==================== SpatioTemporalLSTMCellv2 (优化版) ====================
class SpatioTemporalLSTMCellv2(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super(SpatioTemporalLSTMCellv2, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        
        # 使用深度可分离卷积减少参数数量
        def create_conv_block(in_c, out_c):
            layers = []
            layers.append(
                nn.Conv2d(in_c, out_c, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False)
            )
            if layer_norm:
                layers.append(nn.LayerNorm([out_c, height, width]))
            else:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)
        
        # 使用更高效的卷积块
        self.conv_x = create_conv_block(in_channel, num_hidden * 7)
        self.conv_h = create_conv_block(num_hidden, num_hidden * 4)
        self.conv_m = create_conv_block(num_hidden, num_hidden * 3)
        self.conv_o = create_conv_block(num_hidden * 2, num_hidden)
        
        self.conv_last = nn.Sequential(
            nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(inplace=True)
        )
        
        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_t, h_t, c_t, m_t):
        # 应用数值稳定性措施
        x_t = apply_stability_measures(x_t, "x_t")
        h_t = apply_stability_measures(h_t, "h_t")
        c_t = apply_stability_measures(c_t, "c_t")
        m_t = apply_stability_measures(m_t, "m_t")
        
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        
        # 分割张量
        splits = [self.num_hidden] * 7
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, splits, dim=1)
        
        splits = [self.num_hidden] * 4
        i_h, f_h, g_h, o_h = torch.split(h_concat, splits, dim=1)
        
        splits = [self.num_hidden] * 3
        i_m, f_m, g_m = torch.split(m_concat, splits, dim=1)

        # 门控机制
        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        # 数值稳定性处理
        delta_c = i_t * g_t
        delta_c = apply_stability_measures(delta_c, "delta_c")
        c_new = f_t * c_t + delta_c
        c_new = apply_stability_measures(c_new, "c_new")

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        delta_m = i_t_prime * g_t_prime
        delta_m = apply_stability_measures(delta_m, "delta_m")
        m_new = f_t_prime * m_t + delta_m
        m_new = apply_stability_measures(m_new, "m_new")

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))
        
        # 最终稳定性检查
        h_new = apply_stability_measures(h_new, "h_new")
        
        return h_new, c_new, m_new, delta_c, delta_m

# ==================== PredRNNv2_Model (优化版) ====================
class PredRNNv2_Model(nn.Module):
    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(PredRNNv2_Model, self).__init__()
        T, C, H, W = configs.in_shape

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * C
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        # 关键修复：直接使用输入尺寸，不再除以patch_size
        height = H
        width = W
        
        # 使用混合精度训练
        self.use_amp = configs.use_amp
        
        # 损失函数
        self.MSE_criterion = nn.MSELoss()
        self.SSIM_criterion = self._ssim_loss if configs.use_ssim_loss else None

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCellv2(in_channel, num_hidden[i], height, width,
                                         configs.filter_size, configs.stride, configs.layer_norm))
        self.cell_list = nn.ModuleList(cell_list)
        
        # 输出层
        self.conv_last = nn.Sequential(
            nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.frame_channel),
            nn.Tanh()
        )
        
        # 共享适配器
        adapter_num_hidden = num_hidden[0]
        self.adapter = nn.Sequential(
            nn.Conv2d(adapter_num_hidden, adapter_num_hidden, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(adapter_num_hidden),
            nn.ReLU(inplace=True)
        )
        
        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def _ssim_loss(self, x, y):
        """计算SSIM损失"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu_x = F.avg_pool2d(x, 3, 1, padding=1)
        mu_y = F.avg_pool2d(y, 3, 1, padding=1)
        
        sigma_x = F.avg_pool2d(x**2, 3, 1, padding=1) - mu_x**2
        sigma_y = F.avg_pool2d(y**2, 3, 1, padding=1) - mu_y**2
        sigma_xy = F.avg_pool2d(x*y, 3, 1, padding=1) - mu_x*mu_y
        
        ssim_n = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2)
        ssim_d = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
        
        ssim = ssim_n / ssim_d
        return torch.clamp((1 - ssim) / 2, 0, 1).mean()

    def forward(self, frames_tensor, mask_true, **kwargs):
        return_loss = kwargs.get('return_loss', True)
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        device = frames_tensor.device
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        if return_loss:
            mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []
        delta_c_list = []
        delta_m_list = []

        decouple_loss = []

        for i in range(self.num_layers):
            zeros = torch.zeros(
                [batch, self.num_hidden[i], height, width], device=device)
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)

        memory = torch.zeros(
            [batch, self.num_hidden[0], height, width], device=device)

        # 使用自动混合精度
        with autocast("cuda", enabled=self.use_amp):
            for t in range(self.configs.total_length - 1):
                if not return_loss:
                    net = frames[:, t] if t < frames.shape[1] else x_gen
                elif self.configs.reverse_scheduled_sampling == 1:
                    # reverse schedule sampling
                    if t == 0:
                        net = frames[:, t]
                    else:
                        net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
                else:
                    # schedule sampling
                    if t < self.configs.pre_seq_length:
                        net = frames[:, t]
                    else:
                        net = mask_true[:, t - self.configs.pre_seq_length] * frames[:, t] + \
                              (1 - mask_true[:, t - self.configs.pre_seq_length]) * x_gen

                h_t[0], c_t[0], memory, delta_c, delta_m = \
                    self.cell_list[0](net, h_t[0], c_t[0], memory)
                
                # 数值稳定性处理
                delta_c = apply_stability_measures(delta_c, "delta_c")
                delta_m = apply_stability_measures(delta_m, "delta_m")
                
                delta_c_adapter = self.adapter(delta_c)
                delta_m_adapter = self.adapter(delta_m)
                
                # 添加数值稳定性处理
                delta_c_adapter = apply_stability_measures(delta_c_adapter, "delta_c_adapter")
                delta_m_adapter = apply_stability_measures(delta_m_adapter, "delta_m_adapter")
                
                delta_c_list[0] = F.normalize(
                    delta_c_adapter.view(delta_c_adapter.shape[0], delta_c_adapter.shape[1], -1), dim=2)
                delta_m_list[0] = F.normalize(
                    delta_m_adapter.view(delta_m_adapter.shape[0], delta_m_adapter.shape[1], -1), dim=2)

                for i in range(1, self.num_layers):
                    h_t[i], c_t[i], memory, delta_c, delta_m = \
                        self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
                    
                    # 数值稳定性处理
                    delta_c = apply_stability_measures(delta_c, "delta_c")
                    delta_m = apply_stability_measures(delta_m, "delta_m")
                    
                    delta_c_adapter = self.adapter(delta_c)
                    delta_m_adapter = self.adapter(delta_m)
                    
                    # 添加数值稳定性处理
                    delta_c_adapter = apply_stability_measures(delta_c_adapter, "delta_c_adapter")
                    delta_m_adapter = apply_stability_measures(delta_m_adapter, "delta_m_adapter")
                    
                    delta_c_list[i] = F.normalize(
                        delta_c_adapter.view(delta_c_adapter.shape[0], delta_c_adapter.shape[1], -1), dim=2)
                    delta_m_list[i] = F.normalize(
                        delta_m_adapter.view(delta_m_adapter.shape[0], delta_m_adapter.shape[1], -1), dim=2)

                x_gen = self.conv_last(h_t[self.num_layers - 1])
                next_frames.append(x_gen)

                # decoupling loss
                if return_loss:
                    for i in range(0, self.num_layers):
                        cos_sim = torch.cosine_similarity(delta_c_list[i], delta_m_list[i], dim=2)
                        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
                        cos_sim = apply_stability_measures(cos_sim, "cos_sim")
                        decouple_loss.append(torch.mean(torch.abs(cos_sim)))

        if return_loss:
            decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        
        if return_loss:
            # 添加数值稳定性处理
            frames_tensor_clamped = torch.clamp(frames_tensor[:, 1:], min=0.0, max=1.0)
            next_frames_clamped = torch.clamp(next_frames, min=0.0, max=1.0)
            
            # 组合损失函数
            mse_loss = self.MSE_criterion(next_frames_clamped, frames_tensor_clamped)
            
            if self.SSIM_criterion:
                ssim_loss = self.SSIM_criterion(
                    next_frames_clamped.permute(0, 1, 4, 2, 3).flatten(0, 1),
                    frames_tensor_clamped.permute(0, 1, 4, 2, 3).flatten(0, 1)
                )
            else:
                ssim_loss = torch.tensor(0.0, device=device)
            
            decouple_loss_component = self.configs.decouple_beta * decouple_loss
            loss = mse_loss + self.configs.ssim_weight * ssim_loss + decouple_loss_component
            
            # 检查NaN损失
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                logger.warning("NaN/Inf detected in loss")
                loss = apply_stability_measures(loss, "loss")
        else:
            loss = None

        return next_frames, loss

# ==================== PredRNNv2 (优化版) ====================
class PredRNNv2(nn.Module):
    def __init__(self, configs):
        super(PredRNNv2, self).__init__()
        self.configs = configs
        num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        num_layers = len(num_hidden)
        self.model = PredRNNv2_Model(num_layers, num_hidden, configs)

    def forward(self, frames_tensor, mask_true, **kwargs):
        next_frames, loss = self.model(frames_tensor, mask_true, **kwargs)
        return next_frames, loss

# ==================== 数据处理函数 ====================
def reserve_schedule_sampling_exp(itr, batch_size, args):
    T, img_channel, img_height, img_width = args.in_shape
    if itr < args.r_sampling_step_1:
        r_eta = 0.5
    elif itr < args.r_sampling_step_2:
        r_eta = 1.0 - 0.5 * math.exp(-float(itr - args.r_sampling_step_1) / args.r_exp_alpha)
    else:
        r_eta = 1.0

    if itr < args.r_sampling_step_1:
        eta = 0.5
    elif itr < args.r_sampling_step_2:
        eta = 0.5 - (0.5 / (args.r_sampling_step_2 - args.r_sampling_step_1)) * (itr - args.r_sampling_step_1)
    else:
        eta = 0.0

    r_random_flip = np.random.random_sample(
        (batch_size, args.pre_seq_length - 1))
    r_true_token = (r_random_flip < r_eta)

    random_flip = np.random.random_sample(
        (batch_size, args.aft_seq_length - 1))
    true_token = (random_flip < eta)

    # 修复：使用原始尺寸而不是除以patch_size
    ones = np.ones((img_height, img_width, args.patch_size ** 2 * img_channel))
    zeros = np.zeros((img_height, img_width, args.patch_size ** 2 * img_channel))

    real_input_flag = []
    for i in range(batch_size):
        for j in range(args.total_length - 2):
            if j < args.pre_seq_length - 1:
                if r_true_token[i, j]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)
            else:
                if true_token[i, j - (args.pre_seq_length - 1)]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)

    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (batch_size,
                                  args.total_length - 2,
                                  img_height,
                                  img_width,
                                  args.patch_size ** 2 * img_channel))
    return torch.FloatTensor(real_input_flag).to(args.device)


def schedule_sampling(eta, itr, batch_size, args):
    T, img_channel, img_height, img_width = args.in_shape
    zeros = np.zeros((batch_size,
                      args.aft_seq_length - 1,
                      img_height,
                      img_width,
                      args.patch_size ** 2 * img_channel))
    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (batch_size, args.aft_seq_length - 1))
    true_token = (random_flip < eta)
    # 修复：使用原始尺寸而不是除以patch_size
    ones = np.ones((img_height, img_width, args.patch_size ** 2 * img_channel))
    zeros = np.zeros((img_height, img_width, args.patch_size ** 2 * img_channel))
    real_input_flag = []
    for i in range(batch_size):
        for j in range(args.aft_seq_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (batch_size,
                                  args.aft_seq_length - 1,
                                  img_height,
                                  img_width,
                                  args.patch_size ** 2 * img_channel))
    return eta, torch.FloatTensor(real_input_flag).to(args.device)


def reshape_patch(img_tensor, patch_size):
    assert 5 == img_tensor.ndim
    batch_size, seq_length, img_height, img_width, num_channels = img_tensor.shape
    a = img_tensor.reshape(batch_size, seq_length,
                                img_height//patch_size, patch_size,
                                img_width//patch_size, patch_size,
                                num_channels)
    b = a.transpose(3, 4)
    patch_tensor = b.reshape(batch_size, seq_length,
                                  img_height//patch_size,
                                  img_width//patch_size,
                                  patch_size*patch_size*num_channels)
    return patch_tensor


def reshape_patch_back(patch_tensor, patch_size):
    batch_size, seq_length, patch_height, patch_width, channels = patch_tensor.shape
    img_channels = channels // (patch_size*patch_size)
    a = patch_tensor.reshape(batch_size, seq_length,
                                  patch_height, patch_width,
                                  patch_size, patch_size,
                                  img_channels)
    b = a.transpose(3, 4)
    img_tensor = b.reshape(batch_size, seq_length,
                                patch_height * patch_size,
                                patch_width * patch_size,
                                img_channels)
    return img_tensor

# ==================== 智能NaN/Inf处理函数 ====================
def smart_nan_inf_handling(data):
    """
    智能处理NaN和Inf值：
    1. 首先尝试用邻近有效像素的平均值填充
    2. 如果无效区域太大，使用整个图像的有效像素均值填充
    3. 如果整个图像无效，使用0填充
    """
    # 创建无效像素的掩码
    invalid_mask = np.isnan(data) | np.isinf(data)
    
    # 如果没有无效像素，直接返回
    if not np.any(invalid_mask):
        return data
    
    # 计算有效像素的均值
    valid_mean = np.nanmean(data)
    if np.isnan(valid_mean) or np.isinf(valid_mean):
        valid_mean = 0.0
    
    # 尝试用邻近有效像素填充
    try:
        # 使用距离变换找到最近的邻居
        distance, indices = ndimage.distance_transform_edt(
            ~invalid_mask, return_indices=True)
        
        # 用最近邻居的值填充无效像素
        filled_data = np.copy(data)
        filled_data[invalid_mask] = data[tuple(indices[:, invalid_mask])]
        
        # 检查是否还有无效像素
        if np.any(np.isnan(filled_data) | np.isinf(filled_data)):
            # logger.warning("邻近填充后仍有无效像素，使用全局均值填充")
            filled_data[invalid_mask] = valid_mean
    except Exception as e:
        logger.warning(f"邻近填充失败: {e}, 使用全局均值填充")
        filled_data = np.copy(data)
        filled_data[invalid_mask] = valid_mean
    
    # 最终检查
    if np.any(np.isnan(filled_data) | np.isinf(filled_data)):
        logger.warning("填充后仍有无效像素，使用0填充")
        filled_data = np.nan_to_num(filled_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    return filled_data

# ==================== 数据集类 (优化版) ====================
class NPYDataset(Dataset):
    def __init__(self, data_path, sequence_length, patch_size=4, augment=True):
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.patch_size = patch_size
        self.augment = augment
        
        # 获取所有npy文件并排序
        self.files = sorted([f for f in os.listdir(data_path) if f.endswith('.npy')])
        logger.info(f"找到 {len(self.files)} 个.npy文件")
        
        # 计算全局统计信息用于归一化
        self.global_min, self.global_max = self._compute_global_stats()
        logger.info(f"全局最小值: {self.global_min}, 全局最大值: {self.global_max}")
        
    def _compute_global_stats(self):
        """计算整个数据集的全局最小值和最大值"""
        min_val = np.inf
        max_val = -np.inf
        
        # 随机采样部分文件计算全局统计
        sample_size = min(100, len(self.files))
        sample_files = random.sample(self.files, sample_size)
        
        for file in tqdm(sample_files, desc="计算全局统计"):
            data = np.load(os.path.join(self.data_path, file))
            if data.size == 0:
                continue
                
            # 智能处理NaN和Inf值
            if np.isnan(data).any() or np.isinf(data).any():
                data = smart_nan_inf_handling(data)
                
            min_val = min(min_val, np.min(data))
            max_val = max(max_val, np.max(data))
        
        return min_val, max_val
        
    def __len__(self):
        return max(0, len(self.files) - self.sequence_length + 1)
    
    def __getitem__(self, idx):
        # 读取连续的sequence_length个文件
        sequence = []
        for i in range(self.sequence_length):
            file_path = os.path.join(self.data_path, self.files[idx + i])
            data = np.load(file_path)
            
            # 确保数据是float32
            if data.dtype != np.float32:
                data = data.astype(np.float32)
            
            # 智能处理NaN和Inf值
            if np.isnan(data).any() or np.isinf(data).any():
                # logger.info(f"在文件 {self.files[idx + i]} 中检测到NaN/Inf，进行智能处理")
                data = smart_nan_inf_handling(data)
            
            # 使用全局统计归一化
            if self.global_max - self.global_min > 1e-8:
                data = (data - self.global_min) / (self.global_max - self.global_min)
            else:
                data = np.zeros_like(data)
            
            sequence.append(data)
        
        # 转换为numpy数组: [seq_len, H, W]
        sequence = np.array(sequence)
        
        # 数据增强
        if self.augment:
            sequence = self._apply_augmentation(sequence)
        
        # 添加通道维度: [seq_len, H, W] -> [seq_len, H, W, 1]
        sequence = np.expand_dims(sequence, axis=-1)
        
        # 应用patch处理
        if self.patch_size > 1:
            # 重塑为patch格式
            sequence_tensor = torch.FloatTensor(sequence)
            sequence_tensor = sequence_tensor.unsqueeze(0)  # [1, seq_len, H, W, 1]
            patched_sequence = reshape_patch(sequence_tensor, self.patch_size)
            sequence = patched_sequence.squeeze(0).numpy()  # [seq_len, H//patch, W//patch, patch*patch*1]
        
        return torch.FloatTensor(sequence)
    
    def _apply_augmentation(self, sequence):
        """应用随机数据增强"""
        # 随机水平翻转
        if random.random() > 0.5:
            sequence = sequence[:, :, ::-1]
        
        # 随机垂直翻转
        if random.random() > 0.5:
            sequence = sequence[:, ::-1, :]
        
        # 随机旋转 (0, 90, 180, 270度)
        rot = random.choice([0, 1, 2, 3] if sequence.shape[1] == sequence.shape[2] else [0, 2])
        if rot > 0:
            sequence = np.rot90(sequence, k=rot, axes=(1, 2))
        
        # 随机亮度调整
        brightness_factor = 0.2 * random.random() + 0.9  # 0.9-1.1
        sequence = np.clip(sequence * brightness_factor, 0, 1)
        
        # 随机添加高斯噪声
        if random.random() > 0.7:
            noise = np.random.normal(0, 0.05, sequence.shape)
            sequence = np.clip(sequence + noise, 0, 1)
        
        return sequence

# ==================== 配置类 (优化版) ====================
class Configs:
    def __init__(self):
        # 数据相关
        self.in_shape = [10, 1, 125, 125]  # [seq_length, channels, height, width] - patch处理后
        self.pre_seq_length = 5
        self.aft_seq_length = 5
        self.total_length = 10
        
        # 模型相关
        self.num_hidden = '64,64,64,64'  # 增加隐藏层大小
        self.filter_size = 5
        self.stride = 1
        self.patch_size = 4
        self.layer_norm = 0  # 禁用层归一化，使用批量归一化
        self.decouple_beta = 0.05  # 降低去耦损失权重
        self.use_ssim_loss = True  # 使用SSIM损失
        self.ssim_weight = 0.1  # SSIM损失权重
        self.use_amp = torch.cuda.is_available()  # 使用自动混合精度
        
        # 训练相关
        self.reverse_scheduled_sampling = 1
        self.r_sampling_step_1 = 25000
        self.r_sampling_step_2 = 50000
        self.r_exp_alpha = 5000
        
        self.scheduled_sampling = 1
        self.sampling_stop_iter = 50000
        self.sampling_start_value = 1.0
        self.sampling_changing_rate = 0.00002
        
        self.lr = 1e-4
        self.lr_decay = 0.98  # 学习率衰减
        self.batch_size = 2  # 增加batch size以加速训练
        self.epochs = 50  # 减少训练轮数
        self.grad_clip = 0.5  # 更严格的梯度裁剪
        
        self.fast_dev_run = True  # 快速开发模式，使用部分数据集
        
        # 其他
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== 训练函数 (优化版) ====================
def train_model(data_path, epochs=50, fast_dev_run=False, resume_from=None):
    # 确保导入 torch
    import torch
    import torch.optim
    import torch.nn.utils
    
    # 配置
    configs = Configs()
    
    # 启用CUDA优化
    cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    # 创建数据集和数据加载器
    configs.epochs = epochs
    configs.fast_dev_run = fast_dev_run
    train_dataset = NPYDataset(data_path, configs.total_length, configs.patch_size, augment=True)
    
    if len(train_dataset) == 0:
        raise ValueError(f"训练至少需要 {configs.total_length} 个按时间排列的 NPY 文件")
    sample = np.load(os.path.join(data_path, train_dataset.files[0]), allow_pickle=False)
    if sample.ndim != 2 or any(n % configs.patch_size for n in sample.shape):
        raise ValueError("训练网格的高和宽都必须是 4 的倍数")
    configs.in_shape[2:] = [n // configs.patch_size for n in sample.shape]

    # 使用部分数据集进行快速训练
    if configs.fast_dev_run:
        logger.info("快速开发模式：使用部分数据集")
        subset_size = min(100, len(train_dataset))
        train_dataset = Subset(train_dataset, range(subset_size))
    
    # 使用多进程数据加载器
    num_workers = 0 if fast_dev_run else min(4, max(0, (os.cpu_count() or 1) - 1))
    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=configs.use_amp, persistent_workers=num_workers > 0)
    
    # 创建模型
    model = PredRNNv2(configs).to(configs.device)
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, weight_decay=1e-5)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    
    # 混合精度梯度缩放
    scaler = GradScaler("cuda", enabled=configs.use_amp)
    
    # 训练日志
    train_losses = []
    start_epoch = 0
    best_loss = float('inf')
    
    # 如果提供了恢复点，加载模型和优化器状态
    if resume_from:
        checkpoint = torch.load(resume_from, map_location=configs.device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if checkpoint.get('scaler_state_dict'):
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint.get('train_losses', [])
        best_loss = min(train_losses, default=float('inf'))
        logger.info(f"恢复训练: epoch {start_epoch}/{configs.epochs}")

    logger.info(f"开始训练，使用设备: {configs.device}")
    logger.info(f"数据集大小: {len(train_dataset)}")
    logger.info(f"批次大小: {configs.batch_size}")
    logger.info(f"输入形状 (patch处理后): {configs.in_shape}")
    logger.info(f"使用 {num_workers} 个数据加载工作进程")
    
    # 创建目录保存模型和日志
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # 训练循环
    model.train()
    eta = configs.sampling_start_value
    
    # 梯度累积步数
    accumulation_steps = 2
    
    for epoch in range(start_epoch, configs.epochs):
        epoch_start_time = time.time()
        total_loss = 0
        num_batches = 0
        skipped_batches = 0
        
        # 使用tqdm显示进度条
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{configs.epochs}', mininterval=10)
        optimizer.zero_grad()
        
        for batch_idx, data in enumerate(progress_bar):
            # data shape: [batch_size, seq_len, H//patch, W//patch, patch*patch*C]
            data = data.to(configs.device, non_blocking=True)
            
            # 生成mask
            if configs.reverse_scheduled_sampling == 1:
                mask_true = reserve_schedule_sampling_exp(epoch * len(train_loader) + batch_idx, 
                                                        data.shape[0], configs)
            else:
                eta, mask_true = schedule_sampling(eta, epoch * len(train_loader) + batch_idx, 
                                                 data.shape[0], configs)
            
            # 前向传播
            with autocast("cuda", enabled=configs.use_amp):
                next_frames, loss = model(data, mask_true)
            
            # 检查NaN损失
            if loss is not None and (torch.isnan(loss).any() or torch.isinf(loss).any()):
                logger.warning(f"检测到NaN/Inf损失，跳过该批次 (Epoch {epoch+1}, Batch {batch_idx})")
                skipped_batches += 1
                continue
            
            # 反向传播
            if loss is not None:
                # 使用混合精度缩放损失
                group_start = (batch_idx // accumulation_steps) * accumulation_steps
                group_size = min(accumulation_steps, len(train_loader) - group_start)
                scaled_loss = scaler.scale(loss / group_size)
                scaled_loss.backward()
                
                # 每accumulation_steps步更新一次权重
                if (batch_idx + 1) % accumulation_steps == 0 or batch_idx + 1 == len(train_loader):
                    # 梯度裁剪
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=configs.grad_clip)
                    
                    # 更新参数
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                total_loss += loss.item()
                num_batches += 1
                    
                    # 更新进度条描述
                progress_bar.set_postfix(loss=loss.item())
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            train_losses.append(avg_loss)
            
            # 更新学习率
            scheduler.step(avg_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            epoch_time = time.time() - epoch_start_time
            logger.info(f'Epoch {epoch+1} completed. Time: {epoch_time:.2f}s, Loss: {avg_loss:.6f}, LR: {current_lr:.7f}')
            logger.info(f'跳过批次: {skipped_batches}/{len(train_loader)}')
        else:
            raise RuntimeError("该轮没有有效训练批次")
        
        # 每1个epoch保存一次模型
        checkpoint_path = f'checkpoints/predrnn_v2_epoch_{epoch+1}.pth'
        
        # 保存配置参数而不是整个对象
        configs_dict = {
            'in_shape': configs.in_shape,
            'pre_seq_length': configs.pre_seq_length,
            'aft_seq_length': configs.aft_seq_length,
            'total_length': configs.total_length,
            'num_hidden': configs.num_hidden,
            'filter_size': configs.filter_size,
            'stride': configs.stride,
            'patch_size': configs.patch_size,
            'layer_norm': configs.layer_norm,
            'decouple_beta': configs.decouple_beta,
            'reverse_scheduled_sampling': configs.reverse_scheduled_sampling,
            'r_sampling_step_1': configs.r_sampling_step_1,
            'r_sampling_step_2': configs.r_sampling_step_2,
            'r_exp_alpha': configs.r_exp_alpha,
            'scheduled_sampling': configs.scheduled_sampling,
            'sampling_stop_iter': configs.sampling_stop_iter,
            'sampling_start_value': configs.sampling_start_value,
            'sampling_changing_rate': configs.sampling_changing_rate,
            'lr': configs.lr,
            'lr_decay': configs.lr_decay,
            'batch_size': configs.batch_size,
            'epochs': configs.epochs,
            'grad_clip': configs.grad_clip,
            'use_ssim_loss': configs.use_ssim_loss,
            'ssim_weight': configs.ssim_weight,
            'use_amp': configs.use_amp,
            'device': str(configs.device)
        }
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses,
            'configs_dict': configs_dict,
            'scaler_state_dict': scaler.state_dict(),
        }, checkpoint_path)
        
        logger.info(f'模型已保存: {checkpoint_path}')
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_checkpoint_path = 'checkpoints/predrnn_v2_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'configs_dict': configs_dict,
                'train_losses': train_losses,
            }, best_checkpoint_path)
            logger.info(f'新的最佳模型已保存: {best_checkpoint_path}')
        
        # 绘制损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses)+1), train_losses, 'b-', label='Training Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('logs/training_loss.png')
        plt.close()

# ==================== 预测函数 (优化版) ====================
def predict_sequences(input_files, model_path, output_dir='./predictions'):
    """
    使用训练好的模型进行预测
    input_files: 输入文件路径列表
    model_path: 模型文件路径
    output_dir: 输出目录
    """
    # 配置
    configs = Configs()
    configs.pre_seq_length = len(input_files)
    configs.aft_seq_length = 5
    configs.total_length = configs.pre_seq_length + configs.aft_seq_length
    
    # 创建模型
    model = PredRNNv2(configs).to(configs.device)
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location=configs.device, weights_only=True)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 读取输入数据
    input_sequence = []
    for file_path in input_files:
        data = np.load(file_path)
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        
        # 智能处理NaN和Inf值
        if np.isnan(data).any() or np.isinf(data).any():
            # logger.info(f"在输入文件 {file_path} 中检测到NaN/Inf，进行智能处理")
            data = smart_nan_inf_handling(data)
        
        # 归一化到[0,1]
        data_min = data.min()
        data_max = data.max()
        if data_max - data_min > 1e-8:  # 避免除以零
            data = (data - data_min) / (data_max - data_min)
        else:
            data = np.zeros_like(data)  # 如果所有值相同，设为0
        
        input_sequence.append(data)
    
    # 转换为tensor
    input_sequence = np.array(input_sequence)
    input_sequence = np.expand_dims(input_sequence, axis=-1)  # [seq_len, H, W, 1]
    
    # 应用patch处理
    if configs.patch_size > 1:
        input_sequence_tensor = torch.FloatTensor(input_sequence)
        input_sequence_tensor = input_sequence_tensor.unsqueeze(0)  # [1, seq_len, H, W, 1]
        patched_input = reshape_patch(input_sequence_tensor, configs.patch_size)
        input_sequence = patched_input.squeeze(0).numpy()  # [seq_len, H//patch, W//patch, patch*patch*1]
    
    input_sequence = torch.FloatTensor(input_sequence).unsqueeze(0).to(configs.device)  # [1, seq_len, H//patch, W//patch, patch*patch*1]
    
    with torch.inference_mode():
        predictions, _ = model(input_sequence, None, return_loss=False)
        predictions = predictions[:, -configs.aft_seq_length:]

    # 保存预测结果
    os.makedirs(output_dir, exist_ok=True)
    predictions = predictions.cpu().numpy()[0]  # [seq_len, H//patch, W//patch, patch*patch*1]
    
    # 转换回原始形状
    predictions_tensor = torch.FloatTensor(predictions)
    predictions_tensor = predictions_tensor.unsqueeze(0)  # [1, seq_len, H//patch, W//patch, patch*patch*1]
    original_predictions = reshape_patch_back(predictions_tensor, configs.patch_size)
    predictions = original_predictions.squeeze(0).numpy()  # [seq_len, H, W, 1]
    
    # 可视化并保存结果
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    
    for i in range(predictions.shape[0]):
        pred_img = predictions[i, :, :, 0]  # [H, W]
        
        # 确保值在[0,1]范围内
        pred_img = np.clip(pred_img, 0, 1)
        
        # 保存为npy文件
        np.save(os.path.join(output_dir, f'prediction_{i:03d}.npy'), pred_img)
        
        # 保存为图像
        plt.figure(figsize=(10, 10))
        plt.imshow(pred_img, cmap='viridis')
        plt.colorbar()
        plt.title(f'Prediction Frame {i+1}')
        plt.savefig(os.path.join(output_dir, 'images', f'prediction_{i:03d}.png'), dpi=300)
        plt.close()
        
        logger.info(f'预测结果已保存: {os.path.join(output_dir, f"prediction_{i:03d}.npy")}')
    
    return predictions

# ==================== 主函数 ====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PredRNNv2 for your 500x500 npy data')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict'],
                        help='运行模式: train 或 predict')
    parser.add_argument('--data_path', type=str, default=None,
                        help='训练数据路径')
    parser.add_argument('--model_path', type=str, default='',
                        help='模型文件路径(用于预测)')
    parser.add_argument('--input_files', type=str, nargs='+', default=[],
                        help='输入文件路径列表(用于预测)')
    parser.add_argument('--output_dir', type=str, default='./predictions',
                        help='预测输出目录')
    parser.add_argument('--resume_from', type=str, default='',
                        help='从指定检查点恢复训练')
    parser.add_argument('--fast_dev_run', action='store_true',
                        help='快速开发模式（使用部分数据集）')
    
    parser.add_argument('--epochs', type=int, default=50, help='目标总轮数（含已恢复轮数）')
    args = parser.parse_args()
    
    if args.mode == 'train':
        if not args.data_path or not os.path.isdir(args.data_path) or args.epochs < 1:
            parser.error("训练需要有效的 --data_path 和正整数 --epochs")
        train_model(args.data_path, args.epochs, args.fast_dev_run, args.resume_from)
    elif args.mode == 'predict':
        if not args.model_path or not args.input_files:
            logger.error("预测模式需要提供 --model_path 和 --input_files 参数")
            exit(1)
        predict_sequences(args.input_files, args.model_path, args.output_dir)