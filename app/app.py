import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import base64
import io
from PIL import Image
import matplotlib.pyplot as plt
import logging
import argparse
from flask import Flask, request, jsonify, send_file, render_template, redirect, url_for
from flask_cors import CORS
import glob
import scipy.ndimage as ndimage
import random
import time
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import uuid
import zipfile
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import traceback
import logging

def setup_chinese_font():
    """设置中文字体 - 彻底修复"""
    try:
        # 确保日志记录器可用
        global logger
        if 'logger' not in globals():
            logger = logging.getLogger(__name__)
        
        logger.info("开始设置中文字体...")
        
        # 尝试安装中文字体
        try:
            logger.info("尝试安装中文字体...")
            os.system('apt-get update > /dev/null 2>&1')
            os.system('apt-get install -y fonts-wqy-microhei > /dev/null 2>&1')
            os.system('apt-get install -y fonts-wqy-zenhei > /dev/null 2>&1')
            logger.info("中文字体安装完成")
        except Exception as e:
            logger.warning(f"安装中文字体失败: {str(e)}")
        
        # 设置matplotlib字体
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 验证字体是否可用
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        if 'WenQuanYi Micro Hei' in available_fonts:
            logger.info("成功设置文泉驿微米黑字体")
        else:
            logger.warning("文泉驿字体不可用，使用默认字体")
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
            
    except Exception as e:
        logger.error(f"设置中文字体失败: {str(e)}")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        

setup_chinese_font()
# 设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
warnings.filterwarnings("ignore", category=FutureWarning)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== 配置类 ====================
class Config:
    def __init__(self):
        # 数据相关
        self.in_shape = [10, 1, 128, 128]
        self.pre_seq_length = 5
        self.aft_seq_length = 5
        self.total_length = 10
        
        # 模型相关
        self.num_hidden = '32,32'
        self.filter_size = 3
        self.stride = 1
        self.layer_norm = False
        
        # 训练相关
        self.lr = 1e-4
        self.batch_size = 2
        self.epochs = 50
        self.grad_clip = 1.0
        self.patience = 8
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== 简化的LSTM单元 ====================
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=self.padding,
                              bias=bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

# ==================== 简化的PredRNN模型 ====================
class PredRNN(nn.Module):
    def __init__(self, config):
        super(PredRNN, self).__init__()
        self.config = config
        num_hidden = [int(x) for x in config.num_hidden.split(',')]
        self.num_layers = len(num_hidden)
        self.num_hidden = num_hidden
        
        cell_list = []
        for i in range(self.num_layers):
            in_channel = 1 if i == 0 else num_hidden[i-1]
            cell_list.append(ConvLSTMCell(in_channel, num_hidden[i], config.filter_size))
        self.cell_list = nn.ModuleList(cell_list)
        
        self.conv_last = nn.Conv2d(num_hidden[-1], 1, kernel_size=1, stride=1, padding=0, bias=False)
        
        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, frames_tensor, mask_true=None, return_loss=True):
        device = frames_tensor.device
        frames = frames_tensor.unsqueeze(2)
        
        batch_size, seq_len, _, height, width = frames.shape
        next_frames = []
        
        # 初始化隐藏状态
        h = [torch.zeros(batch_size, self.num_hidden[i], height, width, device=device) 
             for i in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.num_hidden[i], height, width, device=device) 
             for i in range(self.num_layers)]
        
        for t in range(seq_len - 1):
            net = frames[:, t]
            
            # 第一层
            h[0], c[0] = self.cell_list[0](net, (h[0], c[0]))
            
            # 后续层
            for i in range(1, self.num_layers):
                h[i], c[i] = self.cell_list[i](h[i-1], (h[i], c[i]))
            
            # 生成预测
            x_gen = self.conv_last(h[-1])
            next_frames.append(x_gen)
        
        next_frames = torch.stack(next_frames, dim=1)
        
        if return_loss:
            target = frames[:, 1:]
            if torch.all(target == 0):
                logger.warning("目标帧全为0，可能数据有问题")
                return next_frames.squeeze(2), torch.tensor(0.0, device=device)
            
            loss = F.mse_loss(next_frames, target)
            return next_frames.squeeze(2), loss
        
        return next_frames.squeeze(2), None

# ==================== 全局模型和配置 ====================
model = None
config = Config()

# ==================== 初始化模型 ====================
def initialize_model(model_path='best_model.pth'):
    global model, config
    try:
        logger.info("开始初始化模型...")
        
        # 创建模型
        model = PredRNN(config).to(config.device)
        
        # 加载模型权重
        if os.path.exists(model_path):
            logger.info(f"加载模型权重: {model_path}")
            checkpoint = torch.load(model_path, map_location=config.device)
            
            # 部分加载权重
            model_dict = model.state_dict()
            pretrained_dict = {}
            
            for k, v in checkpoint.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    pretrained_dict[k] = v
                else:
                    logger.warning(f"跳过权重 {k} (形状不匹配)")
            
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
            logger.info(f"成功加载 {len(pretrained_dict)} 个权重")
        else:
            logger.warning(f"模型权重文件不存在: {model_path}")
            # 创建虚拟模型用于测试
            model.dummy = True
        
        model.eval()
        logger.info("模型设置为评估模式")
    except Exception as e:
        logger.error(f"模型初始化失败: {str(e)}")
        model = None

# ==================== Flask应用 ====================
app = Flask(__name__)
CORS(app)

# 创建保存预测结果的目录
PREDICTION_DIR = 'predictions'
os.makedirs(PREDICTION_DIR, exist_ok=True)

# 修复中文字体问题
def setup_chinese_font():
    """设置中文字体"""
    try:
        # 尝试安装中文字体
        try:
            os.system('apt-get update > /dev/null 2>&1')
            os.system('apt-get install -y fonts-wqy-microhei > /dev/null 2>&1')
        except:
            pass
        
        # 设置字体
        font_list = [
            'WenQuanYi Micro Hei',  # 文泉驿微米黑
            'DejaVu Sans',
            'Arial',
            'sans-serif'
        ]
        
        plt.rcParams['font.sans-serif'] = font_list
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 12
        
        logger.info("中文字体设置完成")
    except Exception as e:
        logger.warning(f"设置中文字体失败: {e}")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']

# 初始化字体
setup_chinese_font()

# ==================== 辅助函数 ====================
def create_comparison_figure(input_images, pred_frames, session_id):
    """创建输入和预测结果的对比图 - 确保显示5个预测帧"""
    try:
        num_input = len(input_images)
        num_pred = len(pred_frames)
        
        # 创建2行5列的布局
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle('雷达图像预测结果对比', fontsize=16, fontproperties='WenQuanYi Micro Hei')
        
        # 绘制输入图像（第一行） - 总是显示5个
        for i in range(5):
            if i < num_input:
                img = input_images[i]
            else:
                # 如果输入帧不足5个，使用最后一个输入帧
                img = input_images[-1] if num_input > 0 else np.zeros((128, 128))
            
            ax = axes[0, i]
            im = ax.imshow(img, cmap='viridis')
            ax.set_title(f'输入帧 {i+1}', fontsize=12, fontproperties='WenQuanYi Micro Hei')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        # 绘制预测图像（第二行） - 总是显示5个
        for i in range(5):
            if i < num_pred:
                pred_img = pred_frames[i]
            else:
                # 如果预测帧不足5个，使用最后一个预测帧
                pred_img = pred_frames[-1] if num_pred > 0 else np.zeros((128, 128))
            
            pred_img = np.clip(pred_img, 0, 1)
            
            vmin = np.min(pred_img)
            vmax = np.max(pred_img)
            if vmax - vmin < 0.01:
                vmin, vmax = 0, 1
            
            ax = axes[1, i]
            im = ax.imshow(pred_img, cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_title(f'预测帧 {i+1}', fontsize=12, fontproperties='WenQuanYi Micro Hei')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        plt.tight_layout()
        comparison_path = os.path.join(PREDICTION_DIR, session_id, 'comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"对比图已保存: {comparison_path}")
        return comparison_path
    except Exception as e:
        logger.error(f"创建对比图失败: {str(e)}")
        return None

def create_zip_file(file_paths, session_id):
    """创建包含所有文件的zip文件"""
    try:
        zip_filename = f"{session_id}.zip"
        zip_path = os.path.join(PREDICTION_DIR, zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_path in file_paths:
                if os.path.exists(file_path):
                    zipf.write(file_path, os.path.basename(file_path))
        
        logger.info(f"创建zip文件: {zip_path}")
        return zip_path
    except Exception as e:
        logger.error(f"创建zip文件失败: {str(e)}")
        return None

# ==================== 路由 ====================
@app.route('/test')
def test_route():
    return "服务器正常运行!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("收到预测请求")
        uploaded_files = request.files.getlist('npy_files')
        logger.info(f"收到 {len(uploaded_files)} 个文件")
        
        if not uploaded_files:
            logger.warning("未上传文件")
            return jsonify({"error": "未上传文件"}), 400
        
        if len(uploaded_files) != config.pre_seq_length:
            logger.warning(f"文件数量错误: 需要 {config.pre_seq_length} 个文件, 收到 {len(uploaded_files)} 个")
            return jsonify({"error": f"请上传 {config.pre_seq_length} 个文件"}), 400
        
        # 生成唯一会话ID
        session_id = str(uuid.uuid4())
        logger.info(f"会话ID: {session_id}")
        
        # 预处理图像
        input_sequence = []
        input_images = []
        for file in uploaded_files:
            content = file.read()
            npy_data = np.load(io.BytesIO(content))
            
            if npy_data.dtype != np.float32:
                npy_data = npy_data.astype(np.float32)
            
            # 处理NaN和Inf值
            if np.isnan(npy_data).any() or np.isinf(npy_data).any():
                npy_data = np.nan_to_num(npy_data, nan=0.0)
            
            # 归一化
            data_min = npy_data.min()
            data_max = npy_data.max()
            if data_max - data_min > 1e-8:
                npy_data = (npy_data - data_min) / (data_max - data_min)
            else:
                npy_data = np.zeros_like(npy_data)
            
            input_sequence.append(npy_data)
            input_images.append(npy_data)
        
        # 转换为tensor
        input_sequence = np.array(input_sequence)
        input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0).to(config.device)
        
        # 进行预测
        with torch.no_grad():
            if hasattr(model, 'dummy'):
                # 使用虚拟模型生成测试数据 - 生成5个预测帧
                logger.warning("使用虚拟模型生成测试数据")
                predictions = torch.rand(1, 5, *config.in_shape[2:]).to(config.device)
            else:
                predictions, _ = model(input_tensor, return_loss=False)
        
        predictions = predictions.cpu().numpy()[0]
        
        # 保存结果
        session_dir = os.path.join(PREDICTION_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)
        os.makedirs(os.path.join(session_dir, 'images'), exist_ok=True)
        
        # 保存输入帧
        input_file_paths = []
        for i, img in enumerate(input_images):
            file_path = os.path.join(session_dir, f'input_{i:03d}.npy')
            np.save(file_path, img)
            input_file_paths.append(file_path)
            
            plt.figure(figsize=(8, 8))
            plt.imshow(img, cmap='viridis')
            plt.colorbar()
            plt.title(f'输入帧 {i+1}', fontproperties='WenQuanYi Micro Hei')
            plt.savefig(os.path.join(session_dir, 'images', f'input_{i:03d}.png'), dpi=200, bbox_inches='tight')
            plt.close()
        
        # 保存预测帧 - 保存5个预测结果
        pred_file_paths = []
        for i in range(min(len(predictions), 5)):  # 最多保存5个
            pred_img = predictions[i]
            pred_img = np.clip(pred_img, 0, 1)
            
            file_path = os.path.join(session_dir, f'prediction_{i:03d}.npy')
            np.save(file_path, pred_img)
            pred_file_paths.append(file_path)
            
            plt.figure(figsize=(8, 8))
            plt.imshow(pred_img, cmap='viridis')
            plt.colorbar()
            plt.title(f'预测帧 {i+1}', fontproperties='WenQuanYi Micro Hei')
            plt.savefig(os.path.join(session_dir, 'images', f'prediction_{i:03d}.png'), dpi=200, bbox_inches='tight')
            plt.close()
        
        # 创建对比图
        comparison_path = create_comparison_figure(input_images, predictions, session_id)
        
        # 创建zip文件
        zip_path = create_zip_file(input_file_paths + pred_file_paths, session_id)
        
        logger.info("预测成功完成")
        return jsonify({
            "message": "预测成功",
            "session_id": session_id,
            "zip_url": f"/download/{session_id}.zip" if zip_path else None,
            "comparison_url": f"/predictions/{session_id}/comparison.png"
        })
    
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"处理失败: {str(e)}\n{error_trace}")
        return jsonify({"error": f"处理失败: {str(e)}"}), 500

@app.route('/download/<filename>')
def download_file(filename):
    try:
        file_path = os.path.join(PREDICTION_DIR, filename)
        if not os.path.exists(file_path):
            return jsonify({"error": "文件不存在"}), 404
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predictions/<session_id>/<filename>')
def get_prediction_image(session_id, filename):
    try:
        image_path = os.path.join(PREDICTION_DIR, session_id, filename)
        if not os.path.exists(image_path):
            return jsonify({"error": "图片不存在"}), 404
        return send_file(image_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/result/<session_id>')
def show_result(session_id):
    return f"""

<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>预测结果 - 雷达图像预测系统</title>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --primary-color: #4361ee;
            --secondary-color: #3f37c9;
            --accent-color: #4895ef;
            --success-color: #4cc9f0;
            --danger-color: #f72585;
            --light-color: #f8f9fa;
            --dark-color: #212529;
            --gray-color: #6c757d;
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: 'Noto Sans SC', 'WenQuanYi Micro Hei', sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            color: var(--dark-color);
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            text-align: center;
            padding: 40px 0;
            background: linear-gradient(135deg, rgba(67, 97, 238, 0.8) 0%, rgba(63, 55, 201, 0.8) 100%);
            color: white;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
        }}
        
        .header h1 {{
            font-size: 2.8rem;
            margin-bottom: 15px;
            font-weight: 500;
        }}
        
        .header p {{
            font-size: 1.2rem;
            opacity: 0.9;
            max-width: 800px;
            margin: 0 auto;
        }}
        
        .card {{
            background: white;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            padding: 30px;
            margin-bottom: 30px;
        }}
        
        .card-title {{
            font-size: 1.5rem;
            margin-bottom: 20px;
            color: var(--primary-color);
            font-weight: 500;
            display: flex;
            align-items: center;
        }}
        
        .card-title i {{
            margin-right: 10px;
            font-size: 1.8rem;
        }}
        
        .result-image {{
            width: 100%;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            margin: 20px 0;
        }}
        
        .btn {{
            display: inline-block;
            padding: 12px 30px;
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            color: white;
            border: none;
            border-radius: 50px;
            font-size: 1.1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            box-shadow: 0 4px 15px rgba(67, 97, 238, 0.3);
            font-weight: 500;
        }}
        
        .btn:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(67, 97, 238, 0.4);
            color: white;
        }}
        
        .btn-secondary {{
            background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
            margin-right: 10px;
        }}
        
        .action-buttons {{
            display: flex;
            justify-content: center;
            margin-top: 30px;
        }}
        
        .result-details {{
            margin-top: 30px;
        }}
        
        .detail-item {{
            display: flex;
            margin-bottom: 15px;
        }}
        
        .detail-label {{
            font-weight: 500;
            width: 150px;
            color: var(--secondary-color);
        }}
        
        .detail-value {{
            flex: 1;
        }}
        
        .footer {{
            text-align: center;
            padding: 20px;
            color: var(--gray-color);
            font-size: 0.9rem;
        }}
        
        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 2rem;
            }}
            
            .header p {{
                font-size: 1rem;
            }}
            
            .card {{
                padding: 20px;
            }}
            
            .action-buttons {{
                flex-direction: column;
            }}
            
            .btn {{
                width: 100%;
                margin-bottom: 10px;
            }}
        }}
    </style>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-chart-line"></i> 预测结果</h1>
            <p>雷达图像预测系统生成的预测结果</p>
        </div>
        
        <div class="card">
            <div class="card-title">
                <i class="fas fa-image"></i> 预测结果对比
            </div>
            
            <img id="resultImage" class="result-image" src="/predictions/{session_id}/comparison.png" alt="预测结果对比图">
            
            <div class="result-details">
                <div class="detail-item">
                    <div class="detail-label">预测ID:</div>
                    <div class="detail-value">{session_id}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">生成时间:</div>
                    <div class="detail-value" id="timestamp">{time.strftime("%Y-%m-%d %H:%M:%S")}</div>
                </div>
            </div>
            
            <div class="action-buttons">
                <a href="/" class="btn btn-secondary">
                    <i class="fas fa-arrow-left"></i> 返回首页
                </a>
                <a href="/download/{session_id}.zip" class="btn" download>
                    <i class="fas fa-download"></i> 下载结果
                </a>
            </div>
        </div>
    </div>
    
    <div class="footer">
        <p>© 2023 雷达图像预测系统 | 基于深度学习的气象预测平台</p>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            // 更新时间为当前时间
            document.getElementById('timestamp').textContent = new Date().toLocaleString();
            
            // 检查图片是否加载失败
            document.getElementById('resultImage').onerror = function() {{
                this.src = '/predictions/{session_id}/comparison.png?' + new Date().getTime();
            }};
        }});
    </script>
</body>
</html>

"""

@app.route('/')
def index():
    return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>雷达图像预测系统</title>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-color: #4361ee;
            --secondary-color: #3f37c9;
            --accent-color: #4895ef;
            --success-color: #4cc9f0;
            --danger-color: #f72585;
            --light-color: #f8f9fa;
            --dark-color: #212529;
            --gray-color: #6c757d;
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: 'Noto Sans SC', 'WenQuanYi Micro Hei', sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            color: var(--dark-color);
            line-height: 1.6;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            padding: 40px 0;
            background: linear-gradient(135deg, rgba(67, 97, 238, 0.8) 0%, rgba(63, 55, 201, 0.8) 100%);
            color: white;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
        }
        
        .header h1 {
            font-size: 2.8rem;
            margin-bottom: 15px;
            font-weight: 500;
        }
        
        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
            max-width: 800px;
            margin: 0 auto;
        }
        
        .card {
            background: white;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            padding: 30px;
            margin-bottom: 30px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(0, 0, 0, 0.15);
        }
        
        .card-title {
            font-size: 1.5rem;
            margin-bottom: 20px;
            color: var(--primary-color);
            font-weight: 500;
            display: flex;
            align-items: center;
        }
        
        .card-title i {
            margin-right: 10px;
            font-size: 1.8rem;
        }
        
        .upload-area {
            border: 3px dashed var(--accent-color);
            border-radius: 12px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: rgba(72, 149, 239, 0.05);
            margin-bottom: 20px;
        }
        
        .upload-area:hover {
            background: rgba(72, 149, 239, 0.1);
            border-color: var(--secondary-color);
        }
        
        .upload-area i {
            font-size: 3rem;
            color: var(--accent-color);
            margin-bottom: 15px;
        }
        
        .upload-area h3 {
            font-size: 1.3rem;
            margin-bottom: 10px;
            color: var(--secondary-color);
        }
        
        .upload-area p {
            color: var(--gray-color);
            margin-bottom: 20px;
        }
        
        .btn {
            display: inline-block;
            padding: 12px 30px;
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            color: white;
            border: none;
            border-radius: 50px;
            font-size: 1.1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            box-shadow: 0 4px 15px rgba(67, 97, 238, 0.3);
            font-weight: 500;
        }
        
        .btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(67, 97, 238, 0.4);
            color: white;
        }
        
        .btn:disabled {
            background: var(--gray-color);
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        .btn-secondary {
            background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
        }
        
        .file-list {
            margin-top: 20px;
        }
        
        .file-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 15px;
            background: var(--light-color);
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 4px solid var(--accent-color);
        }
        
        .file-name {
            font-weight: 500;
        }
        
        .file-size {
            color: var(--gray-color);
            font-size: 0.9rem;
        }
        
        .progress-container {
            margin: 30px 0;
            display: none;
        }
        
        .progress-bar {
            height: 10px;
            background: #e9ecef;
            border-radius: 5px;
            overflow: hidden;
            margin-bottom: 10px;
        }
        
        .progress {
            height: 100%;
            background: linear-gradient(135deg, var(--success-color) 0%, var(--accent-color) 100%);
            width: 0%;
            transition: width 0.3s ease;
        }
        
        .progress-text {
            text-align: center;
            color: var(--gray-color);
            font-size: 0.9rem;
        }
        
        .alert {
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: none;
        }
        
        .alert-success {
            background: rgba(76, 201, 240, 0.1);
            border-left: 4px solid var(--success-color);
            color: var(--success-color);
        }
        
        .alert-danger {
            background: rgba(247, 37, 133, 0.1);
            border-left: 4px solid var(--danger-color);
            color: var(--danger-color);
        }
        
        .requirements {
            background: rgba(255, 193, 7, 0.1);
            border-left: 4px solid #ffc107;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        
        .requirements h4 {
            color: #856404;
            margin-bottom: 10px;
            font-weight: 500;
        }
        
        .requirements ul {
            padding-left: 20px;
        }
        
        .requirements li {
            margin-bottom: 5px;
        }
        
        .loading {
            text-align: center;
            padding: 30px;
            display: none;
        }
        
        .spinner {
            width: 60px;
            height: 60px;
            border: 6px solid rgba(72, 149, 239, 0.2);
            border-top: 6px solid var(--accent-color);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .footer {
            text-align: center;
            padding: 20px;
            color: var(--gray-color);
            font-size: 0.9rem;
        }
        
        @media (max-width: 768px) {
            .header h1 {
                font-size: 2rem;
            }
            
            .header p {
                font-size: 1rem;
            }
            
            .card {
                padding: 20px;
            }
            
            .upload-area {
                padding: 20px;
            }
            
            .btn {
                width: 100%;
                padding: 15px;
            }
        }
    </style>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-cloud-showers-heavy"></i> 雷达图像预测系统</h1>
            <p>基于深度学习的降水预测平台，为您提供精准的天气预测服务</p>
        </div>
        
        <div class="card">
            <div class="card-title">
                <i class="fas fa-info-circle"></i> 文件要求
            </div>
            <div class="requirements">
                <h4>请确保上传的文件符合以下要求：</h4>
                <ul>
                    <li>上传连续的雷达图像数据文件 (.npy格式)</li>
                    <li>需要上传 <strong>5个</strong> 连续的雷达图像文件</li>
                    <li>文件将按文件名顺序进行处理</li>
                    <li>文件大小建议不超过10MB</li>
                </ul>
            </div>
        </div>
        
        <div class="card">
            <div class="card-title">
                <i class="fas fa-cloud-upload-alt"></i> 上传雷达图像
            </div>
            
            <div id="uploadArea" class="upload-area">
                <i class="fas fa-file-upload"></i>
                <h3>点击或拖拽文件到此处上传</h3>
                <p>支持.npy格式的雷达图像文件</p>
                <input type="file" id="fileInput" class="d-none" multiple accept=".npy">
                <label for="fileInput" class="btn">
                    <i class="fas fa-folder-open"></i> 选择文件
                </label>
            </div>
            
            <div id="fileList" class="file-list"></div>
            
            <div id="alertSuccess" class="alert alert-success">
                <i class="fas fa-check-circle"></i> <span id="successMessage"></span>
            </div>
            
            <div id="alertError" class="alert alert-danger">
                <i class="fas fa-exclamation-circle"></i> <span id="errorMessage"></span>
            </div>
            
            <div class="progress-container" id="progressContainer">
                <div class="progress-bar">
                    <div class="progress" id="progress"></div>
                </div>
                <div class="progress-text" id="progressText">正在上传文件...</div>
            </div>
            
            <div class="text-center">
                <button id="predictBtn" class="btn" disabled>
                    <i class="fas fa-rocket"></i> 开始预测
                </button>
            </div>
            
            <div class="loading" id="loadingIndicator">
                <div class="spinner"></div>
                <p>系统正在处理您的雷达图像数据，请稍候...</p>
            </div>
        </div>
    </div>
    
    <div class="footer">
        <p>© 2023 雷达图像预测系统 | 基于深度学习的气象预测平台</p>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const fileInput = document.getElementById('fileInput');
            const uploadArea = document.getElementById('uploadArea');
            const fileList = document.getElementById('fileList');
            const predictBtn = document.getElementById('predictBtn');
            const loadingIndicator = document.getElementById('loadingIndicator');
            const alertSuccess = document.getElementById('alertSuccess');
            const alertError = document.getElementById('alertError');
            const successMessage = document.getElementById('successMessage');
            const errorMessage = document.getElementById('errorMessage');
            const progressContainer = document.getElementById('progressContainer');
            const progress = document.getElementById('progress');
            const progressText = document.getElementById('progressText');
            
            const REQUIRED_FILES = 5; // 需要5个文件
            
            let selectedFiles = [];
            
            // 处理文件选择
            fileInput.addEventListener('change', function(event) {
                selectedFiles = Array.from(event.target.files);
                updateFileList();
            });
            
            // 拖拽功能
            uploadArea.addEventListener('dragover', function(e) {
                e.preventDefault();
                this.style.borderColor = '#3f37c9';
                this.style.background = 'rgba(72, 149, 239, 0.2)';
            });
            
            uploadArea.addEventListener('dragleave', function(e) {
                e.preventDefault();
                this.style.borderColor = '#4895ef';
                this.style.background = 'rgba(72, 149, 239, 0.05)';
            });
            
            uploadArea.addEventListener('drop', function(e) {
                e.preventDefault();
                this.style.borderColor = '#4895ef';
                this.style.background = 'rgba(72, 149, 239, 0.05)';
                
                if (e.dataTransfer.files.length > 0) {
                    selectedFiles = Array.from(e.dataTransfer.files);
                    fileInput.files = e.dataTransfer.files;
                    updateFileList();
                }
            });
            
            // 点击上传区域
            uploadArea.addEventListener('click', function() {
                fileInput.click();
            });
            
            // 更新文件列表显示
            function updateFileList() {
                fileList.innerHTML = '';
                
                if (selectedFiles.length > 0) {
                    selectedFiles.forEach((file, index) => {
                        const fileItem = document.createElement('div');
                        fileItem.className = 'file-item';
                        fileItem.innerHTML = `
                            <div class="file-name">${index + 1}. ${file.name}</div>
                            <div class="file-size">${formatFileSize(file.size)}</div>
                        `;
                        fileList.appendChild(fileItem);
                    });
                    
                    // 检查文件数量
                    if (selectedFiles.length === REQUIRED_FILES) {
                        predictBtn.disabled = false;
                        hideAlert(alertError);
                    } else {
                        predictBtn.disabled = true;
                        showError(`请选择 ${REQUIRED_FILES} 个文件，当前选择了 ${selectedFiles.length} 个`);
                    }
                } else {
                    predictBtn.disabled = true;
                }
            }
            
            // 预测按钮点击事件
            predictBtn.addEventListener('click', async function() {
                if (selectedFiles.length !== REQUIRED_FILES) {
                    showError(`请选择 ${REQUIRED_FILES} 个文件`);
                    return;
                }
                
                // 重置状态
                hideMessages();
                loadingIndicator.style.display = 'block';
                progressContainer.style.display = 'block';
                predictBtn.disabled = true;
                
                const formData = new FormData();
                selectedFiles.forEach(file => {
                    formData.append('npy_files', file);
                });
                
                try {
                    // 模拟进度条
                    simulateProgress();
                    
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (!response.ok) {
                        throw new Error(data.error || '预测失败');
                    }
                    
                    // 显示成功消息
                    showSuccess('预测完成！正在跳转到结果页面...');
                    
                    // 跳转到结果页面
                    setTimeout(() => {
                        window.location.href = `/result/${data.session_id}`;
                    }, 1500);
                    
                } catch (error) {
                    showError('预测失败: ' + error.message);
                    console.error('Error:', error);
                } finally {
                    loadingIndicator.style.display = 'none';
                    progressContainer.style.display = 'none';
                    predictBtn.disabled = false;
                    progress.style.width = '0%';
                }
            });
            
            // 辅助函数
            function showError(message) {
                errorMessage.textContent = message;
                alertError.style.display = 'block';
                alertSuccess.style.display = 'none';
            }
            
            function showSuccess(message) {
                successMessage.textContent = message;
                alertSuccess.style.display = 'block';
                alertError.style.display = 'none';
            }
            
            function hideAlert(alertElement) {
                alertElement.style.display = 'none';
            }
            
            function hideMessages() {
                alertError.style.display = 'none';
                alertSuccess.style.display = 'none';
            }
            
            function formatFileSize(bytes) {
                if (bytes === 0) return '0 Bytes';
                const k = 1024;
                const sizes = ['Bytes', 'KB', 'MB', 'GB'];
                const i = Math.floor(Math.log(bytes) / Math.log(k));
                return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
            }
            
            function simulateProgress() {
                let width = 0;
                const interval = setInterval(() => {
                    if (width >= 90) {
                        clearInterval(interval);
                        return;
                    }
                    width += Math.random() * 10;
                    progress.style.width = Math.min(width, 90) + '%';
                    progressText.textContent = `正在处理... ${Math.floor(width)}%`;
                }, 200);
            }
        });
    </script>
</body>
</html>
"""



# ==================== 主函数 ====================
if __name__ == '__main__':
    # 初始化模型
    initialize_model()
    
    # 启动Flask应用
    logger.info("启动Flask应用")
    app.run(host='0.0.0.0', port=5000, debug=True)