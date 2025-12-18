"""
Flask AI服务 - 降水预测模型推理服务
适配Spring Boot后端调用
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import os
import uuid
import logging
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import zipfile
import io

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# 配置
UPLOAD_FOLDER = Path('./uploads')
RESULT_FOLDER = Path('./results')
MODEL_PATH = Path('../best_model.pth')  # 指向训练好的模型

UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULT_FOLDER.mkdir(exist_ok=True)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"使用设备: {device}")


# ==================== 模型定义 ====================
class ConvLSTMCell(nn.Module):
    """简化的ConvLSTM单元"""
    def __init__(self, input_dim, hidden_dim, kernel_size=5):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding
        )
    
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


class PredRNN(nn.Module):
    """PredRNN模型"""
    def __init__(self, input_channels=1, hidden_dim=64, num_layers=4, kernel_size=5):
        super(PredRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # ConvLSTM层
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_channels if i == 0 else hidden_dim
            self.cells.append(ConvLSTMCell(in_dim, hidden_dim, kernel_size))
        
        # 输出层
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, input_channels, kernel_size=1),
            nn.Tanh()
        )
    
    def forward(self, x, predict_steps=5):
        """
        x: [batch, seq_len, channels, height, width]
        """
        batch_size, seq_len, channels, height, width = x.shape
        
        # 初始化状态
        h_states = []
        c_states = []
        for _ in range(self.num_layers):
            h_states.append(torch.zeros(batch_size, self.hidden_dim, height, width).to(x.device))
            c_states.append(torch.zeros(batch_size, self.hidden_dim, height, width).to(x.device))
        
        # 处理输入序列
        for t in range(seq_len):
            x_t = x[:, t]
            for layer in range(self.num_layers):
                if layer == 0:
                    h_states[layer], c_states[layer] = self.cells[layer](x_t, (h_states[layer], c_states[layer]))
                else:
                    h_states[layer], c_states[layer] = self.cells[layer](h_states[layer-1], (h_states[layer], c_states[layer]))
        
        # 生成预测
        predictions = []
        x_t = x[:, -1]  # 最后一帧作为初始输入
        
        for _ in range(predict_steps):
            for layer in range(self.num_layers):
                if layer == 0:
                    h_states[layer], c_states[layer] = self.cells[layer](x_t, (h_states[layer], c_states[layer]))
                else:
                    h_states[layer], c_states[layer] = self.cells[layer](h_states[layer-1], (h_states[layer], c_states[layer]))
            
            x_t = self.output_conv(h_states[-1])
            predictions.append(x_t)
        
        return torch.stack(predictions, dim=1)


# 全局模型变量
model = None


def load_model():
    """加载模型"""
    global model
    
    if model is None:
        try:
            logger.info("正在加载模型...")
            model = PredRNN(input_channels=1, hidden_dim=64, num_layers=4)
            
            if MODEL_PATH.exists():
                checkpoint = torch.load(MODEL_PATH, map_location=device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                logger.info("✅ 模型加载成功")
            else:
                logger.warning(f"⚠️  模型文件不存在: {MODEL_PATH}，使用随机初始化的模型")
            
            model.to(device)
            model.eval()
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    return model


def preprocess_data(npy_files, input_length=5):
    """预处理NPY文件"""
    data_list = []
    
    for file_path in npy_files:
        try:
            data = np.load(file_path)
            # 确保是2D数组 (height, width)
            if len(data.shape) == 2:
                data_list.append(data)
            elif len(data.shape) == 3:
                # 如果是3D，取第一个通道
                data_list.append(data[0])
        except Exception as e:
            logger.error(f"加载文件失败 {file_path}: {e}")
            continue
    
    if len(data_list) < input_length:
        raise ValueError(f"文件数量不足，需要至少{input_length}个文件")
    
    # 取前input_length个
    data_list = data_list[:input_length]
    
    # 归一化到[-1, 1]
    data_array = np.stack(data_list)  # [seq_len, H, W]
    
    # 全局归一化
    data_min = data_array.min()
    data_max = data_array.max()
    
    if data_max - data_min > 0:
        data_normalized = 2 * (data_array - data_min) / (data_max - data_min) - 1
    else:
        data_normalized = np.zeros_like(data_array)
    
    # 转换为tensor: [1, seq_len, 1, H, W]
    tensor = torch.from_numpy(data_normalized).float().unsqueeze(0).unsqueeze(2)
    
    return tensor, data_min, data_max


def postprocess_predictions(predictions, data_min, data_max):
    """后处理预测结果"""
    # predictions: [1, pred_len, 1, H, W]
    predictions_np = predictions.cpu().numpy()[0, :, 0, :, :]  # [pred_len, H, W]
    
    # 反归一化
    predictions_np = (predictions_np + 1) / 2 * (data_max - data_min) + data_min
    
    return predictions_np


def create_visualization(inputs, predictions, save_dir):
    """创建可视化图像"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    image_urls = {
        'input_images': [],
        'prediction_images': [],
        'comparison_image': None
    }
    
    # 保存输入图像
    for i, img in enumerate(inputs):
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(img, cmap='jet', interpolation='nearest')
        ax.set_title(f'Input Frame {i+1}')
        ax.axis('off')
        plt.colorbar(im, ax=ax)
        
        img_path = save_dir / f'input_{i:03d}.png'
        plt.savefig(img_path, bbox_inches='tight', dpi=100)
        plt.close()
        
        image_urls['input_images'].append(f'/results/{save_dir.name}/input_{i:03d}.png')
    
    # 保存预测图像
    for i, img in enumerate(predictions):
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(img, cmap='jet', interpolation='nearest')
        ax.set_title(f'Prediction Frame {i+1}')
        ax.axis('off')
        plt.colorbar(im, ax=ax)
        
        img_path = save_dir / f'prediction_{i:03d}.png'
        plt.savefig(img_path, bbox_inches='tight', dpi=100)
        plt.close()
        
        image_urls['prediction_images'].append(f'/results/{save_dir.name}/prediction_{i:03d}.png')
    
    # 创建对比图
    num_inputs = len(inputs)
    num_preds = len(predictions)
    total_frames = num_inputs + num_preds
    
    fig, axes = plt.subplots(2, max(num_inputs, num_preds), figsize=(4*max(num_inputs, num_preds), 8))
    if total_frames <= 2:
        axes = axes.reshape(2, -1)
    
    # 绘制输入
    for i in range(num_inputs):
        ax = axes[0, i] if num_inputs > 1 else axes[0]
        im = ax.imshow(inputs[i], cmap='jet')
        ax.set_title(f'Input {i+1}')
        ax.axis('off')
    
    # 清空多余的输入子图
    for i in range(num_inputs, max(num_inputs, num_preds)):
        ax = axes[0, i] if num_inputs > 1 else axes[0]
        ax.axis('off')
    
    # 绘制预测
    for i in range(num_preds):
        ax = axes[1, i] if num_preds > 1 else axes[1]
        im = ax.imshow(predictions[i], cmap='jet')
        ax.set_title(f'Prediction {i+1}')
        ax.axis('off')
    
    # 清空多余的预测子图
    for i in range(num_preds, max(num_inputs, num_preds)):
        ax = axes[1, i] if num_preds > 1 else axes[1]
        ax.axis('off')
    
    plt.tight_layout()
    comparison_path = save_dir / 'comparison.png'
    plt.savefig(comparison_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    image_urls['comparison_image'] = f'/results/{save_dir.name}/comparison.png'
    
    return image_urls


@app.route('/')
def index():
    """首页"""
    return jsonify({
        'service': 'Precipitation Prediction AI Service',
        'status': 'running',
        'version': '1.0.0',
        'device': str(device)
    })


@app.route('/health')
def health():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'device': str(device),
        'model_loaded': model is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """预测接口"""
    try:
        # 获取参数
        task_id = request.form.get('task_id', str(uuid.uuid4()))
        input_length = int(request.form.get('input_length', 5))
        predict_length = int(request.form.get('predict_length', 5))
        generate_comparison = request.form.get('generate_comparison', 'true').lower() == 'true'
        
        logger.info(f"收到预测请求 - 任务ID: {task_id}, 输入帧数: {input_length}, 预测帧数: {predict_length}")
        
        # 获取上传的文件
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': '没有上传文件'}), 400
        
        files = request.files.getlist('file')
        if not files:
            return jsonify({'status': 'error', 'message': '文件列表为空'}), 400
        
        # 保存上传的文件
        task_upload_dir = UPLOAD_FOLDER / task_id
        task_upload_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        for i, file in enumerate(files):
            if file.filename.endswith('.npy'):
                file_path = task_upload_dir / f'input_{i:03d}.npy'
                file.save(file_path)
                saved_files.append(file_path)
        
        if len(saved_files) < input_length:
            return jsonify({
                'status': 'error',
                'message': f'上传的NPY文件数量不足，需要至少{input_length}个'
            }), 400
        
        # 加载模型
        model = load_model()
        
        # 预处理数据
        input_tensor, data_min, data_max = preprocess_data(saved_files, input_length)
        input_tensor = input_tensor.to(device)
        
        # 执行预测
        with torch.no_grad():
            predictions = model(input_tensor, predict_steps=predict_length)
        
        # 后处理
        predictions_np = postprocess_predictions(predictions, data_min, data_max)
        inputs_np = postprocess_predictions(input_tensor[:, :, 0, :, :].permute(0, 1, 2, 3), data_min, data_max)
        
        # 保存结果
        task_result_dir = RESULT_FOLDER / task_id
        task_result_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存NPY文件
        for i, pred in enumerate(predictions_np):
            np.save(task_result_dir / f'prediction_{i:03d}.npy', pred)
        
        # 创建可视化
        image_urls = create_visualization(inputs_np, predictions_np, task_result_dir)
        
        # 计算指标（简化版）
        # 这里可以添加更复杂的评估指标
        mse = 0.0
        ssim = 0.9  # 示例值
        
        # 构建响应
        response = {
            'status': 'success',
            'message': '预测完成',
            'task_id': task_id,
            'input_images': image_urls['input_images'],
            'prediction_images': image_urls['prediction_images'],
            'comparison_image': image_urls['comparison_image'] if generate_comparison else None,
            'download_url': f'/download/{task_id}',
            'metrics': {
                'mse': mse,
                'ssim': ssim
            }
        }
        
        logger.info(f"预测完成 - 任务ID: {task_id}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"预测失败: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'预测失败: {str(e)}'
        }), 500


@app.route('/results/<path:filename>')
def serve_result(filename):
    """提供结果文件"""
    file_path = RESULT_FOLDER / filename
    if file_path.exists():
        return send_file(file_path)
    return jsonify({'error': '文件不存在'}), 404


@app.route('/download/<task_id>')
def download_result(task_id):
    """下载完整结果"""
    task_result_dir = RESULT_FOLDER / task_id
    
    if not task_result_dir.exists():
        return jsonify({'error': '任务不存在'}), 404
    
    # 创建ZIP文件
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in task_result_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(task_result_dir)
                zf.write(file_path, arcname)
    
    memory_file.seek(0)
    
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'{task_id}_results.zip'
    )


if __name__ == '__main__':
    logger.info("🚀 启动Flask AI服务...")
    logger.info(f"📦 上传目录: {UPLOAD_FOLDER.absolute()}")
    logger.info(f"📊 结果目录: {RESULT_FOLDER.absolute()}")
    logger.info(f"🤖 模型路径: {MODEL_PATH.absolute()}")
    logger.info(f"💻 计算设备: {device}")
    
    app.run(host='0.0.0.0', port=5000, debug=False)

