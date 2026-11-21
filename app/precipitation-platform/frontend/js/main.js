// API基础URL
const API_BASE_URL = location.pathname.startsWith('/api') ? '/api' : 'http://localhost:8080/api';

// 全局变量
let selectedFiles = [];
let currentDownloadUrl = null;

// DOM元素
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const fileList = document.getElementById('fileList');
const predictBtn = document.getElementById('predictBtn');
const btnText = document.getElementById('btnText');
const progressSection = document.getElementById('progressSection');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const resultSection = document.getElementById('result');

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    checkHealth();
});

/**
 * 初始化事件监听器
 */
function initEventListeners() {
    // 上传区域点击
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });
    
    // 文件选择
    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });
    
    // 拖拽上传
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        handleFiles(e.dataTransfer.files);
    });
    
    // 预测按钮
    predictBtn.addEventListener('click', handlePredict);
    
    // 新建预测按钮
    const newPredictBtn = document.getElementById('newPredictBtn');
    if (newPredictBtn) {
        newPredictBtn.addEventListener('click', resetForm);
    }
    
    // 下载按钮
    const downloadBtn = document.getElementById('downloadBtn');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', handleDownload);
    }
}

/**
 * 处理文件选择
 */
function handleFiles(files) {
    const fileArray = Array.from(files);
    
    // 验证文件
    const validFiles = fileArray.filter(file => {
        if (!file.name.toLowerCase().endsWith('.npy')) {
            showToast('只支持NPY格式文件', 'error');
            return false;
        }
        if (file.size > 100 * 1024 * 1024) {
            showToast('文件大小不能超过100MB', 'error');
            return false;
        }
        return true;
    });
    
    // 添加到选中文件列表
    selectedFiles = [...selectedFiles, ...validFiles];
    
    // 更新UI
    updateFileList();
    updatePredictButton();
}

/**
 * 更新文件列表显示
 */
function updateFileList() {
    fileList.innerHTML = '';
    
    selectedFiles.forEach((file, index) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <div class="file-info">
                <span class="file-icon">📄</span>
                <div>
                    <div class="file-name"></div>
                    <div class="file-size">${formatFileSize(file.size)}</div>
                </div>
            </div>
            <button class="file-remove" onclick="removeFile(${index})">移除</button>
        `;
        fileItem.querySelector(".file-name").textContent = file.name;
        fileList.appendChild(fileItem);
    });
}

/**
 * 移除文件
 */
function removeFile(index) {
    selectedFiles.splice(index, 1);
    updateFileList();
    updatePredictButton();
}

/**
 * 更新预测按钮状态
 */
function updatePredictButton() {
    predictBtn.disabled = selectedFiles.length === 0;
}

/**
 * 执行预测
 */
async function handlePredict() {
    if (selectedFiles.length === 0) {
        showToast('请先上传文件', 'error');
        return;
    }
    
    // 获取参数
    const inputLength = parseInt(document.getElementById('inputLength').value);
    const predictLength = parseInt(document.getElementById('predictLength').value);
    const generateComparison = document.getElementById('generateComparison').checked;
    
    // 验证参数
    if (inputLength < 1 || inputLength > 20) {
        showToast('输入帧数量应在1-20之间', 'error');
        return;
    }
    
    if (predictLength < 1 || predictLength > 20) {
        showToast('预测帧数量应在1-20之间', 'error');
        return;
    }
    
    // 禁用按钮
    predictBtn.disabled = true;
    btnText.innerHTML = '<span class="loading"></span>处理中...';
    
    // 显示进度
    showProgress();
    
    try {
        // 构建FormData
        const formData = new FormData();
        selectedFiles.forEach(file => {
            formData.append('files', file);
        });
        formData.append('inputLength', inputLength);
        formData.append('predictLength', predictLength);
        formData.append('generateComparison', generateComparison);
        
        // 发送请求
        updateProgress(20, '正在上传文件...');
        
        const response = await fetch(`${API_BASE_URL}/prediction/predict`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.message || `请求失败 (HTTP ${response.status})`);
        }
        
        updateProgress(40, '文件上传完成，开始预测...');
        
        const result = await response.json();
        
        if (result.code !== 200) {
            throw new Error(result.message || '预测失败');
        }
        
        updateProgress(100, '预测完成！');
        
        // 延迟显示结果
        setTimeout(() => {
            hideProgress();
            displayResult(result.data);
        }, 1000);
        
    } catch (error) {
        console.error('预测失败:', error);
        showToast('预测失败: ' + error.message, 'error');
        hideProgress();
        resetPredictButton();
    }
}

/**
 * 显示进度
 */
function showProgress() {
    progressSection.style.display = 'block';
    resultSection.style.display = 'none';
    updateProgress(0, '准备中...');
    
    // 平滑滚动到进度区域
    progressSection.scrollIntoView({ behavior: 'smooth' });
}

/**
 * 更新进度
 */
function updateProgress(percent, text) {
    progressFill.style.width = percent + '%';
    progressText.textContent = text;
}

/**
 * 隐藏进度
 */
function hideProgress() {
    progressSection.style.display = 'none';
}

/**
 * 显示结果
 */
function displayResult(data) {
    currentDownloadUrl = data.downloadUrl;
    
    // 更新基本信息
    document.getElementById('taskId').textContent = data.taskId || '-';
    document.getElementById('processingTime').textContent = 
        data.processingTime ? `${data.processingTime.toFixed(2)}秒` : '-';
    
    // 显示输入图像
    const inputImagesContainer = document.getElementById('inputImages');
    inputImagesContainer.innerHTML = '';
    if (data.inputImages && data.inputImages.length > 0) {
        data.inputImages.forEach((url, index) => {
            const imgDiv = document.createElement('div');
            imgDiv.className = 'image-item';
            imgDiv.innerHTML = `<img src="${url}" alt="输入图像 ${index + 1}">`;
            inputImagesContainer.appendChild(imgDiv);
        });
    } else {
        inputImagesContainer.innerHTML = '<p>暂无输入图像</p>';
    }
    
    // 显示预测图像
    const predictionImagesContainer = document.getElementById('predictionImages');
    predictionImagesContainer.innerHTML = '';
    if (data.predictionImages && data.predictionImages.length > 0) {
        data.predictionImages.forEach((url, index) => {
            const imgDiv = document.createElement('div');
            imgDiv.className = 'image-item';
            imgDiv.innerHTML = `<img src="${url}" alt="预测图像 ${index + 1}">`;
            predictionImagesContainer.appendChild(imgDiv);
        });
    } else {
        predictionImagesContainer.innerHTML = '<p>暂无预测图像</p>';
    }
    
    // 显示对比图
    const comparisonContainer = document.getElementById('comparisonContainer');
    const comparisonImage = document.getElementById('comparisonImage');
    if (data.comparisonImage) {
        comparisonImage.src = data.comparisonImage;
        comparisonContainer.style.display = 'block';
    } else {
        comparisonContainer.style.display = 'none';
    }
    
    // 显示结果区域
    resultSection.style.display = 'block';
    resultSection.scrollIntoView({ behavior: 'smooth' });
    
    // 重置按钮
    resetPredictButton();
}

/**
 * 下载结果
 */
function handleDownload() {
    if (currentDownloadUrl) {
        const downloadUrl = currentDownloadUrl;
        window.open(downloadUrl, '_blank');
    } else {
        showToast('没有可下载的结果', 'error');
    }
}

/**
 * 重置表单
 */
function resetForm() {
    selectedFiles = [];
    currentDownloadUrl = null;
    fileInput.value = '';
    updateFileList();
    updatePredictButton();
    resultSection.style.display = 'none';
    
    // 滚动到上传区域
    document.getElementById('upload').scrollIntoView({ behavior: 'smooth' });
}

/**
 * 重置预测按钮
 */
function resetPredictButton() {
    predictBtn.disabled = selectedFiles.length === 0;
    btnText.textContent = '开始预测';
}

/**
 * 健康检查
 */
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/prediction/health`);
        const result = await response.json();
        
        if (result.code === 200) {
            console.log('✅ 后端服务运行正常');
        }
    } catch (error) {
        console.error('❌ 后端服务连接失败:', error);
        showToast('警告：无法连接到后端服务，请确保服务已启动', 'warning');
    }
}

/**
 * 格式化文件大小
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * 显示提示消息
 */
function showToast(message, type = 'info') {
    // 创建toast元素
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 1rem 1.5rem;
        background: ${type === 'error' ? '#ff4757' : type === 'warning' ? '#ffa502' : '#667eea'};
        color: white;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        z-index: 10000;
        animation: slideIn 0.3s ease;
    `;
    toast.textContent = message;
    
    document.body.appendChild(toast);
    
    // 3秒后移除
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => {
            document.body.removeChild(toast);
        }, 300);
    }, 3000);
}

// 添加动画样式
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

