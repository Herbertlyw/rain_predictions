# 降水预测平台

基于超图神经网络的雷达回波外推研究 - 完整Web应用

## 📖 项目简介

本项目是一个完整的降水预测Web平台，采用前后端分离的微服务架构。通过上传雷达回波NPY数据文件，利用基于超图神经网络理论的深度学习模型进行未来降水预测，并提供可视化结果展示和下载功能。

### 核心创新点

- **双层超图架构**：设计空间超图状态（c_t）和时间超图状态（m_t），分别建模空间相关性和时间演变规律
- **超图解耦机制**：通过余弦相似度最小化确保两层超图学习不同特征，避免特征冗余
- **卷积-超图等价性**：从理论上证明规则网格上卷积操作与超图卷积的数学等价性
- **微服务架构**：前端（HTML/CSS/JS）、业务后端（Spring Boot）、AI服务（Python Flask）三层分离

## 🏗️ 系统架构

```
┌─────────────────┐
│   前端 Web      │  HTML5 + CSS3 + JavaScript
│   (静态页面)    │  用户交互、文件上传、结果展示
└────────┬────────┘
         │ HTTP/REST
         ↓
┌─────────────────┐
│  业务逻辑层     │  Java Spring Boot
│  (后端API)     │  请求验证、文件管理、服务编排
└────────┬────────┘
         │ HTTP/REST
         ↓
┌─────────────────┐
│   AI服务层      │  Python Flask
│  (模型推理)    │  深度学习模型、预测计算、可视化
└─────────────────┘
```

### 技术栈

**前端**
- HTML5 / CSS3 / JavaScript (原生)
- 响应式设计、拖拽上传、动画效果

**后端（业务层）**
- Java 17
- Spring Boot 3.2.0
- Apache HttpClient 5
- Maven

**AI服务**
- Python 3.8+
- Flask 3.0
- PyTorch 2.1
- NumPy、Matplotlib

**通信协议**
- RESTful API
- JSON数据格式
- Multipart文件上传

## 📦 项目结构

```
precipitation-platform/
├── frontend/                    # 前端代码
│   ├── index.html              # 主页面
│   ├── css/
│   │   └── style.css           # 样式文件
│   └── js/
│       └── main.js             # 交互逻辑
├── backend/                     # Spring Boot后端
│   ├── pom.xml                 # Maven配置
│   └── src/main/
│       ├── java/com/precipitation/
│       │   ├── PrecipitationApplication.java      # 主程序
│       │   ├── controller/
│       │   │   └── PredictionController.java      # 控制器
│       │   ├── service/
│       │   │   └── PredictionService.java         # 业务逻辑
│       │   ├── model/                             # 数据模型
│       │   ├── config/                            # 配置类
│       │   └── util/                              # 工具类
│       └── resources/
│           └── application.yml                     # 配置文件
└── ai-service/                  # Python AI服务
    ├── flask_app.py            # Flask应用
    └── requirements.txt         # Python依赖
```

## 🚀 快速开始

### 环境要求

- **Java**: JDK 17+
- **Maven**: 3.6+
- **Python**: 3.8+
- **Node.js**: (可选，用于前端开发服务器)

### 1. 启动AI服务

```bash
cd ai-service

# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 启动服务
python flask_app.py
```

AI服务将在 `http://localhost:5000` 启动

### 2. 启动Spring Boot后端

```bash
cd backend

# 使用Maven编译和运行
mvn clean install
mvn spring-boot:run

# 或者直接运行jar包
java -jar target/precipitation-prediction-1.0.0.jar
```

后端API将在 `http://localhost:8080/api` 启动

### 3. 启动前端

**方式一：使用静态文件服务器**

```bash
cd frontend

# 使用Python内置服务器
python3 -m http.server 3000

# 或使用Node.js serve
npx serve -s . -p 3000
```

访问 `http://localhost:3000`

**方式二：直接打开**

直接在浏览器中打开 `frontend/index.html` 文件

## 📝 使用说明

### 数据准备

1. 准备NPY格式的雷达回波数据文件
2. 每个文件应为2D数组（高度×宽度）或3D数组（通道×高度×宽度）
3. 文件大小不超过100MB

### 预测流程

1. **上传数据**：点击或拖拽NPY文件到上传区域
2. **配置参数**：
   - 输入帧数量：用于预测的历史帧数（1-20）
   - 预测帧数量：需要预测的未来帧数（1-20）
   - 生成对比图：是否创建输入输出对比可视化
3. **开始预测**：点击"开始预测"按钮
4. **查看结果**：
   - 任务信息（任务ID、处理时间）
   - 性能指标（MSE、SSIM）
   - 输入图像展示
   - 预测结果展示
   - 对比图展示
5. **下载结果**：点击下载按钮获取完整结果压缩包

## 🔧 配置说明

### 后端配置 (application.yml)

```yaml
server:
  port: 8080                    # 后端端口

ai:
  service:
    url: http://localhost:5000  # AI服务地址
    timeout: 300000             # 超时时间（毫秒）

file:
  storage:
    upload-dir: ./uploads       # 上传目录
    result-dir: ./results       # 结果目录
    max-file-size: 104857600    # 最大文件大小
```

### 前端配置 (main.js)

```javascript
const API_BASE_URL = 'http://localhost:8080/api';  // 后端API地址
```

### AI服务配置 (flask_app.py)

```python
MODEL_PATH = Path('../best_model.pth')  # 模型文件路径
```

## 📊 API文档

### 1. 健康检查

**请求**
```
GET /api/prediction/health
```

**响应**
```json
{
  "code": 200,
  "message": "服务运行正常",
  "data": "服务运行正常",
  "timestamp": 1702886400000
}
```

### 2. 执行预测

**请求**
```
POST /api/prediction/predict
Content-Type: multipart/form-data

files: [file1.npy, file2.npy, ...]
inputLength: 5
predictLength: 5
generateComparison: true
```

**响应**
```json
{
  "code": 200,
  "message": "预测成功",
  "data": {
    "taskId": "uuid",
    "status": "success",
    "processingTime": 12.34,
    "inputImages": [...],
    "predictionImages": [...],
    "comparisonImage": "...",
    "downloadUrl": "/download/uuid",
    "mse": 0.001234,
    "ssim": 0.9234
  },
  "timestamp": 1702886400000
}
```

### 3. 查询结果

**请求**
```
GET /api/prediction/result/{taskId}
```

**响应**
```json
{
  "code": 200,
  "message": "查询成功",
  "data": {
    "taskId": "uuid",
    "status": "success",
    ...
  },
  "timestamp": 1702886400000
}
```

## 🎯 项目特点

### 1. 前后端分离
- 前端独立部署，可使用任何静态文件服务器
- 后端提供RESTful API，支持跨域访问
- 便于维护和扩展

### 2. 微服务架构
- 业务逻辑层（Spring Boot）与AI计算层（Flask）分离
- 各服务独立部署、独立扩展
- 支持负载均衡和高可用

### 3. 用户友好
- 现代化UI设计
- 拖拽上传文件
- 实时进度显示
- 响应式布局，支持移动端

### 4. 可扩展性
- 易于添加新的模型
- 支持多种数据格式
- 可集成数据库存储历史记录
- 可添加用户认证系统

## 🔍 常见问题

### Q1: 后端无法连接AI服务

**A**: 检查AI服务是否启动，确认 `application.yml` 中的 `ai.service.url` 配置正确

### Q2: 上传文件失败

**A**: 检查文件大小是否超过限制（默认100MB），确认文件格式为NPY

### Q3: 预测时间过长

**A**: 
- 检查是否使用GPU（CUDA）
- 减少输入/预测帧数
- 降低图像分辨率

### Q4: 前端无法访问后端API

**A**: 检查CORS配置，确认后端已启用跨域支持

## 📚 项目背景

本项目为北京林业大学大学生创新创业训练项目成果，项目团队：

- **项目主持人**：刘禹伟
- **团队成员**：钱胜波、曹丽君、赵雨薇
- **指导教师**：霍光煜

### 研究内容

通过将卷积神经网络的空间卷积操作重新解释为超图卷积，并设计独立的时间超图记忆状态，实现了空间特征与时间依赖的有效解耦，提升了降水临近预报的准确性。

## 📄 许可证

本项目仅用于学术研究和教学目的。

## 🙏 致谢

感谢北京林业大学信息学院提供的支持与指导。

---

**联系方式**：如有问题，请联系项目团队或指导老师。

**最后更新**：2025年12月

