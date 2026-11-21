# 降水预测平台

浏览器把按时间排列的 NPY 文件上传给 Spring Boot。后端按相同顺序转发全部帧，Flask 加载仓库内的 ConvLSTM 权重生成预测和 ZIP。Java 保存任务清单，图片和压缩包由 Python 提供。

## 启动

先按[仓库首页](../../README.md)建立并激活 Python 虚拟环境，安装推理依赖，再准备 JDK 17 和 Maven。在本目录运行 `bash start-all.sh`；Windows 可运行 `start-all.bat`。访问 <http://localhost:8080/api/>。可用 `SERVER_PORT` 更改页面端口，`RAIN_AI_PORT` 更改脚本启动的推理端口。两个服务均默认仅监听本机。macOS / Linux 启动脚本使用 Ctrl+C 停止两个进程，Windows 需同时关闭推理窗口。

也可分别启动。以下命令在本目录的两个终端执行，均以 macOS / Linux shell 为例：

```sh
PORT=5050 python ai-service/flask_app.py
```

```sh
mvn -f backend/pom.xml clean package
AI_SERVICE_URL=http://127.0.0.1:5050 java -jar backend/target/precipitation-prediction-1.0.0.jar
```

浏览器需要能访问 `AI_SERVICE_URL` 才能显示图片、下载文件。默认结果位于 `ai-service/results/<任务编号>/`；Java 任务清单在启动目录的 `results/`。两部分结果都应保留，以便再次打开历史任务。

## API

| 请求 | 用途 |
| --- | --- |
| `POST /api/prediction/predict` | multipart 重复传入 `files` 字段，附 `inputLength`、`predictLength` 和 `generateComparison` |
| `GET /api/prediction/result/{taskId}` | 查询处理时间、输入图片、预测图片、对比图和 ZIP 地址 |
| `GET /api/prediction/health` | 检查 Java 服务 |
| Python 的 `GET /health` | 检查权重能否加载 |

关闭 `generateComparison` 会省略拼图，保留逐帧图片和数组。客户端应使用响应中的 `downloadUrl` 下载。
