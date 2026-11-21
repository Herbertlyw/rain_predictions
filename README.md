# 雷达回波外推

上传同一区域连续的雷达网格，模型会逐帧预测后续回波。浏览器可以查看输入与预测的对比图，也可以下载原始数值。仓库提供一个 Flask 单进程应用、一个 Spring Boot + Flask 的分层应用，以及独立的 PredRNNv2 训练脚本。

## 五分钟体验

准备 Python 3.11–3.13。首次安装 PyTorch 需要下载较大的依赖。以下命令从仓库根目录执行：

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python examples/make_frames.py
PORT=5050 python app/app.py
```

打开 <http://127.0.0.1:5050/>，按文件名顺序选中 `demo-frames` 内的五个 NPY 文件，点击「开始预测」。结果页提供两行对比图和 ZIP。压缩包内有五个 `prediction_*.npy`，与输入尺寸一致，数值使用输入网格的尺度。示例是人为生成的移动回波，仅用于验证流程。

Windows PowerShell 使用 `.\.venv\Scripts\Activate.ps1` 激活环境，并在启动前执行 `$env:PORT = "5050"`。直接启动且未指定端口时使用 5000。

## 输入和模型

- 每个 NPY 文件存放一个实数网格；接受 `(H, W)` 或 `(1, H, W)`，各帧同尺寸，每边最多 512 个像素。按时间先后选择文件，NaN、无穷值和不一致尺寸会返回输入错误。
- 两个 Web 入口共用 `app/forecast.py`，严格加载 `app/best_model.pth` 的两层 ConvLSTM 权重。整段序列共用最小值、最大值。编码所有观测帧后，模型将自己的预测作为下一步输入。
- 单进程入口固定五帧输入、五帧输出。分层应用支持 1–20 帧。没有未来实测值时，页面只显示预测，无法计算预报准确率。
- 在 `RAIN_RESULTS_DIR` 指定结果目录，或使用各入口的默认目录。每次预测分配独立任务编号。计算优先使用 CUDA，CPU 也能运行。

使用 `python -m unittest discover -s tests -v` 可验证上传、真实权重推理、ZIP 内容和无效输入的处理。

## 分层应用

先安装上面的 Python 依赖，另需 JDK 17 和 Maven。激活虚拟环境后运行：

```sh
bash app/precipitation-platform/start-all.sh
```

打开 <http://localhost:8080/api/>。Spring Boot 同时提供网页和业务 API，Python 推理服务默认由启动脚本绑定在本机 5050 端口。部署配置、结果查询和手动启动方法见[平台说明](app/precipitation-platform/README.md)。

## PredRNNv2 研究实验

`app/train.py` 使用另一套时空记忆网络。仓库内与当前实现匹配的研究权重为 `app/predrnn_v2_best.pth`。早期 `app/checkpoints/` 中含有其他网络结构的历史快照。使用 `requirements-train.txt` 安装研究脚本额外需要的 SciPy 和 tqdm。

```sh
python -m pip install -r requirements-train.txt
python app/train.py --mode predict --model_path app/predrnn_v2_best.pth \
  --input_files demo-frames/frame_*.npy --output_dir runs/v2-forecast
```

研究脚本导出的五帧为归一化值。自行训练时，准备至少 10 个连续的二维 NPY 文件，高宽均为 4 的倍数。在独立运行目录执行下面的命令，检查点和损失曲线会写入该目录。输入和脚本路径请换成自己的实际位置。

```sh
python /path/to/rain_predictions/app/train.py --mode train --data_path /path/to/radar-frames --epochs 1 --fast_dev_run
```

继续训练可显式传入 `--resume_from checkpoints/predrnn_v2_epoch_1.pth --epochs 2`。默认从头训练。仓库保留研究材料和历史输出；完整训练数据集需自行准备。
