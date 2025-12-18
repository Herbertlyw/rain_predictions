#!/bin/bash
# 降水预测平台 - 一键启动脚本

echo "========================================"
echo "🌧️  降水预测平台启动脚本"
echo "========================================"

# 检查是否安装了所需的命令
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "❌ $1 未安装，请先安装"
        exit 1
    fi
}

echo "📋 检查环境..."
check_command python3
check_command java
check_command mvn

# 启动AI服务
echo ""
echo "🚀 启动AI服务 (Flask)..."
cd ai-service
if [ ! -d "venv" ]; then
    echo "创建Python虚拟环境..."
    python3 -m venv venv
fi

source venv/bin/activate
pip install -q -r requirements.txt

echo "✅ AI服务启动中 (http://localhost:5000)..."
python flask_app.py &
AI_PID=$!
cd ..

sleep 3

# 启动Spring Boot后端
echo ""
echo "🚀 启动Spring Boot后端..."
cd backend
echo "✅ 后端服务启动中 (http://localhost:8080/api)..."
mvn spring-boot:run &
BACKEND_PID=$!
cd ..

sleep 5

# 启动前端
echo ""
echo "🚀 启动前端服务..."
cd frontend
echo "✅ 前端服务启动中 (http://localhost:3000)..."
python3 -m http.server 3000 &
FRONTEND_PID=$!
cd ..

echo ""
echo "========================================"
echo "✅ 所有服务启动完成！"
echo "========================================"
echo "📍 访问地址："
echo "   前端: http://localhost:3000"
echo "   后端: http://localhost:8080/api"
echo "   AI服务: http://localhost:5000"
echo ""
echo "💡 使用 Ctrl+C 停止所有服务"
echo "========================================"

# 等待用户中断
trap "echo ''; echo '🛑 正在停止所有服务...'; kill $AI_PID $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo '✅ 所有服务已停止'; exit 0" INT

wait

