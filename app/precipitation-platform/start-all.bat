@echo off
REM 降水预测平台 - Windows一键启动脚本
chcp 65001 >nul

echo ========================================
echo 🌧️  降水预测平台启动脚本
echo ========================================

echo 📋 检查环境...

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Python未安装，请先安装Python
    pause
    exit /b 1
)

where java >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Java未安装，请先安装JDK
    pause
    exit /b 1
)

where mvn >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Maven未安装，请先安装Maven
    pause
    exit /b 1
)

echo.
echo 🚀 启动AI服务 (Flask)...
cd ai-service

if not exist venv (
    echo 创建Python虚拟环境...
    python -m venv venv
)

call venv\Scripts\activate
pip install -q -r requirements.txt

echo ✅ AI服务启动中 (http://localhost:5000)...
start /B python flask_app.py
cd ..

timeout /t 3 /nobreak >nul

echo.
echo 🚀 启动Spring Boot后端...
cd backend
echo ✅ 后端服务启动中 (http://localhost:8080/api)...
start /B mvn spring-boot:run
cd ..

timeout /t 5 /nobreak >nul

echo.
echo 🚀 启动前端服务...
cd frontend
echo ✅ 前端服务启动中 (http://localhost:3000)...
start /B python -m http.server 3000
cd ..

echo.
echo ========================================
echo ✅ 所有服务启动完成！
echo ========================================
echo 📍 访问地址：
echo    前端: http://localhost:3000
echo    后端: http://localhost:8080/api
echo    AI服务: http://localhost:5000
echo.
echo 💡 关闭此窗口将停止所有服务
echo ========================================

pause

