@echo off
REM 一键初始化环境脚本（Windows）
REM 作用：创建项目内虚拟环境 venv 并安装全部依赖。
REM 首次在 PyCharm 打开前运行一次即可；之后双击 run.bat 直接启动。

setlocal
cd /d "%~dp0"

REM 优先查找可用的 Python 3.8+ 解释器
set "PY="

REM 1) 项目内已有 venv，直接复用
if exist "venv\Scripts\python.exe" set "PY=venv\Scripts\python.exe"

REM 2) conda 环境 labelimg (Python 3.9)
if "%PY%"=="" if exist "D:\Anaconda3\envs\labelimg\python.exe" set "PY=D:\Anaconda3\envs\labelimg\python.exe"
if "%PY%"=="" if exist "D:\Anaconda3\envs\labelme\python.exe" set "PY=D:\Anaconda3\envs\labelme\python.exe"
if "%PY%"=="" if exist "D:\Anaconda3\envs\belt-project\python.exe" set "PY=D:\Anaconda3\envs\belt-project\python.exe"

REM 3) 回退到系统 python（可能版本过低，会给出提示）
if "%PY%"=="" set "PY=python"

echo 使用解释器: %PY%

REM 创建 venv（如果还不存在）
if not exist "venv\Scripts\python.exe" (
    echo [1/2] 正在创建虚拟环境 venv ...
    "%PY%" -m venv venv
    if errorlevel 1 (
        echo [错误] 创建虚拟环境失败，请确认已安装 Python 3.8 或更高版本。
        pause
        exit /b 1
    )
)

echo [2/2] 正在安装依赖（使用清华镜像加速）...
"venv\Scripts\python.exe" -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
"venv\Scripts\python.exe" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

if errorlevel 1 (
    echo [错误] 依赖安装失败，请检查网络后重试。
    pause
    exit /b 1
)

echo.
echo ============================================
echo  环境初始化完成！可以开始运行了：
echo    1. PyCharm 打开本项目，解释器选 venv\Scripts\python.exe
echo    2. 或直接双击 run.bat 启动
echo ============================================
pause
endlocal
