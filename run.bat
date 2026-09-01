@echo off
REM 快速启动脚本（Windows）
REM 用法：
REM   run.bat                  默认摄像头
REM   run.bat --demo           使用内置演示视频
REM   run.bat --video X.mp4    指定视频
REM   run.bat --tests          运行单元测试

setlocal
cd /d "%~dp0"

REM 优先使用项目内的虚拟环境（首次运行前需先执行 setup.bat 或由 PyCharm 创建）
if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" main.py %*
) else (
    echo [提示] 未找到 venv，正在使用系统 Python ...
    python main.py %*
)
endlocal