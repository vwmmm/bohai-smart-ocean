@echo off
rem 渤海智瞳 - Windows 一键启动脚本
cd /d %~dp0
if not exist .venv (
    echo [提示] 未找到虚拟环境，正在创建...
    py -3.13 -m venv .venv
    .venv\Scripts\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
)
.venv\Scripts\python.exe -m streamlit run app.py
pause
