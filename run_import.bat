@echo off
REM 启动金融知识库 - 导入服务 (端口 8000)
cd /d %~dp0
call .venv\Scripts\activate.bat 2>nul || echo 请先创建 .venv 并安装 requirements.txt
python -m knowledge.api.import_router
