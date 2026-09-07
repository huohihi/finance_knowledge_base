@echo off
REM 启动金融知识库 - 问答服务 (端口 8001)
cd /d %~dp0
call .venv\Scripts\activate.bat 2>nul || echo 请先创建 .venv 并安装 requirements.txt
python -m knowledge.api.query_router
