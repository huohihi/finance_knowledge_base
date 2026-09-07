# 金融知识库（finance_knowledge_base）

基于 LangGraph + FastAPI + Milvus(BGE-M3 混合检索) 的金融领域 RAG 知识库系统。
架构与实现模式对齐 `knowledge_base_0525`（商品知识库）工程，字段与合规逻辑适配金融场景。

## 能力覆盖

| 需求文档章节 | 功能 | 实现落点 |
|---|---|---|
| §2.1 金融资料查询 | 产品说明书/公告/研报/FAQ/术语 等 14 类内容导入 | import_process 全链路 |
| §2.2 金融产品问答 | 产品名/代码/风险等级/申赎规则问答 | entity_confirm + vector/hyde 检索 + answer |
| §2.3 金融知识问答 | 术语、概念通俗解释 | query_type=general 全库检索路径 |
| §2.4/2.5 资讯/公告 | 资讯与公告总结、不编造 | reranked_docs 证据约束 + 兜底话术 |
| §2.6 风险提示 | R1-R5 解释、风险谨慎表达 | ANSWER_PROMPT 合规约束 |
| §2.7 业务规则 | FAQ/流程问答 | general 检索 + FAQ 切片 |
| §5 内容字段 | 14+ 个结构化字段 | schema/finance_meta.py + Milvus 标量字段 |
| §6 回答要求 | 结构清晰/引用来源/无资料话术 | citations + 证据 [N] 编号 + 兜底 |
| §7 合规边界 | 不荐股/不承诺收益/谨慎风险表达 | 提示词硬约束 |
| §8 交互 | 单轮/多轮/流式/历史/引用/重问 | chat.html + SSE + Mongo |

## 快速开始

```bash
# 1. 安装依赖（建议独立 venv）
pip install -r requirements.txt
# MinerU PDF 解析为独立 CLI，请按官方文档单独安装（仅 PDF 导入需要）

# 2. 配置环境变量
cp .env.example .env   # 填写 LLM/BGE-M3/Milvus/Mongo/MinIO/MCP 地址与密钥

# 3. 启动导入服务（端口 8000，含前端 /import 页）
python -m knowledge.api.import_router

# 4. 启动问答服务（端口 8001，含前端 /chat 页）
python -m knowledge.api.query_router

# 5. 浏览器访问
#    导入页 http://localhost:8000/import
#    问答页 http://localhost:8001/chat
```

## 工程结构 / 架构设计

见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) —— 含完整目录树、
双 LangGraph pipeline 图、节点职责表、与 0525 的映射对照、数据字段定义。

## 目录速览

```
finance_knowledge_base/
├── .env.example
├── requirements.txt
├── config/
├── docs/ARCHITECTURE.md
└── knowledge/
    ├── api/            FastAPI 路由（导入 8000 / 问答 8001）
    ├── core/           路径常量、依赖注入
    ├── front/          前端页面（import.html / chat.html）
    ├── processor/
    │   ├── import_process/    金融文档摄取与索引 pipeline（LangGraph）
    │   └── query_process/     检索增强生成问答 pipeline（LangGraph）
    ├── prompt/         导入/查询提示词模板（含金融合规约束）
    ├── schema/         领域字段与请求/响应模型
    ├── services/       业务服务（上传导入 / 问答调度）
    └── utils/          客户端管理器、混合检索、SSE、任务追踪等工具
```
<img width="1912" height="1068" alt="fa77c4bb-04ff-4fae-9033-087b9467060d" src="https://github.com/user-attachments/assets/cf04f453-b811-49b0-a1c7-b51b01a28790" />
<img width="1912" height="1068" alt="c6aaacef-1efa-4f77-a736-4082b475dc01" src="https://github.com/user-attachments/assets/bf3eccce-50c1-41c4-a6b1-a2472e3e35e3" />
