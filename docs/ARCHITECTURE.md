# 金融知识库（finance_knowledge_base）架构设计文档

> 对齐 `knowledge_base_0525`（商品知识库）的架构与实现模式，
> 依据《金融知识库项目实战需求说明》构建的金融领域 RAG 系统。

---

## 1. 工程目录树

```
finance_knowledge_base/
├── .env.example                    # 环境变量模板
├── requirements.txt                # Python 依赖
├── README.md                       # 启动说明
├── config/
│   └── __init__.py
├── docs/
│   └── ARCHITECTURE.md             # 本文档
└── knowledge/
    ├── __init__.py
    ├── api/                        # ===== Web 层（FastAPI）=====
    │   ├── import_router.py        #   导入服务（端口8000）：/import /upload /status
    │   └── query_router.py         #   问答服务（端口8001）：/chat /query /stream /history
    ├── core/                       # ===== 核心支撑 =====
    │   ├── paths.py                #   本地临时目录/前端目录常量
    │   └── deps.py                 #   服务单例（DI 缓存）
    ├── front/                      # ===== 前端静态页 =====
    │   ├── import.html             #   导入页：上传 + 任务进度轮询
    │   └── chat.html               #   问答页：流式/引用来源/多轮/历史
    ├── processor/                  # ===== 业务处理层（双 LangGraph pipeline）=====
    │   ├── import_process/         #   ---- 导入 pipeline：摄取/解析/切片/索引 ----
    │   │   ├── base.py             #     节点基类（日志/任务追踪/异常包装）
    │   │   ├── config.py           #     ImportConfig（环境变量懒加载）
    │   │   ├── exceptions.py       #     分层异常体系
    │   │   ├── state.py            #     ImportGraphState（TypedDict）
    │   │   ├── main_graph.py       #     LangGraph 主图 + 路由
    │   │   └── nodes/
    │   │       ├── entry.py                # 入口：文件类型识别
    │   │       ├── pdf_to_md.py            # PDF -> Markdown（MinerU）
    │   │       ├── md_img.py               # MD 图片摘要/上传/引用替换
    │   │       ├── document_split.py       # Markdown 标题层级切块（长切短合/表格降维）
    │   │       ├── finance_meta_extract.py # LLM 抽取金融元数据 + 实体（写实体集合）
    │   │       ├── bge_embedding.py        # BGE-M3 稠密+稀疏 向量化
    │   │       └── import_milvus.py        # 写入 Milvus 切片集合（schema/索引/插入）
    │   └── query_process/          #   ---- 查询 pipeline：检索/增强/生成 ----
    │       ├── base.py             #     节点基类（+SSE 进度推送）
    │       ├── config.py           #     QueryConfig
    │       ├── exceptions.py       #     分层异常体系
    │       ├── state.py            #     QueryGraphState
    │       ├── main_graph.py       #     LangGraph 主图 + 并行检索
    │       └── nodes/
    │           ├── entity_confirm.py      # 意图/实体识别 + 向量对齐确认
    │           ├── vector_search.py       # 向量混合检索（dense+sparse）
    │           ├── hyde_search.py         # HyDE 假设文档检索
    │           ├── web_search_mcp.py      # 可选联网检索（MCP）
    │           ├── rrf.py                 # 倒序融合（RRF）
    │           ├── rerank.py              # BGE-Reranker 精排 + 断崖截断
    │           └── answer_output.py       # 证据组装 + 金融合规答案 + 引用
    ├── prompt/                     # ===== 提示词 =====
    │   ├── import_prompt.py        #   元数据/实体抽取模板
    │   └── query_prompt.py         #   实体提取/改写、HyDE、答案（合规约束）
    ├── schema/                     # ===== 领域与接口模型 =====
    │   ├── finance_meta.py         #   金融内容类型/标量字段/实体字段定义
    │   ├── query_schema.py         #   问答请求/响应/历史模型
    │   └── upload_schema.py        #   上传/任务状态模型
    ├── services/                   # ===== 服务层 =====
    │   ├── file_import_service.py  #   上传(本地+MinIO) + 后台启动导入图
    │   └── query_service.py        #   问答调度 + 会话/任务 + 历史
    └── utils/                      # ===== 工具层 =====
        ├── client/                 #   客户端管理器（线程安全懒加载单例）
        │   ├── base.py             #     双重检查锁模板
        │   ├── ai_clients.py       #     OpenAI LLM/VLM、BGE-M3、Reranker
        │   └── storage_clients.py  #     Milvus、MongoDB、MinIO
        ├── embedding_util.py       #   BGE-M3 混合向量封装（queries/documents）
        ├── milvus_util.py          #   混合搜索请求/执行/过滤表达式
        ├── markdown_util.py        #   表格降维（HTML/Markdown 表格 -> 文本）
        ├── mongo_history_util.py   #   对话历史读写
        ├── sse_util.py             #   SSE 队列/打包/生成器
        └── task_util.py            #   任务状态/节点进度/耗时（内存）
```

---

## 2. 技术选型（与 0525 对齐）

| 能力 | 选型 | 说明 |
|---|---|---|
| 工作流编排 | LangGraph StateGraph | 导入/查询两条有向图，节点可测、可观察 |
| Web 框架 | FastAPI + Uvicorn | 两个独立服务（8000 导入 / 8001 问答），CORS 全开 |
| PDF 解析 | MinerU（本地 CLI） | PDF -> Markdown 高保真转换 |
| 切片 | 自研 Markdown 标题层级切块 | 长切短合、表格降维、标题层级上下文 |
| 嵌入模型 | BGE-M3（本地，dense+sparse） | 1024 维稠密 + 稀疏（混合检索） |
| 向量库 | Milvus | 切片集合(金融标量字段) + 实体集合(对齐) |
| 混合检索 | WeightedRanker(dense COSINE + sparse IP) | 查询时按实体过滤 |
| 精排 | BGE-Reranker-Large 交叉编码器 | 第一断崖动态截断 |
| 融合 | RRF（Reciprocal Rank Fusion） | 多路检索去噪 |
| 会话历史 | MongoDB | chat_message 集合，多轮上下文 |
| 对象存储 | MinIO | 原始文件/图片归档 |
| 流式 | SSE（progress/delta/final） | FastAPI StreamingResponse |
| 任务进度 | 内存字典 + 轮询 | task_id 维度，导入/查询通用 |

---

## 3. 领域数据模型

### 3.1 金融元数据字段（对齐需求文档 §5）

定义于 `schema/finance_meta.py`，导入与查询共用：

| 字段 | 含义 | 示例 |
|---|---|---|
| content | 内容正文（切片） | 本基金为债券型基金… |
| content_type | 内容类型（14 类枚举） | 基金产品资料概要 |
| document_title | 资料名称 | XX 纯债债券型证券投资基金产品资料概要 |
| product_name | 产品名称 | XX 纯债债券型证券投资基金 |
| product_code | 产品代码 | 000001 |
| institution_name | 机构名称 | XX 基金管理有限公司 |
| risk_level | 风险等级 | R2 |
| industry | 行业分类 | 债券市场 |
| market | 市场类型 | 债券市场 |
| publish_date | 发布时间 | 2024-03-01 |
| entry_name | 条目名称 | 风险等级说明 |
| entity_name | 金融实体（检索过滤键） | 主实体（产品名/机构名/术语） |
| source_file | 来源文件名 | XX招募说明书.pdf |
| source_path | 来源路径/链接 | /origin_files/20240301/xx.pdf |

### 3.2 实体字段（实体集合 `fin_entities_v1`）

| 字段 | 说明 |
|---|---|
| entity_name | 实体名（产品/机构/概念） |
| entity_type | product / institution / concept / announcement |
| product_code / institution_name / risk_level / content_type / document_title | 冗余展示字段 |
| dense_vector / sparse_vector | BGE-M3 混合向量 |

### 3.3 Milvus 切片集合 schema（`fin_chunks_v1`）

```
chunk_id(INT64 PK) dense_vector(1024) sparse_vector(SPARSE)
+ content/title/parent_title/content_type/document_title/product_name/
  product_code/institution_name/risk_level/industry/market/publish_date/
  entry_name/entity_name/source_file/source_path (VARCHAR)
enable_dynamic_field=True
索引：dense AUTOINDEX/COSINE；sparse SPARSE_INVERTED_INDEX/IP
```

---

## 4. 导入 pipeline（摄取与索引）

### 4.1 图结构

```
 entry_node ─(条件路由)─> pdf_to_md_node ─> md_img_node ─┐
      │  (file 类型识别)   (MinerU 转换)   (图片处理)    │
      └─ .md 直接 ────────────────────────────────────────┤
         └─ 其他 -> END                                   ▼
                                              document_split_node
                                                    │  (标题层级切块)
                                                    ▼
                                         finance_meta_extract_node
                                                    │  (元数据+实体)
                                                    ▼
                                            bge_embedding_node
                                                    │  (dense+sparse)
                                                    ▼
                                            import_milvus_node ─> END
```

### 4.2 节点职责与实现逻辑

| 节点 | 输入 -> 输出 | 核心逻辑 | 失败策略 |
|---|---|---|---|
| **entry_node** | import_file_path -> 类型标志+file_title | 后缀识别 .pdf/.md，其余抛 ValidationError | 抛错结束 |
| **pdf_to_md_node** | pdf_path -> md_path | 调 mineru CLI，输出 `<stem>/hybrid_auto/<stem>.md` | 非 0 抛 PdfConversionError |
| **md_img_node** | md_path -> md_content(替换后) | 扫描 images 目录→贪心取图上下文→VLM 生成摘要→MinIO 上传→正则替换 `![摘要](URL)` | 分步降级（无 images 目录/无 VLM/无 MinIO 均继续） |
| **document_split_node** | md_content -> chunks | ① 正则解析 `#{1,6}` 标题，用 7 层 hierarchy 维护标题栈，flush 成 section（含 parent_title）；② 超长章节按 `\n\n/。/！/？` 等 RecursiveCharacterTextSplitter 二次切分并标 `-N`；③ 同父标题的短章节（<min_content_length）合并；④ 表格先 `MarkdownTableLinearizer` 降维；⑤ content = title+body；⑥ 落盘 chunks.json | 参数校验抛错；备份失败仅告警 |
| **finance_meta_extract_node** | chunks -> finance_meta+entities+回填 chunk | 取前 k 个切片作上下文 → LLM(JSON 模式)一次抽取 `finance_meta` 与 `entities`；对每个 entity_name 做 BGE-M3 向量化并写实体集合（不存在则建）；元数据回填全部 chunk；落盘 chunks_meta_entities.json | LLM 失败返回空元数据不阻断；单实体入库失败仅告警 |
| **bge_embedding_node** | chunks(带元数据) -> chunks(带向量) | 嵌入文本=`{entity_name}\n{content}`（实体名参与语义匹配）；按 batch=16 分批；dense 取 list、sparse 从 CSR 矩阵解构为 {token_id:weight} | 单批失败保留原 chunk（由入库过滤） |
| **import_milvus_node** | chunks(带向量) -> chunks(带 chunk_id) | 过滤无向量 chunk → 确保集合存在（SchemaBuilder/IndexBuilder）→ `_MilvusInserter.insert` → 回填 chunk_id | 客户端为空安全返回 |

> 需求对应：切片粒度覆盖"条款级问答"与"章节级总结"，14 类内容类型统一入库；
> 实体集合解决了"产品名/机构名/术语"的查询对齐问题（等同 0525 的 item_name 集合）。

---

## 5. 查询 pipeline（检索增强生成）

### 5.1 图结构

```
                    +-------------------+
                    | __start__         |
                    +-------------------+
                              |
                              v
                    +-------------------+
                    |  entity_confirm   | 意图/实体提取 -> 向量对齐 -> 决策
                    +-------------------+
                         |                \
            (无answer)    |                 \ (有answer: 澄清/无法识别 -> 直接出答案)
                         v                  v
                    +-------------+    +----------------+
                    | multi_search|    | answer_output  |
                    +-------------+    +----------------+
                    /      |       \         | (citation/历史)
                   v       v        v        v
        +------------+ +------------+ +------------+     __end__
        |search_embed| |search_embed| |web_search_ |
        |   ding     | |  ding_hyde | |    mcp     |
        +------------+ +------------+ +------------+
                   \       |        /
                    v      v       v
                    +-------------+
                    |    join     |
                    +-------------+
                          |
                          v
                    +-------------+
                    |    rrf      | 倒序融合
                    +-------------+
                          |
                          v
                    +-------------+
                    |   rerank    | BGE-Reranker 精排+断崖截断
                    +-------------+
                          |
                          v
                    +-----------------+
                    |  answer_output  | 组装证据/合规生成/引用/历史
                    +-----------------+
                          |
                          v
                       __end__
```

### 5.2 节点职责与实现逻辑

| 节点 | 输入 -> 输出 | 核心逻辑 | 金融适配点 |
|---|---|---|---|
| **entity_confirm** | original_query+history -> query_type/entity_names/rewritten_query 或 answer | ① `EntityExtractor`：LLM 读「历史+当前问题」，输出 query_type、entity_names、rewritten_query（JSON）；② `EntityAligner`：抽取实体 BGE-M3 向量化，与实体集合 WeightedRanker 混合检索（0.5/0.5，norm），0.7+ 高置信确认、0.6+ 进候选，多个 confirmed 时按 0.15 断崖收口；③ `_decide` 五分支决策（见下） | 相比 0525：query_type=general（金融知识/FAQ/业务问题）**不打断用户**，降级全库通用检索，满足需求 §2.3/§2.7 |
| **vector_search** | rewritten_query -> embedding_chunks | BGE-M3 encode_queries → dense+sparse 两个 AnnSearchRequest → Milvus hybrid_search(WeightedRanker 0.5/0.5) → 输出 14 个金融字段 | entity_names 非空拼 `entity_name in {...}` 过滤；空则全库 |
| **hyde_search** | rewritten_query -> hyde_embedding_chunks | LLM 生成假设性金融文档（150-300字）→ `问题\n假设文档` 一起向量化检索 | 缓解口语（"到账"）与文档（"赎回款项划拨"）鸿沟 |
| **web_search_mcp** | rewritten_query -> web_search_docs | 通过 DashScope MCP WebSearch 检索 3 条 | 无配置/失败静默降级为空（本地为主，实时信息可选） |
| **rrf** | 两路chunks -> rrf_chunks | 统一剥壳取 entity 字段，按 `weight/(k+rank)` 累加，k=60，取 top10 | chunk_id 去重键 |
| **rerank** | rrf_chunks+web -> reranked_docs | BGE-Reranker `compute_score(normalize)` 打分排序；`cliff_cutoff`：扫描首个相邻 gap>0.15 处截断，且截断点 ∈ [min_top_k=3, max_top_k=10] | 本地 doc 保留引用元数据；web doc 标 source=web |
| **answer_output** | reranked_docs+history -> answer+citations+历史 | ① 证据带 `[N][资料=][产品=][风险=]...` 编号格式化 + 历史格式，受 12000 字符预算；② 调用 LLM(流式/非流式)；③ 收集 citations 列表（去重）；④ user/assistant 双写 Mongo | 无证据直接返回需求 §6.3 话术；流式把增量 push SSE |

### 5.3 entity_confirm 决策分支

| 分支 | 条件 | 行为 |
|---|---|---|
| 1 | confirmed 非空 | 带实体过滤走三路检索（产品/公告类问答） |
| 2 | query_type=general 且允许 | 全库通用检索（知识/术语/FAQ/业务问题） |
| 3 | query_type=industry/announcement 且无候选 | 主题全库检索（资讯/政策/行业观点） |
| 4 | options 候选非空 | 返回澄清话术（"您是指…吗"）→ 短路 answer |
| 5 | 其余（product 型但无匹配） | 礼貌提示无法识别 → 短路 answer |

> 这是相对 0525 的关键差异：商品域"无商品即打断"，金融域对知识/FAQ/流程类
> 问题必须能继续回答（需求 §2.3/§2.7 示例问题均无具体产品实体）。

---

## 6. 模块间协同调用关系

### 6.1 导入链（一次上传）

```
浏览器(import.html)
  │ POST /upload (multipart 文件)
  ▼
FastAPI import_router(8000)
  │ service.upload_file()  → 本地 temp_data/{task_id}/ + MinIO origin_files/归档
  │ 返回 {task_id}  ← 前端立即拿到
  │ background_tasks.add_task(run_import_graph)
  ▼
ImportFileService.run_import_graph
  │ update_task_status(processing)
  ▼
LangGraph: entry → (pdf_to_md → md_img)? → document_split
         → finance_meta_extract → bge_embedding → import_milvus
  │ 每节点 BaseNode.__call__ → add_running/add_done + add_node_duration
  ▼
update_task_status(completed/failed)

前端 import.html：轮询 GET /status/{task_id}
  → {status, running_list, done_list, durations} 渲染步骤条
```

### 6.2 问答链（一次提问）

```
浏览器(chat.html)
  │ POST /query {query, session_id, is_stream}
  ▼
FastAPI query_router(8001)
  │ is_stream=true:  create_sse_queue(task_id) + 后台 run_query_graph
  │                 返回 {task_id}
  │ is_stream=false: run_in_executor(线程池) 同步执行，返回 {answer, citations}
  ▼
QueryService.run_query_graph → LangGraph invoke(entity_confirm → multi_search
      → [vector|hyde|web] ∥ → join → rrf → rerank → answer_output)
  │ 各节点在 BaseNode.__call__ 中（is_stream）→ push_sse_event(PROGRESS)
  │ answer_output 流式生成 → push_sse_event(DELTA, {delta})
  │                 结束后 → push_sse_event(FINAL, {answer, citations})
  │ user/assistant 消息双写 Mongo chat_message（供多轮追问/历史）
  ▼
浏览器：GET /stream/{task_id}（EventSource）
  → progress 事件更新步骤条；delta 事件追加文本；final 事件收尾 + 渲染引用
```

### 6.3 多轮追问如何工作（需求 §3.6）

1. `entity_confirm` 节点读取 Mongo 最近 10 条 `get_recent_messages(session_id)` 反转为时间正序；
2. 拼入 ENTITY_EXTRACT_TEMPLATE 的 history_text；
3. LLM 对代词（"它/这个产品"）做指代消解，输出完整 entity_names 与 rewritten_query；
4. answer 的 prompt 也携带历史，保证追问上下文连续。

---

## 7. 与 0525 工程映射对照表

| 0525（商品知识库） | 金融知识库（本项目） | 变更说明 |
|---|---|---|
| knowledge_base_0525/knowledge | finance_knowledge_base/knowledge | 同构包 |
| import_process/nodes/item_name_recognition | finance_meta_extract | 单一商品名 → 14 字段金融元数据 + 多实体 |
| query_process/nodes/item_name_confirm | entity_confirm | 增加 query_type 意图分类 + general 全库检索分支 |
| item_name_collection(kb_item_names_v1) | entity_collection(fin_entities_v1) | 实体集合，多字段冗余 |
| chunks 字段 item_name | entity_name + 金融标量字段 | 过滤键升级为可多条件过滤 |
| 切片按文档技术手册标题 | 相同切块器 + 表格降维 | 适配费率表/要素表 |
| ANSWER_PROMPT | 金融合规 ANSWER_PROMPT | §6.3 话术/§7 合规/§6.2 引用 [N] |
| task_util 节点中文映射 | 新增 finance 节点中文名 | 前端展示 |
| chat.html/import.html | 金融主题化 + 引用来源区块 | §6.2 引用来源可见 |

---

## 8. 需求文档合规实现对照

| 需求条款 | 实现 |
|---|---|
| §6.1 回答结构清晰 | prompt 要求结论/内容/风险/引用分层 |
| §6.2 引用来源 | reranked 证据带编号与元数据 → LLM 用 [N] 标注 → citations 结构化回传前端 |
| §6.3 无资料话术 | reranked_docs 为空 → 固定返回"当前知识库中未检索到足够信息…" |
| §7.1 不提供投资建议 | prompt 硬约束 + 实体集合只做资料索引 |
| §7.2 不承诺收益 | prompt 禁用词表（一定赚钱/保证收益/保本无风险…） |
| §7.3 风险提示谨慎 | prompt 强制风险声明模板 |
| §7.4 不保证实时性 | web 结果标 source=web；prompt 提示以官方渠道为准 |
| §8 多轮/流式/历史/重问 | Mongo 历史 + SSE + /history + 新会话按钮 |
| §9 导入失败不影响整体 | 每节点降级策略 + 任务状态 failed 不挂服务 |
| §10 扩展方向 | 集合预留 dynamic_field，实体/字段可增量扩展 |

---

## 9. 启动与验证步骤

1. `cp .env.example .env`，填入 LLM/模型路径/Milvus/Mongo/MinIO 配置；
2. `pip install -r requirements.txt`；单独安装 mineru（PDF 转 MD）；
3. `python -m knowledge.api.import_router`（8000）、`python -m knowledge.api.query_router`（8001）；
4. 浏览器开 `/import` 上传金融文档 → 观察切块/元数据/入库进度；
5. 开 `/chat` 提问：产品类（"XX 基金风险等级"）、知识类（"什么是净值型理财"）、
   业务类（"赎回多久到账"）、公告类，验证引用来源与风险话术。
