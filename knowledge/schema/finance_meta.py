"""
金融领域公共常量与元数据字段定义

对齐需求文档《第 5 节 内容字段要求》，作为
导入元数据抽取、Milvus 集合 schema、查询过滤、引用来源回显的统一依据。
"""

# ==================== 内容类型枚举 ====================
# 需求文档 4 内容导入范围：产品资料/说明书/风险揭示书/招募说明书/资料概要/
# 理财说明书/资讯/行业分析/公告摘要/政策解读/术语/风险提示/FAQ/流程说明
CONTENT_TYPE_FINANCIAL_PRODUCT = "金融产品资料"
CONTENT_TYPE_PRODUCT_SPEC = "产品说明书"
CONTENT_TYPE_RISK_REVEAL = "风险揭示书"
CONTENT_TYPE_FUND_PROSPECTUS = "基金招募说明书"
CONTENT_TYPE_FUND_SNAPSHOT = "基金产品资料概要"
CONTENT_TYPE_WM_SPEC = "理财产品说明书"
CONTENT_TYPE_MARKET_NEWS = "市场资讯"
CONTENT_TYPE_INDUSTRY_ANALYSIS = "行业分析"
CONTENT_TYPE_ANNOUNCEMENT = "公告摘要"
CONTENT_TYPE_POLICY = "政策解读"
CONTENT_TYPE_TERM = "金融术语"
CONTENT_TYPE_RISK_TIPS = "风险提示"
CONTENT_TYPE_FAQ = "FAQ"
CONTENT_TYPE_PROCESS = "业务流程说明"

CONTENT_TYPES: list = [
    CONTENT_TYPE_FINANCIAL_PRODUCT,
    CONTENT_TYPE_PRODUCT_SPEC,
    CONTENT_TYPE_RISK_REVEAL,
    CONTENT_TYPE_FUND_PROSPECTUS,
    CONTENT_TYPE_FUND_SNAPSHOT,
    CONTENT_TYPE_WM_SPEC,
    CONTENT_TYPE_MARKET_NEWS,
    CONTENT_TYPE_INDUSTRY_ANALYSIS,
    CONTENT_TYPE_ANNOUNCEMENT,
    CONTENT_TYPE_POLICY,
    CONTENT_TYPE_TERM,
    CONTENT_TYPE_RISK_TIPS,
    CONTENT_TYPE_FAQ,
    CONTENT_TYPE_PROCESS,
]

# ==================== Milvus 标量字段（chunk 集合） ====================
# 需求文档 5 内容字段要求：正文 + 内容类型 + 资料名称 + 产品名称 + 产品代码 +
# 机构名称 + 风险等级 + 行业分类 + 市场类型 + 发布时间 + 条目名称 + 来源文件 + 来源路径
#
# 字段说明：
#   content           内容正文（切分后的文本）
#   title             条目/章节标题（切分得到的标题）
#   parent_title      父章节标题（层级上下文）
#   content_type      内容类型（上面 14 类之一，用于用户按资料类型检索/过滤）
#   document_title    资料名称（原文档标题）
#   product_name      产品名称（如：某某纯债债券型证券投资基金）
#   product_code      产品代码（如：000001）
#   institution_name  机构名称（发行/管理人，如：某某基金管理有限公司）
#   risk_level        风险等级（R1-R5 / 低-高）
#   industry          行业分类（如：消费/医药/新能源）
#   market            市场类型（A股/港股/债市/货币 等）
#   publish_date      发布时间
#   entry_name        条目名称（抽取摘要的条目，如某 FAQ 的问题）
#   entity_name       金融实体（统一检索键：产品名/机构名/概念名，等同0525的item_name）
#   source_file       来源文件名
#   source_path       来源路径/资源链接
#   file_title        文件标题（不含扩展名，内部使用）
FINANCE_SCALAR_FIELDS: list = [
    # —— 结构化标量（对齐需求文档字段）——
    "content",
    "title",
    "parent_title",
    "content_type",
    "document_title",
    "product_name",
    "product_code",
    "institution_name",
    "risk_level",
    "industry",
    "market",
    "publish_date",
    "entry_name",
    "entity_name",
    "source_file",
    "source_path",
]

# ==================== 实体集合字段 ====================
# 实体集合用于查询链路中"用户问题 -> 金融实体(产品/机构/概念)对齐确认"，
# 等价于 0525 工程中的 item_name_collection。
ENTITY_FIELDS: list = [
    "entity_name",          # 实体名（产品全名/机构名/概念名）
    "entity_type",          # product / institution / concept / announcement
    "product_code",         # 产品代码（若有）
    "institution_name",     # 机构名称（若有）
    "risk_level",           # 风险等级（若有，便于展示）
    "content_type",         # 内容类型
    "document_title",       # 资料名称（来源文档）
    "source_file",          # 来源文件名
]

# 实体类型枚举
ENTITY_TYPE_PRODUCT = "product"
ENTITY_TYPE_INSTITUTION = "institution"
ENTITY_TYPE_CONCEPT = "concept"
ENTITY_TYPE_ANNOUNCEMENT = "announcement"
