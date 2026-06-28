# RAG 智能问答系统 v2.0

本项目只是一次对RAG的简单尝试，有些地方并未优化好，仅作为一次备份提交至仓库。提交的文件中缺少了.env环境文件与数据文档文件，需要自行添加才可运行。

基于 **Milvus + DashScope + DeepSeek** 的检索增强生成（RAG）系统，支持多轮对话、Rerank 重排序、多路召回。

## 核心功能

| 功能 | 说明 |
|------|------|
| 多格式文档加载 | PDF、Word (.docx)、TXT、Markdown |
| 智能文本切割 | 中英文递归分割，512 字符/块，128 字符重叠 |
| 向量嵌入 | DashScope text-embedding-v4（1024 维） |
| 向量存储 | Milvus 向量数据库 |
| 多路召回 | 向量检索 + BM25 关键词检索，RRF 融合 |
| Rerank 重排序 | DashScope gte-rerank-v2 精排 |
| 多轮对话 | 带记忆的连续对话，滑动窗口 10 轮 |

## 技术栈

| 组件 | 技术 |
|------|------|
| 向量数据库 | [Milvus](https://milvus.io/) (Docker) |
| 嵌入模型 | [DashScope text-embedding-v4](https://dashscope.aliyun.com/) |
| 重排序 | [DashScope gte-rerank-v2](https://dashscope.aliyun.com/) |
| 关键词检索 | [jieba](https://github.com/fxsjy/jieba) + [rank_bm25](https://github.com/dorianbrown/rank_bm25) |
| 大语言模型 | [DeepSeek-v4-pro](https://platform.deepseek.com/) |
| 编排 | [LangChain](https://langchain.com/) |

## 使用方法

```bash
cd RAG

# ── 启动 Milvus ──
docker compose up -d

# ── 摄入文档（把文件扔进 documents/ 文件夹）──
uv run python main.py ingest

# ── 单次问答 ──
uv run python main.py query '转专业成绩要求'

# ── 多轮对话 ──
uv run python main.py chat

# ── 查看历史对话 ──
uv run python main.py history


# ── 查看系统状态 ──
uv run python main.py status
```

## 项目结构

```
RAG/
├── main.py              # CLI 入口
├── data_processor.py    # 数据处理模块
├── query.py             # 检索问答
├── config.py            # 配置管理
├── documents/           # 放入文件即可摄入
├── modules/
│   ├── loader.py        # 文档加载
│   ├── splitter.py      # 文本切割
│   ├── embeddings.py    # 向量嵌入
│   ├── vector_store.py  # Milvus 操作
│   ├── llm.py           # DeepSeek LLM
│   ├── reranker.py      # Rerank 重排序
│   ├── retriever.py     # 多路召回 + RRF
│   └── chat_session.py  # 对话管理
└── docker-compose.yml   # Milvus 部署
```

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        数据摄入流程                                  │
│                                                                     │
│  documents/ 文件夹 ──▶ 加载 ──▶ 清洗 ──▶ 切割 ──▶ 嵌入 ──▶ Milvus │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        问答检索流程                              │
│                                                                 │
│  用户问题 ──┬──▶ 向量检索 (Milvus) ──┐                           │
│             │                        ├──▶ RRF 融合 ──▶ Rerank  │
│             └──▶ BM25 关键词检索 ────┘              │           │
│                                                     ▼           │
│  对话历史 ──▶ 注入 Prompt ◀── 上下文 ◀── Top-N 文档             │
│                     │                                           │
│                     ▼                                           │
│             DeepSeek 生成回答                                    │
└─────────────────────────────────────────────────────────────────┘
```

## 配置说明

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `DASHSCOPE_API_KEY` | — | DashScope API 密钥|
| `DEEPSEEK_API_KEY` | — | DeepSeek API 密钥|
| `EMBEDDING_MODEL` | `text-embedding-v4` | 嵌入模型 |
| `DEEPSEEK_MODEL` | `deepseek-v4-pro` | LLM 模型 |
| `RERANK_MODEL` | `gte-rerank-v2` | Rerank 模型 |
| `RERANK_TOP_N` | `3` | Rerank 返回文档数 |
| `RETRIEVAL_TOP_K` | `8` | 每路检索返回数 |
| `CHUNK_SIZE` | `512` | 切割块大小 |
| `CHUNK_OVERLAP` | `128` | 切割块重叠 |
| `CHAT_HISTORY_LENGTH` | `10` | 对话记忆轮数 |

## 环境准备

```bash
# 安装依赖
uv add pymupdf python-docx gradio jieba rank_bm25

# 启动 Milvus
docker compose up -d

# 配置 API Keys
cp .env.example .env
# 编辑 .env 填入 DASHSCOPE_API_KEY 和 DEEPSEEK_API_KEY
```

## 停止服务

```bash
docker compose down
```
