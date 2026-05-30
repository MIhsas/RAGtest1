# RAG 智能问答系统 v2.0

基于 **Milvus + DashScope + DeepSeek** 的检索增强生成（RAG）系统，支持多轮对话、Rerank 重排序、多路召回和 Web 前端。

## 功能概览

| 功能 | 说明 |
|------|------|
| 📄 多格式文档加载 | 支持 PDF、Word (.docx)、TXT、Markdown |
| ✂️ 智能文本切割 | 中英文友好的递归字符分割策略 |
| 🧮 向量嵌入 | DashScope text-embedding-v4 |
| 🗄️ 向量存储 | Milvus 向量数据库 |
| 🔍 多路召回 | 向量检索 + BM25 关键词检索，RRF 融合 |
| 🎯 Rerank 重排序 | DashScope gte-rerank 精排 |
| 💬 多轮对话 | 带记忆的连续对话，滑动窗口管理 |
| 🌐 Web 前端 | Gradio 构建的交互式界面，支持文件上传 |

## 系统架构

```
┌────────────────────────────────────────────────────────────────┐
│                        数据摄入流程                             │
│                                                                │
│  documents/ 文件夹 ──▶ 加载 ──▶ 清洗 ──▶ 切割 ──▶ 嵌入 ──▶ Milvus │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│                        问答检索流程                             │
│                                                                │
│  用户问题 ──┬──▶ 向量检索 (Milvus) ──┐                         │
│             │                        ├──▶ RRF 融合 ──▶ Rerank  │
│             └──▶ BM25 关键词检索 ────┘              │          │
│                                                     ▼          │
│  对话历史 ──▶ 注入 Prompt ◀── 上下文压缩 ◀── Top-N 文档        │
│                     │                                          │
│                     ▼                                          │
│             DeepSeek 生成回答                                   │
│                     │                                          │
│                     ▼                                          │
│             更新对话记忆 ──▶ 返回回答                           │
└────────────────────────────────────────────────────────────────┘
```

## 项目结构

```
RAG/
├── .env.example              # 环境变量模板
├── config.py                 # 全局配置
├── docker-compose.yml        # Milvus Docker 部署
├── requirements.txt          # Python 依赖
├── main.py                   # 主入口 CLI
├── data_processor.py         # 独立数据处理模块
├── ingest.py                 # 数据摄入入口（调用 data_processor）
├── query.py                  # 检索问答（多轮对话 + 多路召回）
├── web_ui.py                 # Gradio Web 前端
├── modules/
│   ├── __init__.py
│   ├── loader.py             # 多格式文档加载器
│   ├── splitter.py           # 文本智能切割
│   ├── embeddings.py         # DashScope Embedding
│   ├── vector_store.py       # Milvus 向量数据库
│   ├── llm.py                # DeepSeek LLM
│   ├── reranker.py           # DashScope Rerank 重排序（新增）
│   ├── chat_session.py       # 多轮对话管理（新增）
│   └── retriever.py          # 多路召回 + RRF 融合（新增）
├── documents/                # 📂 放入文件即可自动摄入
└── README.md
```

## 环境准备

### 1. 安装依赖

```bash
cd RAG
uv add pymupdf python-docx gradio jieba rank_bm25
```

### 2. 启动 Milvus

```bash
docker compose up -d
```

### 3. 配置 API Keys

```bash
cp .env.example .env
```

编辑 `.env`：

```env
DASHSCOPE_API_KEY=sk-你的dashscope密钥
DEEPSEEK_API_KEY=sk-你的deepseek密钥
```

## 使用方法

### 快速开始

```bash
# 1. 将文件放入 documents/ 文件夹
# 2. 摄入文档
uv run python main.py ingest
# 3. 启动 Web 界面
uv run python main.py web
```

### CLI 命令

```bash
# 摄入文档（默认扫描 documents/）
uv run python main.py ingest

# 单次问答
uv run python main.py query '领克汽车质保年限'

# 多轮对话（带记忆）
uv run python main.py chat

# 启动 Web 前端
uv run python main.py web

# 查看系统状态
uv run python main.py status
```

### Web 前端

启动后访问 http://127.0.0.1:7860，支持：
- 💬 多轮对话聊天
- 📤 拖拽上传文件自动摄入
- 📊 查看系统状态

## 配置说明

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `DASHSCOPE_API_KEY` | — | DashScope API 密钥（必填） |
| `DEEPSEEK_API_KEY` | — | DeepSeek API 密钥（必填） |
| `EMBEDDING_MODEL` | `text-embedding-v4` | 嵌入模型 |
| `DEEPSEEK_MODEL` | `deepseek-v4-pro` | LLM 模型 |
| `RERANK_MODEL` | `gte-rerank` | Rerank 模型 |
| `RERANK_TOP_N` | `3` | Rerank 返回文档数 |
| `RETRIEVAL_TOP_K` | `8` | 每路检索返回数 |
| `BM25_TOP_K` | `8` | BM25 检索返回数 |
| `CHUNK_SIZE` | `512` | 切割块大小 |
| `CHUNK_OVERLAP` | `128` | 切割块重叠 |
| `CHAT_HISTORY_LENGTH` | `10` | 对话记忆轮数 |

## 技术栈

| 组件 | 技术 |
|------|------|
| 向量数据库 | [Milvus](https://milvus.io/) |
| 文本嵌入 | [DashScope text-embedding-v4](https://dashscope.aliyun.com/) |
| 重排序 | [DashScope gte-rerank](https://dashscope.aliyun.com/) |
| 关键词检索 | [jieba](https://github.com/fxsjy/jieba) + [rank_bm25](https://github.com/dorianbrown/rank_bm25) |
| 大语言模型 | [DeepSeek](https://platform.deepseek.com/) |
| 前端界面 | [Gradio](https://www.gradio.app/) |
| 编排框架 | [LangChain](https://langchain.com/) |

## 停止服务

```bash
docker compose down
```
