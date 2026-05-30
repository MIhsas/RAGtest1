"""
全局配置管理

使用 pydantic-settings 从 .env 文件加载配置，支持环境变量覆盖。
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """RAG 项目全局配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── 项目路径 ──────────────────────────────────────────────
    project_root: Path = Path(__file__).parent
    documents_dir: Path = Path(__file__).parent / "documents"

    # ── DashScope Embedding ───────────────────────────────────
    dashscope_api_key: str = ""
    embedding_model: str = "text-embedding-v4"
    embedding_dim: int = 1024

    # ── DeepSeek LLM ─────────────────────────────────────────
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-pro"
    llm_temperature: float = 0.3

    # ── Milvus 向量数据库 ─────────────────────────────────────
    milvus_host: str = "127.0.0.1"
    milvus_port: int = 19530
    milvus_collection: str = "rag_documents"

    # ── 文本切割参数 ─────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 128

    # ── 检索参数 ─────────────────────────────────────────────
    retrieval_top_k: int = 8

    # ── Rerank 重排序 ─────────────────────────────────────────
    rerank_model: str = "gte-rerank"
    rerank_top_n: int = 3

    # ── BM25 关键词检索 ───────────────────────────────────────
    bm25_top_k: int = 8

    # ── 多轮对话 ─────────────────────────────────────────────
    chat_history_length: int = 10


settings = Settings()
