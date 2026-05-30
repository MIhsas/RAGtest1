"""
Milvus 向量数据库封装（pymilvus MilvusClient API）

提供集合管理、文档写入、相似度检索等功能。
"""

import json
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pymilvus")

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pymilvus import MilvusClient

from config import settings


class MilvusStore:
    """Milvus 向量数据库操作封装（使用 MilvusClient API）。"""

    def __init__(self, embeddings: Embeddings):
        self._embeddings = embeddings
        self._uri = f"http://{settings.milvus_host}:{settings.milvus_port}"
        self._collection = settings.milvus_collection
        self._dim = settings.embedding_dim
        self._client: MilvusClient | None = None

    def _get_client(self) -> MilvusClient:
        """获取或创建 MilvusClient 实例（延迟初始化）。"""
        if self._client is None:
            self._client = MilvusClient(uri=self._uri)
        return self._client

    def _ensure_collection(self, load: bool = True):
        """确保集合存在（不存在则创建），并加载到内存。"""
        client = self._get_client()
        collections = client.list_collections()

        if self._collection not in collections:
            from pymilvus import CollectionSchema, DataType, FieldSchema

            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self._dim),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=4096),
            ]
            schema = CollectionSchema(fields=fields, description="RAG documents")

            client.create_collection(
                collection_name=self._collection,
                schema=schema,
            )

            # 创建向量索引
            index_params = client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_type="IVF_FLAT",
                metric_type="COSINE",
                params={"nlist": 128},
            )
            client.create_index(
                collection_name=self._collection,
                index_params=index_params,
            )

        # 加载集合到内存（搜索前必须）
        if load:
            client.load_collection(collection_name=self._collection)

    def add_documents(self, documents: list[Document]) -> list[int]:
        """
        将文档写入 Milvus。

        Args:
            documents: 待写入的 Document 列表

        Returns:
            写入成功的文档 ID 列表
        """
        if not documents:
            return []

        client = self._get_client()
        self._ensure_collection()

        texts = [doc.page_content for doc in documents]
        sources = [doc.metadata.get("source", "") for doc in documents]
        metadatas = [json.dumps(doc.metadata, ensure_ascii=False) for doc in documents]

        # 批量生成嵌入
        embeddings = self._embeddings.embed_documents(texts)

        # 插入数据
        data = [
            {"text": t, "embedding": e, "source": s, "metadata": m}
            for t, e, s, m in zip(texts, embeddings, sources, metadatas)
        ]

        result = client.insert(collection_name=self._collection, data=data)
        return result.get("ids", [])

    def search(self, query: str, top_k: int | None = None) -> list[Document]:
        """
        执行相似度检索。

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            相似文档列表
        """
        k = top_k or settings.retrieval_top_k
        client = self._get_client()
        self._ensure_collection()

        query_embedding = self._embeddings.embed_query(query)

        results = client.search(
            collection_name=self._collection,
            data=[query_embedding],
            limit=k,
            output_fields=["text", "source", "metadata"],
            search_params={"metric_type": "COSINE", "params": {"nprobe": 16}},
        )

        docs: list[Document] = []
        for hit in results[0]:
            entity = hit.get("entity", {})
            text = entity.get("text", "")
            source = entity.get("source", "")
            metadata_str = entity.get("metadata", "{}")
            try:
                metadata = json.loads(metadata_str)
            except (json.JSONDecodeError, TypeError):
                metadata = {}
            metadata["source"] = source
            metadata["score"] = hit.get("distance", 0)
            docs.append(Document(page_content=text, metadata=metadata))

        return docs

    def query_all(self, limit: int = 10000) -> list[dict]:
        """
        查询所有文档（用于 BM25 检索）。

        Returns:
            包含 text, source, metadata 的字典列表
        """
        client = self._get_client()
        self._ensure_collection()

        results = client.query(
            collection_name=self._collection,
            filter="id >= 0",
            output_fields=["text", "source", "metadata"],
            limit=limit,
        )
        return results

    def drop(self) -> None:
        """删除当前集合（慎用）。"""
        client = self._get_client()
        if self._collection in client.list_collections():
            client.drop_collection(self._collection)

    def count(self) -> int:
        """返回当前集合中的文档数量。"""
        client = self._get_client()
        self._ensure_collection()
        stats = client.get_collection_stats(self._collection)
        return int(stats.get("row_count", 0))
