"""
多路召回 + RRF 融合 + Rerank 重排序模块

实现向量检索和 BM25 关键词检索两路召回，
通过 Reciprocal Rank Fusion (RRF) 融合结果，
最后通过 Reranker 精排。
"""

import json
import math

import jieba
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from config import settings
from modules.embeddings import DashScopeEmbeddings
from modules.reranker import DashScopeReranker
from modules.vector_store import MilvusStore


class HybridRetriever:
    """
    混合检索器：向量检索 + BM25 关键词检索 + RRF 融合 + Rerank 精排。
    """

    def __init__(self):
        self._embeddings = DashScopeEmbeddings()
        self._store = MilvusStore(self._embeddings)
        self._reranker = DashScopeReranker()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        use_rerank: bool = True,
    ) -> list[Document]:
        """
        执行混合检索。

        Args:
            query: 用户查询
            top_k: 最终返回的文档数量
            use_rerank: 是否启用 rerank 重排序

        Returns:
            检索到的 Document 列表
        """
        k = top_k or settings.rerank_top_n

        # ── 路径 1：向量检索 ──
        vector_docs = self._store.search(query, top_k=settings.retrieval_top_k)

        # ── 路径 2：BM25 关键词检索 ──
        bm25_docs = self._bm25_search(query, top_k=settings.bm25_top_k)

        # ── RRF 融合 ──
        fused = self._rrf_fusion(vector_docs, bm25_docs)

        # ── Rerank 精排 ──
        if use_rerank and len(fused) > k:
            fused = self._reranker.rerank(query, fused)

        return fused[:k]

    def _bm25_search(self, query: str, top_k: int) -> list[Document]:
        """
        BM25 关键词检索。

        从 Milvus 中取出所有文档的文本，构建 BM25 索引，
        然后对 query 进行关键词匹配。
        """
        results = self._store.query_all(limit=10000)

        if not results:
            return []

        # 中文分词
        corpus_tokens = [list(jieba.cut(r["text"])) for r in results]
        query_tokens = list(jieba.cut(query))

        # BM25 检索
        bm25 = BM25Okapi(corpus_tokens)
        scores = bm25.get_scores(query_tokens)

        # 取 top_k
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        docs: list[Document] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            r = results[idx]
            try:
                metadata = json.loads(r.get("metadata", "{}"))
            except (json.JSONDecodeError, TypeError):
                metadata = {}
            metadata["source"] = r.get("source", "")
            metadata["bm25_score"] = float(scores[idx])
            docs.append(Document(page_content=r["text"], metadata=metadata))

        return docs

    @staticmethod
    def _rrf_fusion(
        vector_docs: list[Document],
        bm25_docs: list[Document],
        k: int = 60,
    ) -> list[Document]:
        """
        Reciprocal Rank Fusion (RRF) 融合两路检索结果。

        RRF 公式: score = Σ 1 / (k + rank_i)

        Args:
            vector_docs: 向量检索结果
            bm25_docs: BM25 检索结果
            k: RRF 参数（通常取 60）

        Returns:
            融合后按 RRF 分数降序排列的文档列表
        """
        doc_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        # 向量检索排名
        for rank, doc in enumerate(vector_docs):
            key = doc.page_content[:100]  # 用前100字符作为去重 key
            doc_scores[key] = doc_scores.get(key, 0) + 1.0 / (k + rank + 1)
            doc_map[key] = doc

        # BM25 检索排名
        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content[:100]
            doc_scores[key] = doc_scores.get(key, 0) + 1.0 / (k + rank + 1)
            if key not in doc_map:
                doc_map[key] = doc

        # 按 RRF 分数排序
        sorted_keys = sorted(doc_scores, key=lambda x: doc_scores[x], reverse=True)
        result: list[Document] = []
        for key in sorted_keys:
            doc = doc_map[key]
            doc.metadata["rrf_score"] = doc_scores[key]
            result.append(doc)

        return result
