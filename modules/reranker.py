"""
DashScope Rerank 重排序模块

使用阿里云 DashScope 的 gte-rerank 模型对检索结果进行精排，
提升最终返回给 LLM 的文档相关性。
"""

from langchain_core.documents import Document

from config import settings


class DashScopeReranker:
    """基于 DashScope gte-rerank 的文档重排序器。"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        top_n: int | None = None,
    ):
        self._api_key = api_key or settings.dashscope_api_key
        self._model = model or settings.rerank_model
        self._top_n = top_n or settings.rerank_top_n

        if not self._api_key:
            raise ValueError(
                "DashScope API Key 未配置，请在 .env 中设置 DASHSCOPE_API_KEY"
            )

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        """
        对文档列表进行重排序。

        Args:
            query: 用户查询
            documents: 待排序的 Document 列表

        Returns:
            重排序后的 Document 列表（按相关性降序，取 top_n）
        """
        if not documents:
            return []

        import dashscope

        dashscope.api_key = self._api_key

        # 提取文档文本
        doc_texts = [doc.page_content for doc in documents]

        response = dashscope.TextReRank.call(
            model=self._model,
            query=query,
            documents=doc_texts,
            top_n=min(self._top_n, len(documents)),
            return_documents=True,
        )

        # 根据 rerank 结果重组文档列表
        reranked: list[Document] = []
        for item in response.output.results:
            original_doc = documents[item.index]
            original_doc.metadata["rerank_score"] = item.relevance_score
            reranked.append(original_doc)

        return reranked
