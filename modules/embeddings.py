"""
DashScope Embedding 封装

将阿里云 DashScope 的 text-embedding-v3 模型封装为
LangChain Embeddings 接口，供 Milvus 向量数据库使用。
"""

from langchain_core.embeddings import Embeddings

from config import settings


class DashScopeEmbeddings(Embeddings):
    """
    基于 DashScope API 的文本向量嵌入模型。

    使用 OpenAI 兼容接口调用 DashScope 的 text-embedding-v3，
    支持批量嵌入，中文效果优秀。
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
    ):
        self.api_key = api_key or settings.dashscope_api_key
        self.model = model or settings.embedding_model
        self.dimensions = dimensions or settings.embedding_dim

        if not self.api_key:
            raise ValueError(
                "DashScope API Key 未配置，请在 .env 中设置 DASHSCOPE_API_KEY"
            )

        # 使用 OpenAI 兼容接口
        from openai import OpenAI

        self._client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        批量嵌入多段文本。

        DashScope 单次最多支持 25 条文本，自动分批处理。

        Args:
            texts: 待嵌入的文本列表

        Returns:
            向量列表，每个向量为 float 列表
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        batch_size = 10

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._client.embeddings.create(
                model=self.model,
                input=batch,
                dimensions=self.dimensions,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """
        嵌入单条查询文本。

        Args:
            text: 查询文本

        Returns:
            向量表示
        """
        response = self._client.embeddings.create(
            model=self.model,
            input=[text],
            dimensions=self.dimensions,
        )
        return response.data[0].embedding
