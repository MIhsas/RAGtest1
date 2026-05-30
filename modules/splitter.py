"""
文本切割模块

基于 LangChain RecursiveCharacterTextSplitter 实现，
支持中英文混合文本的智能分块。
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings


class TextSplitter:
    """文本切割器，将长文档拆分为适合向量嵌入的 chunk。"""

    # 中英文友好的分隔符优先级列表
    DEFAULT_SEPARATORS = [
        "\n\n",  # 段落
        "\n",    # 换行
        "。",    # 中文句号
        "！",    # 中文感叹号
        "？",    # 中文问号
        "；",    # 中文分号
        ".",     # 英文句号
        "!",     # 英文感叹号
        "?",     # 英文问号
        ";",     # 英文分号
        " ",     # 空格
        "",      # 字符级回退
    ]

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
    ):
        """
        初始化切割器。

        Args:
            chunk_size: 每个 chunk 的最大字符数，默认从配置读取
            chunk_overlap: chunk 之间的重叠字符数，默认从配置读取
            separators: 自定义分隔符列表
        """
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size or settings.chunk_size,
            chunk_overlap=chunk_overlap or settings.chunk_overlap,
            separators=separators or self.DEFAULT_SEPARATORS,
            length_function=len,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """
        将文档列表切割为更小的 chunk。

        Args:
            documents: 原始 Document 列表

        Returns:
            切割后的 Document 列表，保留原始元数据并追加 chunk 序号
        """
        chunks = self._splitter.split_documents(documents)

        # 为每个 chunk 添加序号元数据
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i

        return chunks

    def split_text(self, text: str) -> list[str]:
        """直接切割纯文本字符串。"""
        return self._splitter.split_text(text)
