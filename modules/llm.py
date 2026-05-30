"""
DeepSeek LLM 封装

通过 OpenAI 兼容接口调用 DeepSeek Chat 模型，
提供标准的 LangChain ChatModel 接口。
"""

from langchain_openai import ChatOpenAI

from config import settings


class DeepSeekLLM:
    """DeepSeek 大语言模型封装。"""

    # 默认系统提示词
    DEFAULT_SYSTEM_PROMPT = (
        "你是一个基于文档的智能问答助手。请遵循以下规则回答：\n"
        "1. 仔细阅读每一条上下文片段，从中提取与问题相关的所有信息\n"
        "2. 即使信息分散在不同片段中，也要综合整理后给出完整回答\n"
        "3. 回答时引用原文中的关键数据（如数字、年限、条件等）作为依据\n"
        "4. 只有当所有上下文中确实完全没有相关信息时，才告知用户无法回答\n"
        "5. 不要编造文档中不存在的信息"
    )

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ):
        """
        初始化 DeepSeek LLM。

        Args:
            api_key: DeepSeek API Key
            base_url: API 基础 URL
            model: 模型名称
            temperature: 生成温度（0-1），越低越确定
        """
        _api_key = api_key or settings.deepseek_api_key
        if not _api_key:
            raise ValueError(
                "DeepSeek API Key 未配置，请在 .env 中设置 DEEPSEEK_API_KEY"
            )

        self._llm = ChatOpenAI(
            api_key=_api_key,
            base_url=base_url or settings.deepseek_base_url,
            model=model or settings.deepseek_model,
            temperature=temperature if temperature is not None else settings.llm_temperature,
        )

    @property
    def llm(self) -> ChatOpenAI:
        """返回底层 ChatOpenAI 实例，可直接用于 LangChain 链。"""
        return self._llm

    def invoke(self, query: str, context: str, system_prompt: str | None = None) -> str:
        """
        基于上下文生成回答。

        Args:
            query: 用户问题
            context: 检索到的上下文文本
            system_prompt: 自定义系统提示词

        Returns:
            生成的回答文本
        """
        from langchain_core.messages import SystemMessage, HumanMessage

        prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        user_message = (
            f"以下是检索到的相关文档片段，请基于这些内容回答问题。\n"
            f"{'='*50}\n"
            f"{context}\n"
            f"{'='*50}\n\n"
            f"问题：{query}\n\n"
            f"请综合以上文档片段中的信息，给出准确、完整的回答。"
        )

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=user_message),
        ]

        response = self._llm.invoke(messages)
        return response.content
