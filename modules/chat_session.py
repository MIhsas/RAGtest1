"""
多轮对话管理模块

维护对话历史，支持滑动窗口记忆，
将历史对话 + 检索上下文 + 当前问题组装为完整的 LLM Prompt。
"""

from dataclasses import dataclass, field

from config import settings


@dataclass
class Message:
    """单条对话消息。"""
    role: str      # "user" 或 "assistant"
    content: str


class ChatSession:
    """
    多轮对话会话管理。

    维护对话历史，支持滑动窗口（保留最近 N 轮），
    并将历史上下文注入到 LLM 的 Prompt 中。
    """

    def __init__(self, max_history: int | None = None):
        """
        Args:
            max_history: 最大保留轮数，默认从配置读取
        """
        self._max_history = max_history or settings.chat_history_length
        self._history: list[Message] = []

    def add_user_message(self, content: str) -> None:
        """添加用户消息。"""
        self._history.append(Message(role="user", content=content))

    def add_assistant_message(self, content: str) -> None:
        """添加助手回复。"""
        self._history.append(Message(role="assistant", content=content))

    def get_history(self) -> list[Message]:
        """获取当前对话历史（滑动窗口）。"""
        # 每轮 = 1条用户消息 + 1条助手回复
        max_messages = self._max_history * 2
        if len(self._history) > max_messages:
            return self._history[-max_messages:]
        return list(self._history)

    def build_context_prompt(
        self,
        query: str,
        retrieved_context: str,
        system_prompt: str | None = None,
    ) -> list[dict[str, str]]:
        """
        构建包含对话历史的完整 Prompt。

        Args:
            query: 当前用户问题
            retrieved_context: 检索到的文档上下文
            system_prompt: 自定义系统提示词

        Returns:
            OpenAI 格式的消息列表 [{"role": ..., "content": ...}, ...]
        """
        from modules.llm import DeepSeekLLM

        prompt = system_prompt or DeepSeekLLM.DEFAULT_SYSTEM_PROMPT

        messages: list[dict[str, str]] = [
            {"role": "system", "content": prompt},
        ]

        # 注入历史对话
        for msg in self.get_history():
            messages.append({"role": msg.role, "content": msg.content})

        # 注入当前问题 + 检索上下文
        user_message = (
            f"以下是检索到的相关文档片段，请基于这些内容回答问题。\n"
            f"{'='*50}\n"
            f"{retrieved_context}\n"
            f"{'='*50}\n\n"
            f"问题：{query}\n\n"
            f"请综合以上文档片段中的信息，给出准确、完整的回答。"
        )
        messages.append({"role": "user", "content": user_message})

        return messages

    def clear(self) -> None:
        """清空对话历史。"""
        self._history.clear()

    @property
    def turn_count(self) -> int:
        """当前对话轮数。"""
        return len(self._history) // 2
