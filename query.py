"""
检索与生成流水线（重构版）

支持多轮对话记忆 + 多路召回 + Rerank 重排序。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from modules.chat_session import ChatSession
from modules.llm import DeepSeekLLM
from modules.retriever import HybridRetriever


def query(
    question: str,
    session: ChatSession | None = None,
    top_k: int | None = None,
    verbose: bool = True,
) -> str:
    """
    执行 RAG 检索问答（支持多轮对话）。

    Args:
        question: 用户问题
        session: 对话会话对象（传入可保持记忆）
        top_k: 最终返回的文档数量
        verbose: 是否打印详细信息

    Returns:
        生成的回答文本
    """
    if session is None:
        session = ChatSession()

    k = top_k or settings.rerank_top_n

    if verbose:
        print(f"\n🔍 检索中 (向量+BM25, top_k={k}) ...")

    # 1. 多路召回 + Rerank
    retriever = HybridRetriever()
    docs = retriever.retrieve(question, top_k=k)

    if not docs:
        return "未检索到相关文档，请确认已摄入数据。"

    if verbose:
        print(f"   检索到 {len(docs)} 个相关片段")
        for i, doc in enumerate(docs):
            preview = doc.page_content[:80].replace("\n", " ")
            source = doc.metadata.get("source", "未知来源")
            rerank_score = doc.metadata.get("rerank_score", "")
            score_info = f" (rerank={rerank_score:.3f})" if rerank_score else ""
            print(f"   [{i+1}] {source}{score_info} | {preview}...")

    # 2. 构建上下文
    context_parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "未知来源")
        page = doc.metadata.get("page", "")
        header = f"[文档{i+1}] 来源: {source}"
        if page:
            header += f", 第{page}页"
        context_parts.append(f"{header}\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    # 3. 记录用户消息
    session.add_user_message(question)

    # 4. 构建含历史的 prompt
    messages = session.build_context_prompt(question, context)

    # 5. 调用 LLM
    if verbose:
        print(f"\n🤖 正在生成回答 (历史 {session.turn_count} 轮) ...")

    llm = DeepSeekLLM()
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    lc_messages = []
    for m in messages:
        if m["role"] == "system":
            lc_messages.append(SystemMessage(content=m["content"]))
        elif m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))

    response = llm.llm.invoke(lc_messages)
    answer = response.content

    # 6. 记录助手回复
    session.add_assistant_message(answer)

    if verbose:
        print(f"\n{'='*60}")
        print(f"📌 问题: {question}")
        print(f"{'='*60}")
        print(f"💡 回答:\n{answer}")
        print(f"{'='*60}")

    return answer


def interactive():
    """交互式多轮对话模式。"""
    session = ChatSession()

    print("=" * 60)
    print("🤖 RAG 智能问答系统 (多轮对话模式)")
    print("   输入 'quit' 退出 | 输入 'clear' 清空历史")
    print("=" * 60)

    while True:
        try:
            question = input(f"\n[第{session.turn_count + 1}轮] 请输入问题: ").strip()
            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                print("👋 再见！")
                break
            if question.lower() == "clear":
                session.clear()
                print("🗑️  对话历史已清空")
                continue
            query(question, session=session)
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")


def main():
    """CLI 入口。"""
    if len(sys.argv) < 2:
        print("用法:")
        print("  python query.py '你的问题'")
        print("  python query.py --interactive")
        sys.exit(1)

    if sys.argv[1] in ("--interactive", "-i"):
        interactive()
    else:
        question = " ".join(sys.argv[1:])
        query(question)


if __name__ == "__main__":
    main()
