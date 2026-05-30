"""
Gradio 前端交互界面

提供聊天对话 + 文件上传功能的 Web UI。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr

from config import settings
from data_processor import process_file
from modules.chat_session import ChatSession
from modules.retriever import HybridRetriever
from modules.llm import DeepSeekLLM
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# ── 全局状态 ─────────────────────────────────────────────────
chat_session = ChatSession()
retriever = HybridRetriever()
llm = DeepSeekLLM()


def upload_file(file) -> str:
    """处理上传的文件，自动摄入到 Milvus。"""
    if file is None:
        return "⚠️ 请先选择文件"

    file_path = Path(file.name)
    if not file_path.exists():
        return f"❌ 文件不存在: {file_path}"

    from modules.loader import DocumentLoader
    if not DocumentLoader.is_supported(file_path):
        return f"❌ 不支持的格式: {file_path.suffix}，支持 PDF/Word/TXT/MD"

    try:
        count = process_file(file_path, verbose=False)
        return f"✅ 文件 '{file_path.name}' 摄入成功！共 {count} 个文本块已存入数据库。"
    except Exception as e:
        return f"❌ 摄入失败: {e}"


def chat(message: str, history: list[list[str]]) -> tuple:
    """
    处理聊天消息。

    Args:
        message: 用户输入
        history: Gradio 聊天历史 [[user, assistant], ...]

    Returns:
        更新后的聊天历史
    """
    if not message.strip():
        return history, ""

    # 1. 多路召回 + Rerank
    docs = retriever.retrieve(message, top_k=settings.rerank_top_n)

    # 2. 构建上下文
    if docs:
        context_parts = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "未知来源")
            page = doc.metadata.get("page", "")
            header = f"[文档{i+1}] {Path(source).name}"
            if page:
                header += f" 第{page}页"
            context_parts.append(f"{header}\n{doc.page_content}")
        context = "\n\n---\n\n".join(context_parts)
    else:
        context = "（未检索到相关文档）"

    # 3. 记录用户消息
    chat_session.add_user_message(message)

    # 4. 构建含历史的 prompt
    messages = chat_session.build_context_prompt(message, context)

    # 5. 调用 LLM
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
    chat_session.add_assistant_message(answer)

    # 7. 更新历史（Gradio 6.0 messages 格式）
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return history, ""


def clear_history():
    """清空对话历史。"""
    chat_session.clear()
    return [], "🗑️ 对话历史已清空"


def get_status() -> str:
    """获取系统状态信息。"""
    lines = [
        "📊 **系统状态**\n",
        f"- Embedding 模型: `{settings.embedding_model}`",
        f"- LLM 模型: `{settings.deepseek_model}`",
        f"- Rerank 模型: `{settings.rerank_model}`",
        f"- Milvus: `{settings.milvus_host}:{settings.milvus_port}`",
        f"- Chunk 大小: `{settings.chunk_size}`",
        f"- 检索 Top-K: `{settings.retrieval_top_k}`",
        f"- Rerank Top-N: `{settings.rerank_top_n}`",
        f"- 对话记忆轮数: `{settings.chat_history_length}`",
    ]
    try:
        from modules.vector_store import MilvusStore
        from modules.embeddings import DashScopeEmbeddings
        store = MilvusStore(DashScopeEmbeddings())
        count = store.count()
        lines.append(f"- 已摄入文档数: `{count}`")
    except Exception:
        lines.append("- Milvus: ❌ 连接失败")

    return "\n".join(lines)


def build_ui():
    """构建 Gradio 界面。"""
    with gr.Blocks(title="RAG 智能问答系统") as app:
        gr.Markdown(
            """
            # 🤖 RAG 智能问答系统
            基于 Milvus + DashScope + DeepSeek 的检索增强生成系统
            """
        )

        with gr.Row():
            # ── 左侧：聊天区域 ──
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="对话",
                    height=500,
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="请输入你的问题...",
                        show_label=False,
                        scale=4,
                        container=False,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("🗑️ 清空对话", size="sm")
                    status_msg = gr.Textbox(
                        label="提示",
                        interactive=False,
                        visible=True,
                        max_lines=1,
                    )

            # ── 右侧：文件上传 + 状态 ──
            with gr.Column(scale=1):
                gr.Markdown("### 📂 文档上传")
                file_input = gr.File(
                    label="拖入或选择文件",
                    file_types=[".pdf", ".docx", ".txt", ".md"],
                    type="filepath",
                )
                upload_btn = gr.Button("📤 上传并摄入", variant="secondary")
                upload_status = gr.Textbox(
                    label="上传状态",
                    interactive=False,
                    max_lines=3,
                )

                gr.Markdown("### 📊 系统状态")
                status_display = gr.Markdown(
                    value=get_status(),
                    elem_classes="status-box",
                )
                refresh_btn = gr.Button("🔄 刷新状态", size="sm")

        # ── 事件绑定 ──
        msg_input.submit(
            chat,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input],
        ).then(lambda: "", outputs=[status_msg])

        send_btn.click(
            chat,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input],
        ).then(lambda: "", outputs=[status_msg])

        clear_btn.click(
            clear_history,
            outputs=[chatbot, status_msg],
        )

        upload_btn.click(
            upload_file,
            inputs=[file_input],
            outputs=[upload_status],
        )

        refresh_btn.click(
            get_status,
            outputs=[status_display],
        )

    return app


def main():
    """启动 Web UI。"""
    print("\n🚀 启动 RAG Web UI ...")
    print(f"   地址: http://127.0.0.1:7860")
    print(f"   按 Ctrl+C 停止\n")

    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
