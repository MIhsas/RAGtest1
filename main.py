"""
RAG 项目主入口

用法:
    python main.py ingest                  摄入 documents/ 文件夹中的所有文件
    python main.py ingest <文件或目录>      摄入指定路径
    python main.py query <问题>            单次问答
    python main.py chat                    交互式多轮对话
    python main.py web                     启动 Web 前端
    python main.py status                  查看系统状态
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import settings


def print_banner():
    print(
        r"""
 ╔══════════════════════════════════════╗
 ║       RAG 智能问答系统 v2.0          ║
 ║   Milvus · DashScope · DeepSeek      ║
 ║  多轮对话 · Rerank · 多路召回        ║
 ╚══════════════════════════════════════╝"""
    )


def cmd_ingest(args: list[str]):
    from data_processor import process_directory, process_file

    if not args:
        process_directory()
    else:
        target = Path(args[0])
        if target.is_file():
            process_file(target)
        elif target.is_dir():
            process_directory(target)
        else:
            print(f"❌ 路径不存在: {target}")
            sys.exit(1)


def cmd_query(args: list[str]):
    if not args:
        print("❌ 请提供问题")
        print("用法: python main.py query '你的问题'")
        sys.exit(1)
    from query import query
    question = " ".join(args)
    query(question)


def cmd_chat(_args: list[str]):
    from query import interactive
    interactive()


def cmd_web(_args: list[str]):
    from web_ui import main as web_main
    web_main()


def cmd_status(_args: list[str]):
    print("\n📊 系统状态:")
    print(f"   项目路径:        {settings.project_root}")
    print(f"   文档目录:        {settings.documents_dir}")
    print(f"   Embedding 模型:  {settings.embedding_model}")
    print(f"   LLM 模型:        {settings.deepseek_model}")
    print(f"   Rerank 模型:     {settings.rerank_model}")
    print(f"   Milvus 地址:     {settings.milvus_host}:{settings.milvus_port}")
    print(f"   Chunk 大小:      {settings.chunk_size}")
    print(f"   Chunk 重叠:      {settings.chunk_overlap}")
    print(f"   检索 Top-K:      {settings.retrieval_top_k}")
    print(f"   Rerank Top-N:    {settings.rerank_top_n}")
    print(f"   对话记忆轮数:    {settings.chat_history_length}")

    docs_dir = settings.documents_dir
    if docs_dir.exists():
        from modules.loader import DocumentLoader
        files = [f for f in docs_dir.iterdir() if f.is_file() and DocumentLoader.is_supported(f)]
        print(f"\n   📂 documents/ 目录中: {len(files)} 个待处理文件")
        for f in files:
            size_kb = f.stat().st_size / 1024
            print(f"      - {f.name} ({size_kb:.1f} KB)")

    try:
        from modules.embeddings import DashScopeEmbeddings
        from modules.vector_store import MilvusStore
        embeddings = DashScopeEmbeddings()
        store = MilvusStore(embeddings)
        count = store.count()
        print(f"\n   ✅ Milvus 连接正常, 已摄入文档数: {count}")
    except Exception as e:
        print(f"\n   ❌ Milvus 连接失败: {e}")

    print("\n   API Key 状态:")
    print(f"   {'✅' if settings.dashscope_api_key else '❌'} DASHSCOPE_API_KEY")
    print(f"   {'✅' if settings.deepseek_api_key else '❌'} DEEPSEEK_API_KEY")


def print_help():
    print(
        """
用法:
  python main.py ingest                摄入 documents/ 文件夹中的所有文件
  python main.py ingest <路径>         摄入指定文件或目录
  python main.py query <问题>          基于已摄入文档回答问题
  python main.py chat                  交互式多轮对话模式
  python main.py web                   启动 Web 前端界面
  python main.py status                查看系统状态

快速开始:
  1. 将 PDF/Word/TXT/MD 文件放入 documents/ 文件夹
  2. python main.py ingest
  3. python main.py web"""
    )


COMMANDS = {
    "ingest": cmd_ingest,
    "query": cmd_query,
    "chat": cmd_chat,
    "web": cmd_web,
    "status": cmd_status,
    "help": print_help,
}


def main():
    print_banner()

    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)

    command = sys.argv[1]
    args = sys.argv[2:]

    handler = COMMANDS.get(command)
    if handler is None:
        print(f"❌ 未知命令: {command}")
        print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
