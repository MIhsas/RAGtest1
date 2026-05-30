"""
独立数据处理模块

提供一次调用完成 load → clean → split → embed → store 的完整流水线。
可被 ingest.py、web_ui.py、main.py 共同调用。
"""

import time
from pathlib import Path

from tqdm import tqdm

from config import settings
from modules.embeddings import DashScopeEmbeddings
from modules.loader import DocumentLoader
from modules.splitter import TextSplitter
from modules.vector_store import MilvusStore


def clean_documents(documents):
    """
    基础文档清洗：去除首尾空白、过滤空文档、合并连续空行。
    """
    cleaned = []
    for doc in documents:
        text = doc.page_content.strip()
        if not text:
            continue
        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join(line for line in lines if line)
        doc.page_content = text
        if text:
            cleaned.append(doc)
    return cleaned


def process_file(file_path: str | Path, verbose: bool = True) -> int:
    """
    处理单个文件：加载 → 清洗 → 切割 → 嵌入 → 存入 Milvus。

    Args:
        file_path: 文件路径
        verbose: 是否打印详细信息

    Returns:
        写入 Milvus 的 chunk 数量
    """
    path = Path(file_path)
    if verbose:
        print(f"\n📄 加载文件: {path.name}")

    docs = DocumentLoader.load(path)
    if verbose:
        print(f"   加载完成: {len(docs)} 个文档片段")

    docs = clean_documents(docs)
    if verbose:
        print(f"   清洗完成: {len(docs)} 个有效片段")

    splitter = TextSplitter()
    chunks = splitter.split(docs)
    if verbose:
        print(f"   切割完成: {len(chunks)} 个 chunk")

    embeddings = DashScopeEmbeddings()
    store = MilvusStore(embeddings)

    if verbose:
        print("   正在写入 Milvus ...")
    start = time.time()
    store.add_documents(chunks)
    elapsed = time.time() - start
    if verbose:
        print(f"   ✅ 写入完成: {len(chunks)} 个 chunk, 耗时 {elapsed:.1f}s")

    return len(chunks)


def process_directory(
    dir_path: str | Path | None = None,
    verbose: bool = True,
) -> int:
    """
    批量处理目录下所有支持格式的文件。

    Args:
        dir_path: 目录路径，默认为 documents/
        verbose: 是否打印详细信息

    Returns:
        总写入 chunk 数量
    """
    directory = Path(dir_path) if dir_path else settings.documents_dir
    if verbose:
        print(f"\n📂 扫描目录: {directory}")

    docs = DocumentLoader.load_dir(directory)
    if not docs:
        if verbose:
            print("   ⚠️  未找到任何支持格式的文件")
            print(f"   请将 PDF/Word/TXT/MD 文件放入 {directory}/ 目录")
        return 0
    if verbose:
        print(f"   加载完成: {len(docs)} 个文档片段")

    docs = clean_documents(docs)
    if verbose:
        print(f"   清洗完成: {len(docs)} 个有效片段")

    splitter = TextSplitter()
    chunks = splitter.split(docs)
    if verbose:
        print(f"   切割完成: {len(chunks)} 个 chunk")

    embeddings = DashScopeEmbeddings()
    store = MilvusStore(embeddings)

    if verbose:
        print("   正在批量写入 Milvus ...")
    start = time.time()

    batch_size = 100
    total_written = 0
    iterator = range(0, len(chunks), batch_size)
    if verbose:
        iterator = tqdm(iterator, desc="   写入进度")

    for i in iterator:
        batch = chunks[i : i + batch_size]
        store.add_documents(batch)
        total_written += len(batch)

    elapsed = time.time() - start
    if verbose:
        print(f"   ✅ 全部完成: {total_written} 个 chunk, 耗时 {elapsed:.1f}s")

    return total_written


if __name__ == "__main__":
    """直接运行：处理 documents/ 目录"""
    process_directory()
