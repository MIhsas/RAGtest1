"""
数据摄入入口

调用 data_processor 模块完成文档处理。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_processor import process_directory, process_file


def main():
    """CLI 入口：python ingest.py [文件或目录路径]"""
    if len(sys.argv) < 2:
        process_directory()
    else:
        target = Path(sys.argv[1])
        if target.is_file():
            process_file(target)
        elif target.is_dir():
            process_directory(target)
        else:
            print(f"❌ 路径不存在: {target}")
            sys.exit(1)


if __name__ == "__main__":
    main()
