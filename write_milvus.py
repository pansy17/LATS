# 流程图注释：
# ┌─────────────────────────────┐
# │ 主进程：main                │
# │ 1. 创建 Queue(maxsize=20)   │
# │ 2. 启动 parser_proc         │
# │ 3. 启动 writer_proc         │
# │ 4. join 等待两子进程结束    │
# └─────┬───────────────────────┘
#       │fork                      fork
#       ▼                          ▼
# ┌──────────────┐         ┌────────────────────┐
# │parser_proc   │         │writer_proc         │
# │(file_parser_ │         │(milvus_writer_     │
# │ process)      │         │ process)            │
# │1. 扫描*.md    │         │1. 连接 Milvus       │
# │2. 解析→docs   │─batch─►│2. 循环get队列        │
# │3. 满batch或   │         │3. 收到None→break     │
# │  结束→put    │         │4. 收到list→写入      │
# │4. 最后put None│         │5. 打印写入计数       │
# └──────────────┘         └────────────────────┘
#       │put None               ▲
#       └───────────────────────┘
#
# 队列 docs_queue 作为生产者-消费者桥梁，实现异步解析与写入。

import multiprocessing
import os
from multiprocessing import Queue

from documents.MarkdownParser import MarkdownParser
from documents.milvus_db import MilvusVectorSave
from documents.log_utils import log

def file_parser_process(dir_path: str, output_queue: Queue, batch_size: int = 20):
    """进程1：解析目录下的所有markdown文件，并将解析后的文档放入输出队列"""
    # 记录开始解析目录的日志
    log.info(f"开始解析目录:{dir_path}")
    # 遍历目录，筛选出所有以.md结尾的文件，构造完整路径列表
    md_files = [
        os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.md')
    ]

    # 如果没有找到markdown文件，记录警告日志并提前返回
    if not md_files:
        log.warning(f"目录:{dir_path} 下没有找到 markdown 文件")
        return
    # 初始化Markdown解析器实例
    parser = MarkdownParser()
    # 初始化文档批次缓存列表
    doc_batch = []
    # 遍历每个markdown文件进行解析
    for file_path in md_files:
        try:
            # 调用解析器解析当前markdown文件，返回文档列表
            docs = parser.parse_markdown_to_documents(file_path)
            # 如果解析结果非空，将文档加入批次缓存
            if docs:
                doc_batch.extend(docs)

            # 当批次缓存达到设定大小时，将副本放入输出队列并清空缓存
            if len(doc_batch) >= batch_size:
                output_queue.put(doc_batch.copy())
                doc_batch.clear()
        except Exception as e:
            # 捕获解析异常，记录错误日志
            log.error(f"解析文件:{file_path} 失败:{e}")
    # 遍历结束后，如果批次缓存还有剩余文档，将其副本放入输出队列
    if doc_batch:
        output_queue.put(doc_batch.copy())
    # 向队列放入None作为结束信号，通知消费者进程结束
    output_queue.put(None)
    # 记录解析目录完成的日志
    log.info(f"解析目录:{dir_path} 结束")

def milvus_writer_process(output_queue: Queue):
    """进程2：从输入队列获取文档并写入Milvus数据库"""
    log.info("开始写入 Milvus")
    mv = MilvusVectorSave()
    mv.create_connection()
    total_count = 0
    while True:
        try:
            datas = output_queue.get()
            if datas is None:
                break
            
            if isinstance(datas, list):
                mv.add_documents(datas)
                total_count += len(datas)
                log.info(f"成功写入 {len(datas)} 条文档，当前总数:{total_count}")
        except Exception as e:
            log.error(f"写入 Milvus 失败:{e}")
            log.exception(e)
    log.info("写入 Milvus 结束")

if __name__ == '__main__':
    md_dir = "D:/a_job/y1/project+train/llm_learn/my_rag/pdf2md/trans_md"
    queue_max_size = 20
    mv = MilvusVectorSave()
    mv.create_connection()
    docs_queue = Queue(maxsize=queue_max_size)

    # 启动子进程
    parser_proc = multiprocessing.Process(
        target=file_parser_process,
        args=(md_dir, docs_queue)
    )
    writer_proc = multiprocessing.Process(
        target=milvus_writer_process,
        args=(docs_queue,)
    )
    parser_proc.start()
    writer_proc.start()
    
    # 等待进程结束
    parser_proc.join()
    writer_proc.join()

    print("系统提示：所有任务完成")