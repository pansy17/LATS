'''负责将文档数据存储到milvus数据库中'''
import sys
import os
import math
import jieba
from collections import Counter
from typing import List
from dotenv import load_dotenv, find_dotenv

# --- 第三方库导入 ---
from langchain_core.documents import Document
from langchain_milvus import Milvus
from pymilvus import IndexType, MilvusClient, Function, Collection
from pymilvus.client.types import MetricType, DataType, FunctionType

# --- 本地模块导入 (请确保这些文件存在) ---
try:
    from .llm_models import bge_embedding
    from .MarkdownParser import MarkdownParser
except ImportError:
    # 为了防止直接运行时报错，这里做一个简单的mock，实际运行时请确保路径正确
    print("Warning: 本地模块导入失败，请确保在项目根目录下运行或调整PYTHONPATH")
    bge_embedding = None 
    MarkdownParser = None

_ = load_dotenv(find_dotenv())
MILVUS_URI = os.getenv("MILVUS_URI", "./milvus_demo.db") # 默认给个本地文件路径以防报错
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_collection")
# OPENAI配置根据需要保留

class MilvusVectorSave:
    """把新的document数据插入到数据库中"""

    def __init__(self) -> object:
        """初始化"""
        self.vector_store_saved: Milvus = None

    def build_bm25_sparse_vector(self, text: str) -> dict:
        """
        使用 BM25 算法将文本转为稀疏向量（手动实现）
        返回: {token_id: weight} 格式的 dict
        """
        # 1. 分词
        tokens = list(jieba.cut(text, cut_all=False))
        # 2. 过滤非中文、非字母数字
        tokens = [t for t in tokens if t.strip() and t.isalnum()]
        if not tokens:
            return {}

        # 3. 计算词频
        tf = Counter(tokens)
        total_tokens = len(tokens)

        # 4. 计算 BM25 权重
        k1 = 1.2
        b = 0.75
        avgdl = 100  # 平均文档长度，可调整
        sparse_vec = {}
        for term, freq in tf.items():
            # 简化 IDF：假设整个语料库只有当前文档（生产环境建议计算全局IDF，这里仅做示例）
            idf = 1.0
            # BM25 公式
            weight = idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * total_tokens / avgdl))
            # 使用 hash 作为 token_id，限制范围防止过大
            token_id = abs(hash(term)) % (10 ** 8)
            sparse_vec[token_id] = weight
        return sparse_vec

    def create_collection(self):
        """
        创建集合：如果集合不存在则创建，如果存在则跳过。
        不会删除已有数据。
        """
        client = MilvusClient(uri=MILVUS_URI) # 创建milvus连接
        
        # --- 关键修改：检查集合是否存在 ---
        if client.has_collection(collection_name=COLLECTION_NAME):
            print(f"集合 '{COLLECTION_NAME}' 已存在，跳过创建步骤。")
            return
        
        print(f"集合 '{COLLECTION_NAME}' 不存在，正在创建...")
        schema = client.create_schema() # 创建schema
        
        # 定义字段
        schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name='text', datatype=DataType.VARCHAR, max_length=6000, enable_analyzer=True,
                         analyzer_params={"tokenizer": "jieba", "filter": ["cnalphanumonly"]})
        schema.add_field(field_name='category', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='source', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='filename', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='filetype', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='title', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='category_depth', datatype=DataType.INT64)
        schema.add_field(field_name='sparse', datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name='dense', datatype=DataType.FLOAT_VECTOR, dim=512) # 请确保 embedding 模型维度匹配
        
        # 创建bm25函数
        bm25_func = Function(
            name="manual_bm25",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=["sparse"],
            params={"k1": 1.2, "b": 0.75}
        )
        schema.add_function(bm25_func)
        
        index_params = client.prepare_index_params()
        # 创建稀疏索引
        index_params.add_index(
            field_name="sparse",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={
                "inverted_index_algo": "DAAT_MAXSCORE",
                "bm25_k1": 1.2,
                "bm25_b": 0.75
            },
        )
        # 创建稠密索引
        index_params.add_index(
            field_name="dense",
            index_name="dense_inverted_index",
            index_type=IndexType.HNSW,
            metric_type=MetricType.IP,
            params={"M": 16, "efConstruction": 64}
        )

        client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params
        )
        print(f"集合 '{COLLECTION_NAME}' 创建成功。")

    def create_connection(self):
        """创建一个Connection： milvus + langchain"""
        # 注意：这里假设 bge_embedding 已经正确初始化
        self.vector_store_saved = Milvus(
            embedding_function=bge_embedding,
            collection_name=COLLECTION_NAME,
            vector_field='dense',   # 告诉 LangChain 默认的稠密字段
            text_field='text',      # 告诉 LangChain 文本字段
            consistency_level="Strong",
            auto_id=True,
            connection_args={"uri": MILVUS_URI}
        )

    def add_documents(self, datas: List[Document]):
        """
        把新的document保存到Milvus中。
        逻辑：先删除同名文件的旧数据，再插入新数据（覆盖更新）。
        """
        if not datas:
            print("没有数据需要处理。")
            return

        # 1. 获取当前批次的文件名（假设这批 datas 属于同一个文件）
        # 这一步非常关键，用于定位旧数据
        current_filename = datas[0].metadata.get("filename")
        
        # --- 关键修改：先删除旧数据 ---
        if current_filename:
            print(f"正在检查并清理旧数据: filename == '{current_filename}' ...")
            delete_expr = f'filename == "{current_filename}"'
            try:
                # 使用 col (pymilvus.Collection) 直接执行删除
                self.vector_store_saved.col.delete(delete_expr)
                # 刷新以确保删除生效（Strong consistency level 下通常不需要手动 flush，但在批量操作间加上更稳妥）
                # self.vector_store_saved.col.flush() 
                print(f"旧数据清理完成（如果存在）。")
            except Exception as e:
                print(f"警告：尝试删除旧数据时出错 (可能是首次上传): {e}")
        else:
            print("警告：文档缺少 'filename' 元数据，无法执行覆盖更新，将直接追加。")

        # --- 准备插入新数据 ---
        batch_to_insert = []
        for doc in datas:
            text = doc.page_content
            metadata = doc.metadata
            
            # 2. 手动生成稠密向量
            dense_vector = bge_embedding.embed_query(text)
            
            # 3. 手动生成稀疏向量
            sparse_vector = self.build_bm25_sparse_vector(text)
            
            # 4. 组装 entity
            entity = {
                "text": text,
                "dense": dense_vector,
                "sparse": sparse_vector,
                "category": metadata.get("category", ""),
                "source": metadata.get("source", ""),
                "filename": metadata.get("filename", ""),
                "filetype": metadata.get("filetype", ""),
                "title": metadata.get("title", ""),
                "category_depth": metadata.get("category_depth", 0),
            }
            batch_to_insert.append(entity)

        # 5. 执行插入
        if batch_to_insert:
            print(f"正在向 Milvus 插入 {len(batch_to_insert)} 条数据...")
            self.vector_store_saved.col.insert(batch_to_insert)
            self.vector_store_saved.col.flush() # 确保数据落盘可见
            print(f"文件 '{current_filename}' 写入完成。")
        

if __name__ == '__main__':
    # 模拟运行流程
    
    # 1. 定义文件路径
    file_path = r'documents/test.md'
    
    # 2. 解析文件 (确保你有这个类)
    if MarkdownParser:
        parser = MarkdownParser()
        docs = parser.parse_markdown_to_documents(file_path)
        
        # 为了演示覆盖效果，手动给 docs 加上 filename (如果 parser 没加的话)
        for doc in docs:
            if "filename" not in doc.metadata:
                doc.metadata["filename"] = os.path.basename(file_path)

        # 3. 执行入库
        mv = MilvusVectorSave()
        mv.create_collection()  # 第一次创建，第二次跳过
        mv.create_connection()  # 建立连接
        mv.add_documents(docs)  # 删除旧的test.md，写入新的test.md
    else:
        print("缺少 MarkdownParser 模块，无法执行完整流程。")