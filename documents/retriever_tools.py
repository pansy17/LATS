# 流程图注释：
# 1. 连接 Milvus 并初始化 MilvusVectorSave 实例
# 2. 定义 ManualSparseRetriever 类，实现稀疏向量检索
# 3. 【新增】定义 DynamicEnsembleRetriever 类，支持动态权重和缓存
# 4. 构建 dense_retriever 和 sparse_retriever
# 5. 实例化动态检索器，并封装为 tool

from typing import List, Callable, Dict, Any, Optional
from functools import lru_cache
import hashlib
import json

from langchain_core.tools import create_retriever_tool
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_milvus import Milvus
from .milvus_db import MilvusVectorSave

# --- 简单内存缓存实现 ---
class SimpleMemoryCache:
    def __init__(self, capacity: int = 100):
        self.cache = {}
        self.capacity = capacity
        self.order = [] # 用于 LRU 淘汰

    def get(self, key: str) -> Optional[List[Document]]:
        if key in self.cache:
            # 更新 LRU 位置
            self.order.remove(key)
            self.order.append(key)
            print(f"⚡ [Cache Hit] 命中热点查询缓存")
            return self.cache[key]
        return None

    def set(self, key: str, value: List[Document]):
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            # 淘汰最久未使用的
            oldest = self.order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = value
        self.order.append(key)

# 全局缓存实例
query_cache = SimpleMemoryCache(capacity=100)

mv = MilvusVectorSave()
mv.create_connection()

# --- 1. 自定义稀疏检索器 (保持原有逻辑，稍作优化) ---
class ManualSparseRetriever(BaseRetriever):
    """
    自定义检索器，用于手动将查询字符串转换为稀疏向量，
    并直接调用 pymilvus 的 col.search() 方法。
    """
    # 与 Milvus 向量数据库的连接实例，用于执行向量搜索
    vectorstore: Milvus
    # 将查询字符串转换为稀疏向量（BM25）的函数
    sparse_builder: Callable[[str], dict]
    # 传给 Milvus 搜索的额外参数，如 top-k、过滤条件、向量字段名等
    search_kwargs: dict

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        
        # 1. BM25 转换
        sparse_vector = self.sparse_builder(query)
        if not sparse_vector:
            return []

        # 2. 获取参数
        k = self.search_kwargs.get("k", 4)
        field = self.search_kwargs.get("vector_field", "sparse")
        filter_expr_str = self.search_kwargs.get("filter", {}).get("category", "")
        filter_expr = f"category == '{filter_expr_str}'" if filter_expr_str else ""
        
        # 3. 构造 Milvus 搜索参数
        search_params = {
            "metric_type": "IP",
            "params": {} 
        }
        
        # 4. 执行搜索
        try:
            res = self.vectorstore.col.search(
                data=[sparse_vector],
                anns_field=field,
                param=search_params,
                limit=k,
                expr=filter_expr,
                output_fields=["text", "source", "filename", "title", "category"]
            )
        except Exception as e:
            print(f"⚠️ Sparse Search Error: {e}")
            return []
        
        # 5. 结果转换
        docs = []
        if not res or not res[0]:
            return []
            
        for hit in res[0]:
            metadata = {
                "source": hit.entity.get("source"),
                "filename": hit.entity.get("filename"),
                "title": hit.entity.get("title"),
                "category": hit.entity.get("category"),
                "score": hit.distance,
                "retriever_source": "sparse" # 标记来源
            }
            # 清理 None 值
            metadata = {k: v for k, v in metadata.items() if v is not None}
            
            docs.append(Document(
                page_content=hit.entity.get("text"), 
                metadata=metadata
            ))
        
        return docs

# --- 2. 【核心优化】动态混合检索器 (Dynamic Hybrid Retriever) ---
class DynamicEnsembleRetriever(BaseRetriever):
    """
    支持动态权重调整和缓存的混合检索器。
    替代 LangChain 原生的 EnsembleRetriever。
    """
    dense_retriever: BaseRetriever
    sparse_retriever: BaseRetriever
    # 默认权重
    default_dense_weight: float = 0.5
    default_sparse_weight: float = 0.5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        
        # --- A. 缓存层检查 ---
        # 生成缓存 Key (Query + 意图 tag 如果有的话)
        cache_key = hashlib.md5(query.encode()).hexdigest()
        cached_docs = query_cache.get(cache_key)
        if cached_docs:
            return cached_docs

        # --- B. 意图识别与动态权重 ---
        # 默认权重
        dense_w = self.default_dense_weight
        sparse_w = self.default_sparse_weight
        
        # 简单规则：检测是否包含特定的意图标记（这些标记通常由 Agent 在 Query 中注入，或者用正则判断）
        # 例如：如果 Query 看起来像型号（字母+数字），增加 Sparse 权重
        import re
        is_spec_query = bool(re.search(r'[a-zA-Z]+\d+', query)) # 粗略判断是否包含型号
        
        if is_spec_query:
            print(f"⚖️ [Dynamic Weight] 检测到型号特征，调高关键词检索权重 (Sparse=0.7)")
            dense_w = 0.3
            sparse_w = 0.7
        else:
            # 默认为概念/原理查询，偏向语义
            # print(f"⚖️ [Dynamic Weight] 默认为语义查询 (Dense=0.5)")
            pass

        # --- C. 并行执行检索 ---
        # 这里为了简单串行执行，生产环境可用 asyncio.gather
        dense_docs = self.dense_retriever.invoke(query)
        sparse_docs = self.sparse_retriever.invoke(query)

        # --- D. 加权融合算法 (Reciprocal Rank Fusion - RRF 的加权变体) ---
        # 简化版：基于分数的加权合并 (Weighted Sum of Scores)
        # 注意：Dense(Cosine/IP) 和 Sparse(IP) 的分数范围可能不同，建议先归一化。
        # 这里使用一种简化的去重+排序策略。
        
        all_docs_map = {}
        
        # 处理 Dense 结果
        for rank, doc in enumerate(dense_docs):
            doc_id = doc.page_content # 假设内容作为唯一ID，最好有真实ID
            # 给予排名倒数分 (RRF)
            score = (1 / (rank + 60)) * dense_w
            if doc_id not in all_docs_map:
                doc.metadata['retriever_source'] = 'dense'
                doc.metadata['fusion_score'] = score
                all_docs_map[doc_id] = doc
            else:
                all_docs_map[doc_id].metadata['fusion_score'] += score

        # 处理 Sparse 结果
        for rank, doc in enumerate(sparse_docs):
            doc_id = doc.page_content
            score = (1 / (rank + 60)) * sparse_w
            if doc_id not in all_docs_map:
                # doc.metadata['retriever_source'] = 'sparse' # 已经在类里加了
                doc.metadata['fusion_score'] = score
                all_docs_map[doc_id] = doc
            else:
                all_docs_map[doc_id].metadata['fusion_score'] += score
                # 标记为混合命中
                all_docs_map[doc_id].metadata['retriever_source'] = 'hybrid'

        # 排序
        sorted_docs = sorted(
            all_docs_map.values(), 
            key=lambda x: x.metadata.get('fusion_score', 0), 
            reverse=True
        )
        
        final_docs = sorted_docs[:4] # 取 Top 4
        
        # --- E. 写入缓存 ---
        query_cache.set(cache_key, final_docs)
        
        return final_docs

# --- 3. 实例化检索器 ---

# Dense Retriever (LangChain Built-in)
dense_retriever = mv.vector_store_saved.as_retriever(
    search_type='similarity',
    search_kwargs={
        "k": 4,
        "vector_field": 'dense',
        "score_threshold": 0.1,
        'filter': {"category": "content"}
    }
)

# Sparse Retriever (Custom Manual)
sparse_retriever = ManualSparseRetriever(
    vectorstore=mv.vector_store_saved,
    sparse_builder=mv.build_bm25_sparse_vector,
    search_kwargs={
        "k": 4,
        "vector_field": 'sparse',
        'filter': {"category": "content"}
    }
)

# 使用新的动态检索器替代 EnsembleRetriever
# 默认权重设为 0.5/0.5，但在运行时会根据 query 动态调整
ensemble_retriever = DynamicEnsembleRetriever(
    dense_retriever=dense_retriever,
    sparse_retriever=sparse_retriever,
    default_dense_weight=0.5,
    default_sparse_weight=0.5
)

# --- 4. 封装为 Tool ---
retriever_tool = create_retriever_tool(
    ensemble_retriever,
    'rag_retriever',
    '搜索并返回关于 ‘半导体和芯片’ 的信息, 内容涵盖：半导体和芯片的封装、测试、光刻胶等'
)

print("✅ [Retriever] 动态混合检索器已加载 (含 LRU 缓存 & 意图权重调整)")