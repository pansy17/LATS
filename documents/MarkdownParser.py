from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from dotenv import load_dotenv, find_dotenv
from .log_utils import log

_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]
openai_api_base = os.environ["OPENAI_API_BASE"]

from typing import List
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document

"""
    作用
    - 解析 Markdown 文件为结构化的 Document 列表，并对长文本执行语义分块。
    - 根据 Unstructured 提取的标题/内容层级，将标题与其下属内容合并为聚合文档。

    核心职责
    - parse_markdown: 使用 UnstructuredMarkdownLoader 按元素解析 Markdown，生成 List[Document]。
    - text_chunker: 对超长 Document（page_content > 5000）进行语义分块，提升后续检索与向量化效果。
    - merge_title_content: 依据 metadata 的层级信息，将标题与内容聚合，形成带层级前缀的综合文档。

    输入与输出
    - 输入：
      - parse_markdown: 文件路径字符串（md_file）。
      - text_chunker: List[Document]，每个 Document 包含 page_content 与 metadata。
      - merge_title_content: List[Document]，通常为 parse_markdown 的输出。
    - 输出：
      - parse_markdown: 按元素解析后的 Document 列表。
      - text_chunker: 分块后的 Document 列表（保持原顺序，长文被拆分为多个子文档）。
      - merge_title_content: 合并后的 Document 列表，包含：
        - 独立正文（NarrativeText 且无 parent_id）原样保留；
        - 标题文档聚合其子内容，并在 page_content 中附加层级前缀与汇总内容。

    关键字段说明（metadata）
    - parent_id: 当前元素的父标题 ID（指向所属上级标题）。
    - category: 元素类型，例如 'Title'（标题）、'NarrativeText'（正文内容）等。
    - element_id: 当前元素的唯一标识，用于建立标题缓存索引。
    - language/languages: 语言信息字段；在聚合时会移除以减少噪声（具体字段名在不同解析配置下可能为 language 或 languages）。

    方法细节
    - __init__:
      - 初始化 SemanticChunker，底层使用 OpenAIEmbeddings 对文本向量化，按语义边界进行分段。
      - breakpoint_threshold_type="percentile"：以分位数阈值控制分段点的敏感度。
    - text_chunker:
      - 遍历输入文档；若 page_content 长度超过 5000，则使用 SemanticChunker 的 split_document 进行语义分块。
      - 其他文档直接保留，最终返回新的 List[Document]。
    - parse_markdown:
      - 构造 UnstructuredMarkdownLoader(md_file, 'elements', 'fast')，以元素级方式快速解析 Markdown。
      - 使用 lazy_load 流式读取每个元素，转换为 Document 并收集。
    - merge_title_content（聚合算法概述）：
      1. 遍历所有 Document，移除 metadata 中的语言字段（如存在）。
      2. 对 category=='Title' 的文档：
         - 将标题文本写入 metadata['title']；
         - 如果该标题有父标题（parent_id 在缓存中），将父标题的 page_content 作为层级前缀，用 "->" 串联构造“父 -> 子”层级标题；
         - 将该标题以 element_id 为键缓存到 parent_dict。
      3. 对非标题且有 parent_id 的内容文档：
         - 将内容的 page_content 追加到对应父标题的 page_content（以 "->" 连接）；
         - 将父标题的 metadata['category'] 标记为 'content'，表明该标题已聚合了内容。
      4. 对独立正文（NarrativeText 且无 parent_id）的文档：
         - 直接加入结果列表 merged_data。
      5. 遍历结束后，将 parent_dict 中所有已处理过的标题文档加入 merged_data 并返回。

    注意与边界条件
    - 标题/内容顺序：默认假设标题在其子内容之前出现；若不满足，追加到父标题前应检查 parent_id 是否已缓存。
    - 元数据副作用：聚合会修改 Document 的 metadata（写入 'title'、更新 'category'）。如需保留原始值，建议先复制。
    - 输出顺序：parent_dict 依赖插入顺序（Python 3.7+ 字典保持插入有序）；若需严格顺序可额外排序。
    - 字段一致性：不同解析配置下可能存在 'language' 与 'languages' 的命名差异，建议统一清理策略。

    复杂度
    - 时间：O(n)，单次线性遍历。
    - 空间：O(k)，其中 k 为标题元素数量（用于 parent_dict 缓存）。
    """
class MarkdownParser:
    def __init__(self):
        self.text_spliter = SemanticChunker(
            OpenAIEmbeddings(openai_api_key=openai_api_key, openai_api_base=openai_api_base),
            breakpoint_threshold_type="percentile",
        )#这段逻辑是使用OpenAI的Embedding模型对文本进行向量化，并使用SemanticChunker进行文本分块。
    def text_chunker(self, datas: List[Document]) -> List[Document]:#
        new_docs = []
        for d in datas:
            if len(d.page_content)>5000:
                new_docs.extend(self.text_spliter.split_document(d))#函数将原始数据进行分块，并返回分块后的数据，每个分块的长度不超过5000，每个分块的metadata与原始数据保持一致。
                continue
            new_docs.append(d)
        return new_docs#函数将原始数据进行分块，并返回分块后的数据。
    def parse_markdown(self, md_file: str) -> List[Document]:
        loader = UnstructuredMarkdownLoader(md_file, mode="elements")  # 去掉不支持的 strategy 参数
        docs = []
        for doc in loader.lazy_load():#遍历loader.lazy_load()生成的元素级
            docs.append(doc)
        return docs  # 返回元素级 Document 列表
    def merge_title_content(self,datas: List[Document])->List[Document]:
        merge_data = []  # 初始化一个空列表，用于存放最终合并后的文档
        parent_dict = {}  # 初始化一个空字典，用于缓存标题元素，key 为 element_id，value 为对应的 Document
        for document in datas:  # 遍历输入的文档列表
            metadata = document.metadata  # 取出当前文档的元数据
            if 'language' in metadata:  # 如果元数据中存在 language 字段
                metadata.pop('language')#删除metadata中的language字段
            parent_id = metadata.get('parent_id',None)  # 尝试获取当前元素的父级 ID，若无则为 None
            category = metadata.get('category',None)  # 尝试获取当前元素的类别，若无则为 None
            element_id = metadata.get('element_id',None)  # 尝试获取当前元素的唯一 ID，若无则为 None
            if category == 'NarrativeText' and parent_id is None :  # 若当前元素为独立正文且无父级
                merge_data.append(document)  # 直接将其加入结果列表
            if category == 'Title':  # 若当前元素为标题
                document.metadata['title'] = document.page_content  # 将标题文本写入元数据的 title 字段
                if parent_id in parent_dict:  # 若该标题存在父标题（已缓存）
                    document.page_content = parent_dict[parent_id].page_content+'->'+document.page_content  # 将父标题内容作为前缀拼接到当前标题
                parent_dict[element_id] = document#创建一个字典，将标题元素和内容元素进行关联
            if category != 'Title' and parent_id is not None:  # 若当前元素非标题且存在父级
                parent_dict[parent_id].page_content += '->' + document.page_content  # 将其内容追加到对应父标题的 page_content 中
                parent_dict[parent_id].metadata['category'] = 'content'  # 将父标题的类别标记为 content，表示已聚合内容
        if parent_dict:  # 若缓存字典非空
            merge_data.extend(parent_dict.values())  # 将所有已聚合的标题文档加入结果列表
        
        return merge_data#函数将标题元素和内容元素进行合并，并返回合并后的数据。


    def parse_markdown_to_documents(self, md_file: str, encoding: str = "utf-8") -> List[Document]:
        documents = self.parse_markdown(md_file)  # 调用上面的"元素解析"方法
        # 下游处理：合并标题与内容 → 语义分块
        merge_data = self.merge_title_content(documents)
        chunk_docs = self.text_chunker(merge_data)
        
        # 确保每个文档都有必要的元数据字段
        processed_docs = []
        for doc in chunk_docs:
            # 只保留我们需要的字段，避免添加未定义的字段
            metadata = {}
            
            # 添加文件相关信息
            import os
            file_path = os.path.abspath(md_file)
            metadata['source'] = file_path
            metadata['filename'] = os.path.basename(file_path)
            metadata['filetype'] = 'markdown'
            
            # 从原始metadata中复制允许的字段
            if doc.metadata:
                allowed_fields = ['category', 'title', 'category_depth']
                for field in allowed_fields:
                    if field in doc.metadata:
                        metadata[field] = doc.metadata[field]
            
            # 确保所有必需的字段都存在
            if 'category' not in metadata:
                metadata['category'] = 'content'  # 默认为内容类型
            if 'title' not in metadata:
                metadata['title'] = metadata.get('filename', '')  # 使用文件名作为默认标题
            if 'category_depth' not in metadata:
                metadata['category_depth'] = 0  # 默认深度为0
            
            # 创建新文档
            processed_doc = Document(page_content=doc.page_content, metadata=metadata)
            processed_docs.append(processed_doc)
        
        return processed_docs

if __name__ == '__main__':
    md_file = 'D:/a_job/y1/project+train/llm_learn/my_rag/documents/test.md'
    parser = MarkdownParser()
    docs = parser.parse_markdown_to_documents(md_file)
    print(docs)