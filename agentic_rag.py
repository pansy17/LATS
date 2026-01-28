# 整体流程图（ASCII）：
# ┌──────────────┐
# │  用户输入query  │
# └──────┬───────┘
#        ▼
# ┌──────────────┐
# │ retrieve_node │ ← 调用ensemble_retriever获取RAG文档
# └──────┬───────┘
#        ▼
# ┌──────────────┐
# │ evaluate_node │ ← LLM评估文档质量，返回good/poor
# └──────┬───────┘
#        ▼
# ┌──────────────┐
# │should_web_search│ ← 条件分支：poor→web_search；good→generate
# └──┬───────┬───┘
#   ▼       ▼
# ┌────────┐ ┌────────┐
# │web_search│ │ generate│
# └────┬────┘ └────┬────┘
#      └────┬──────┘
#           ▼
# ┌────────────────┐
# │  最终answer输出  │
# └────────────────┘

# 导入所需类型与模块
from typing import List,TypedDict
# 引入Document类型，用于封装检索到的文档
from langchain_core.documents import Document
# 引入ChatOpenAI，用于调用大模型
from langchain_openai import ChatOpenAI
# 引入提示模板，用于构造LLM输入
from langchain_core.prompts import ChatPromptTemplate
# 引入字符串输出解析器
from langchain_core.output_parsers import StrOutputParser
# 引入Google Serper搜索工具
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
# 引入自定义的ensemble检索器
from documents.retriever_tools import ensemble_retriever
# 引入状态图与结束节点
from langgraph.graph import StateGraph, END

# 导入系统库
import os
# 导入dotenv，用于读取本地.env文件
from dotenv import load_dotenv, find_dotenv
# 加载.env文件
load_dotenv(find_dotenv())
# 读取OpenAI API密钥
openai_api_key = os.environ.get("OPENAI_API_KEY")
# 读取OpenAI API基础地址
openai_api_base = os.environ.get("OPENAI_API_BASE")
# 读取Serper API密钥
serper_api_key = os.environ.get("SERPER_API_KEY")

# 初始化LLM与搜索工具
try:
    # 创建ChatOpenAI实例，使用gpt-4o-mini模型，temperature=0确保确定性
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, openai_api_key=openai_api_key, openai_api_base=openai_api_base)
    # 创建GoogleSerper搜索工具实例
    web_search_tool = GoogleSerperAPIWrapper(api_key=serper_api_key) 
except Exception as e:
    # 如果初始化失败，打印异常并提示检查环境变量
    print(e)
    print("请检查环境变量 OPENAI_API_KEY 和 SERPER_API_KEY 是否正确设置。")

# 定义一个图状态类，用于在图节点间传递数据
class GraphState(TypedDict):
    """A state of the graph."""
    query: str  # 用户查询
    documents: List[Document]  # 检索到的文档列表
    context_quality: str  # 文档质量评估结果
    answer: str  # 最终生成的答案

# 定义retrieve节点：负责从RAG知识库中检索文档
def retrieve_node(state: GraphState):
    # 打印节点开始信息
    print("1.正在RAG检索节点...")
    # 从状态中获取用户查询
    query = state["query"]
    
    # 直接调用ensemble_retriever，返回List[Document]
    documents = ensemble_retriever.invoke(query) 
    
    # 打印检索到的文档数量
    print(f"检索到 {len(documents)} 篇 RAG 文档")
    # 返回更新后的文档列表
    return {"documents": documents}

# 定义evaluate节点：使用LLM评估检索到的文档质量
def evaluate_node(state:GraphState):
    """
    使用LLM评估节点质量
    """
    # 打印节点开始信息
    print("2.正在评估节点质量...")
    # 从状态中获取查询与文档
    query = state["query"]
    documents = state["documents"]

    # 如果文档为空，直接返回poor
    if not documents: # 如果没有文档，则返回 "无"
        print("无文档")
        return {"context_quality": "poor"}
    # 定义评估提示模板
    eval_prompt_template = """
    给定以下用户查询和一组检索到的文档：

    查询: {query}

    文档:
    {documents}

    请评估这些文档是否足够、相关且高质量，足以回答该查询。
    你只需要回答两个词中的一个：
    - 'good' (如果文档足够好)
    - 'poor' (如果文档不相关、不充分或质量低下)
    """
    # 创建提示模板对象
    eval_prompt = ChatPromptTemplate.from_template(eval_prompt_template)
    # 定义函数：将文档列表格式化为字符串
    def format_docs_for_prompt(docs: List[Document]) -> str:
       return "\n\n".join(f"--- Doc {i+1} ---\n{doc.page_content}" for i, doc in enumerate(docs)) 
    # 构建评估链：提示 | LLM | 输出解析
    eval_chain = eval_prompt | llm | StrOutputParser()
    # 格式化文档
    formatted_docs = format_docs_for_prompt(documents)
    # 调用评估链获取质量结果
    quality = eval_chain.invoke({"query": query, "documents": formatted_docs})
    # 根据返回字符串判断最终质量
    quality_decision = "poor" if "poor" in quality.lower() else "good"
    # 打印评估结果
    print(f"评估结果: {quality_decision}")
    # 返回质量结果
    return {"context_quality": quality_decision}

# 定义web_search节点：当RAG质量为poor时，执行网页搜索补充文档
def web_search_node(state: GraphState):
    """
    如果 RAG 质量 'poor'，则执行此节点进行网页搜索。
    """
    # 打印节点开始信息
    print("--- 3. (修正) 节点：执行网页搜索 ---")
    # 从状态中获取用户查询
    query = state["query"]
    
    # 调用Serper搜索工具获取结果
    search_results = web_search_tool.results(query) 
    
    # 初始化网页文档列表
    web_docs = []
    # 如果搜索结果中有organic字段
    if "organic" in search_results:
        # 仅选择前3个结果
        for result in search_results["organic"][:3]:
            # 封装为Document对象
            web_docs.append(Document(
                page_content=result.get("snippet", "No snippet available"),
                metadata={
                    "source": result.get("link", "N/A"),
                    "title": result.get("title", "N/A"),
                    "source_type": "web" # 标记为网页来源
                }
            ))
    
    # 打印网页搜索到的文档数量
    print(f"网页搜索到 {len(web_docs)} 篇新文档")
    
    # 将网页搜索结果追加到原有RAG文档列表
    all_documents = state["documents"] + web_docs
    
    # 返回合并后的文档列表
    return {"documents": all_documents} # 用合并后的列表覆盖状态

# 定义generate节点：利用最终文档列表生成答案
def generate_node(state: GraphState):
    """
    使用最终的文档列表 (RAG 或 RAG + Web) 来生成答案。
    """
    # 打印节点开始信息
    print("--- 4. 节点：生成最终答案 ---")
    # 从状态中获取查询与文档
    query = state["query"]
    documents = state["documents"]

    # 定义生成提示模板
    gen_prompt_template = """
    你是一个关于半导体和芯片的专家助手。
    请使用以下上下文信息来全面回答用户的问题。
    如果上下文信息不足，请根据你的知识库回答，但要说明上下文未提供此信息。

    上下文 (可能包含知识库和网页搜索结果):
    {context}

    问题: {query}

    回答:
    """
    # 创建生成提示模板对象
    gen_prompt = ChatPromptTemplate.from_template(gen_prompt_template)

    # 定义函数：将文档列表格式化为带来源的字符串
    def format_context_for_gen(docs: List[Document]) -> str:
         return "\n\n".join(
            f"--- 来源: {doc.metadata.get('source', 'N/A')} (类型: {doc.metadata.get('source_type', 'RAG')}) ---\n{doc.page_content}" 
            for doc in docs
        )
    def format_context_for_gen(docs: List[Document]) -> str:
        return "\n\n".join(
            f"--- 来源: {doc.metadata.get('source', 'N/A')} (类型: {doc.metadata.get('source_type', 'RAG')}) ---\n{doc.page_content}" 
            for doc in docs
        )

    # --- 以下是补全的部分 ---
    
    # 构建生成链：提示 | LLM | 输出解析
    rag_chain = gen_prompt | llm | StrOutputParser()
    
    # 格式化上下文
    formatted_context = format_context_for_gen(documents)
    
    # 生成答案
    answer = rag_chain.invoke({"context": formatted_context, "query": query})
    
    # 打印答案片段
    print(f"生成的答案片段: {answer[:50]}...")
    
    # 返回最终答案
    return {"answer": answer}

# ==========================================
# 5. 构建 LangGraph 工作流
# ==========================================

# 定义条件分支逻辑：根据评估结果决定下一步
def decide_to_web_search(state: GraphState):
    """
    根据评估节点的 context_quality 决定下一个节点。
    """
    print("--- 检测是否需要网页搜索 ---")
    quality = state["context_quality"]
    
    if quality == "poor":
        print("评估结果为 poor -> 转向网页搜索 (web_search)")
        return "web_search"
    else:
        print("评估结果为 good -> 直接生成答案 (generate)")
        return "generate"

# 初始化状态图
workflow = StateGraph(GraphState)

# 添加节点
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("evaluate", evaluate_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("generate", generate_node)

# 定义边的连接逻辑
# 1. 起点 -> 检索
workflow.set_entry_point("retrieve")

# 2. 检索 -> 评估
workflow.add_edge("retrieve", "evaluate")

# 3. 评估 -> 条件分支 (web_search 或 generate)
workflow.add_conditional_edges(
    "evaluate",               # 上一个节点
    decide_to_web_search,     # 决策函数
    {                         # 映射关系：函数返回值 -> 下一个节点名
        "web_search": "web_search",
        "generate": "generate"
    }
)

# 4. 网页搜索 -> 生成 (搜索补全后，必须去生成)
workflow.add_edge("web_search", "generate")

# 5. 生成 -> 结束
workflow.add_edge("generate", END)

# 编译图（生成可执行的 Runnable）
app = workflow.compile()

# ==========================================
# 6. 执行调用 (Main 函数)
# ==========================================

if __name__ == "__main__":
    import pprint
    
    # 测试问题
    # query = "什么是 GAA 晶体管技术？"  # 这个可能直接走 RAG
    query = "2024年台积电的最新3nm产能情况如何？" # 这个可能需要走 Web Search (如果知识库没更新)
    
    print(f"\n🚀 开始执行 Agentic RAG，问题：{query}\n" + "="*50)
    
    # 运行图
    inputs = {"query": query}
    
    # app.invoke 会返回最终的状态字典
    try:
        final_state = app.invoke(inputs)
        
        print("\n" + "="*50)
        print("✅ 执行完成！最终答案：\n")
        print(final_state["answer"])
        print("\n" + "="*50)
        
        # (可选) 打印使用的文档来源，确认是否用了网页搜索
        print("📚 参考文档来源：")
        for i, doc in enumerate(final_state.get("documents", [])):
           source = doc.metadata.get("source", "Unknown")
           type_ = doc.metadata.get("source_type", "RAG")
           print(f"{i+1}. [{type_}] {source}")
            
    except Exception as e:
        print(f"\n❌ 执行过程中发生错误: {e}")