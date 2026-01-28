# 导入所需类型与模块
import os
from typing import List
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 导入自定义的ensemble检索器 (保持与复杂版本一致的检索源)
from documents.retriever_tools import ensemble_retriever

# 导入dotenv
from dotenv import load_dotenv, find_dotenv

# 1. 环境配置与初始化
# ==========================================
load_dotenv(find_dotenv())
openai_api_key = os.environ.get("OPENAI_API_KEY")
openai_api_base = os.environ.get("OPENAI_API_BASE")

# 初始化LLM (保持与复杂版本一致的模型参数，确保公平对比)
try:
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0.0, 
        openai_api_key=openai_api_key, 
        openai_api_base=openai_api_base
    )
except Exception as e:
    print(e)
    print("请检查环境变量 OPENAI_API_KEY 是否正确设置。")

# 2. 定义辅助函数
# ==========================================
def format_docs(docs: List[Document]) -> str:
    """
    将检索到的文档列表格式化为字符串，拼接到 Prompt 中
    """
    return "\n\n".join(
        f"--- 来源: {doc.metadata.get('source', 'N/A')} ---\n{doc.page_content}" 
        for doc in docs
    )

# 3. 构建 RAG 链 (LCEL 风格)
# ==========================================

# 定义与复杂版本一致的 Prompt，去掉网页搜索相关的描述即可
template = """
你是一个关于半导体和芯片的专家助手。
请使用以下上下文信息来全面回答用户的问题。
如果上下文信息不足，请根据你的知识库回答，但要说明上下文未提供此信息。

上下文:
{context}

问题: {query}

回答:
"""
prompt = ChatPromptTemplate.from_template(template)

# 构建标准的 RAG 链: 
# 1. 检索(ensemble_retriever) -> 2. 格式化(format_docs) -> 3. 提示词填充(prompt) -> 4. 生成(llm) -> 5. 解析(parser)
rag_chain = (
    {"context": ensemble_retriever | format_docs, "query": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 4. 执行调用 (Main 函数)
# ==========================================

if __name__ == "__main__":
    # 测试问题 (使用与 Agentic 版本相同的问题进行对比)
    # query = "什么是 GAA 晶体管技术？" 
    query = "2024年台积电的最新3nm产能情况如何？" 
    
    print(f"\n🚀 [对照组] 开始执行标准 RAG，问题：{query}\n" + "="*50)
    
    try:
        # 1. 获取答案
        # invoke 传入 query 字符串，RunnablePassthrough 会将其传给 query，ensemble_retriever 会将其传给 retrieve
        answer = rag_chain.invoke(query)
        
        print("\n" + "="*50)
        print("✅ 执行完成！最终答案：\n")
        print(answer)
        print("\n" + "="*50)

        # # (可选) 为了调试，单独打印一下检索到的文档，看看单纯检索到了什么
        # print("📚 [Debug] 检索到的 Top 文档片段：")
        # retrieved_docs = ensemble_retriever.invoke(query)
        # for i, doc in enumerate(retrieved_docs[:3]): # 只看前3个
        #     print(f"{i+1}. {doc.page_content[:100]}... (Source: {doc.metadata.get('source', 'N/A')})")

    except Exception as e:
        print(f"\n❌ 执行过程中发生错误: {e}")