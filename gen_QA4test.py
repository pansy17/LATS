import os
import json
import uuid
from typing import List, Dict
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv

# 加载环境变量
load_dotenv(find_dotenv())
OpenAI_API_KEY = os.getenv("OPENAI_API_KEY")
OpenAI_API_BASE = os.getenv("OPENAI_API_BASE")
# --- 配置 ---
TARGET_FOLDER = "D:/a_job/y1/project+train/llm_learn/my_rag/documents/datas/md"  # 你的Markdown文件夹路径
OUTPUT_FILE = "agent_knowledge_base.json" # 输出文件名

# --- 定义输出格式 ---
class QAPair(BaseModel):
    question: str = Field(description="基于文本生成的具体问题")
    answer: str = Field(description="基于文本的详细答案")
    keywords: List[str] = Field(description="涉及的核心实体或关键词")

# --- 核心逻辑 ---
def process_markdown_folder(folder_path: str) -> List[Dict]:
    """遍历文件夹并加载MD文件"""
    all_chunks = []
    
    # 1. 定义Markdown分割策略 (按标题切分以保留语义完整性)
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    # 二次切分：防止某个章节过长，超过LLM窗口或检索块限制
    char_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    print(f"📂 开始扫描文件夹: {folder_path} ...")

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                print(f"   处理文件: {file}")
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    
                    # 第一次切分：按章节
                    header_splits = md_splitter.split_text(text)
                    
                    # 第二次切分：按长度
                    final_splits = char_splitter.split_documents(header_splits)
                    
                    for split in final_splits:
                        # 补充元数据
                        split.metadata["source_file"] = file
                        all_chunks.append(split)
                        
                except Exception as e:
                    print(f"   ❌ 读取失败 {file}: {e}")

    print(f"✅ 共生成 {len(all_chunks)} 个文本片段，开始生成 QA...")
    return all_chunks
def generate_qa_pairs(chunks: List, output_path: str):
    """利用 LLM 为每个片段生成 QA"""
    
    # --- 修正部分 Start ---
    # 使用正确的参数名: openai_api_key 和 openai_api_base
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0.5, 
        openai_api_key=OpenAI_API_KEY, 
        openai_api_base=OpenAI_API_BASE
    ) 
    # --- 修正部分 End ---
    
    parser = JsonOutputParser(pydantic_object=QAPair)
    
    prompt = ChatPromptTemplate.from_template(
        """
        你是一个专门为 RAG 智能体构建知识库的专家。
        请阅读下面的技术文档片段，并生成一个高质量的问答对。
        
        文档片段:
        {context}
        
        要求:
        1. **问题 (Question)**: 必须是用户可能会问的真实问题，包含具体的实体名称（不要问“它是什么”，要问“Tx-Module是什么”）。
        2. **答案 (Answer)**: 必须完全基于文档片段，事实准确，不要编造。
        3. **格式**: 必须是合法的 JSON。
        
        {format_instructions}
        """
    )
    
    chain = prompt | llm | parser
    
    knowledge_base = []
    
    for i, chunk in enumerate(chunks):
        content = chunk.page_content
        # 简单的过滤：太短的片段不生成
        if len(content) < 50:
            continue
            
        print(f"   [{i+1}/{len(chunks)}] 生成中...")
        
        try:
            # 构造上下文，包含标题信息增强语义
            header_context = " > ".join([v for k,v in chunk.metadata.items() if "Header" in k])
            full_context = f"章节路径: {header_context}\n内容: {content}"
            
            result = chain.invoke({
                "context": full_context,
                "format_instructions": parser.get_format_instructions()
            })
            
            # 构建最终存储结构
            entry = {
                "id": str(uuid.uuid4()),
                "source_file": chunk.metadata.get("source_file"),
                "context": full_context, 
                "question": result["question"], 
                "answer": result["answer"], 
                "keywords": result.get("keywords", [])
            }
            knowledge_base.append(entry)
            
        except Exception as e:
            # 打印更详细的错误，方便排查
            print(f"   ⚠️ 生成失败 (Chunk {i}): {e}")

    # 保存文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 完成！数据已保存至 {output_path}")
    print(f"   共生成 {len(knowledge_base)} 条 QA 数据。")
    
if __name__ == "__main__":
    if not os.path.exists(TARGET_FOLDER):
        os.makedirs(TARGET_FOLDER)
        print(f"请在 {TARGET_FOLDER} 下放入 .md 文件")
    else:
        chunks = process_markdown_folder(TARGET_FOLDER)
        generate_qa_pairs(chunks, OUTPUT_FILE)