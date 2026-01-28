import json
import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv

# LangChain 组件
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# 引入 Serper 搜索工具
from langchain_community.utilities import GoogleSerperAPIWrapper

# ============================
# 0. 路径与环境配置
# ============================
# 加载 .env 文件 (确保包含 SERPER_API_KEY 和 OPENAI_API_KEY)
load_dotenv(find_dotenv())

# 检查 Serper Key
if not os.environ.get("SERPER_API_KEY"):
    print("⚠️ 警告: 未检测到 SERPER_API_KEY，搜索功能将不可用，可能导致报错。")

# 初始化搜索工具
try:
    search_tool = GoogleSerperAPIWrapper(k=3) # 获取前3条结果
    print("✅ 成功初始化 Google Serper 搜索工具")
except Exception as e:
    print(f"❌ Serper 工具初始化失败: {e}")
    search_tool = None

# 尝试导入 Agent
try:
    # 假设您的文件结构如下，请根据实际情况调整
    from test.easy_rag import rag_chain as easy_rag_chain
    from test.agentic_rag import app as agentic_app
    from test.agentic_corrective_rag import app as corrective_app
    print("✅ 成功导入所有 Agent")
except ImportError as e:
    try:
        from easy_rag import rag_chain as easy_rag_chain
        from agentic_rag import app as agentic_app
        from agentic_corrective_rag import app as corrective_app
        print("✅ 成功导入所有 Agent (同级目录)")
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        # 为了演示代码运行，这里如果不导入成功可能会报错，实际使用请确保路径正确
        # exit(1) 

# ============================
# 1. 辅助函数：获取搜索证据
# ============================
def fetch_search_evidence(query):
    """
    利用 Serper 对 Query 进行搜索，返回摘要作为事实证据
    """
    if not search_tool:
        return "无法连接搜索引擎，无外部证据。"
    
    try:
        # 使用 .run 获取拼接好的字符串，或者用 .results 获取详细 dict
        # 这里为了给 LLM 阅读，直接用 run 获取简洁文本
        results = search_tool.run(query)
        if not results:
            return "未搜索到相关结果。"
        return results
    except Exception as e:
        print(f"搜索出错: {e}")
        return "搜索过程中发生错误。"

# ============================
# 2. 定义评估函数 (Checklist + Search Verification)
# ============================

def llm_evaluate_checklist(query, prediction, ground_truth):
    """
    使用 Checklist 方法进行细粒度评估，并结合网络搜索验证幻觉。
    """
    # 1. 获取网络证据 (这是本次优化的核心)
    search_evidence = fetch_search_evidence(query)
    
    # 2. 调用 LLM 裁判
    eval_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    checklist_prompt = ChatPromptTemplate.from_template("""
    你是一个半导体领域的权威技术评估员。请结合“标准答案”和“网络事实证据”，对“模型回答”进行严格评估。

    === 输入信息 ===
    【用户问题】: {query}
    【标准答案】(Ground Truth): {ground_truth}
    【网络事实证据】(Search Evidence): {search_evidence}
    【模型回答】(Model Prediction): {prediction}

    === 评估任务 ===
    请对模型回答进行以下 5 项检查（每项 2 分，满分 10 分）。
    
    判定规则特别说明：
    - 如果模型回答了标准答案中没有、但网络证据证实为真的细节，**不属于幻觉**，应视为正确（completeness 或 no_hallucination 给 true）。
    - 如果模型回答与网络证据或标准答案直接矛盾，视为错误。

    === Checklist ===
    1. [key_terms] 关键术语: 是否包含了标准答案中的核心技术名词（如 FinFET, GAA, HBM 等）？
    2. [numeric_accuracy] 数值准确: 涉及的数字参数是否与标准答案或网络证据一致？
    3. [logic_mechanism] 逻辑机制: 技术原理解释是否符合物理事实逻辑？
    4. [completeness] 完整性: 是否涵盖了标准答案的主要点？(若补充了额外的真实信息也算完整)
    5. [no_hallucination] 无幻觉: 回答内容是否**没有**编造虚假信息？
       (注意：如果内容在标准答案未提及，但在网络证据中存在，则**不是**幻觉，此项应为 true)。

    === 输出格式 ===
    请严格输出 JSON 格式，不要包含 Markdown 标记：
    {{
        "checklist": {{
            "key_terms": true,
            "numeric_accuracy": true,
            "logic_mechanism": true,
            "completeness": true,
            "no_hallucination": true
        }},
        "reason": "简短评价，指出哪里有幻觉或哪里补充了额外真实信息"
    }}
    """)
    
    try:
        chain = checklist_prompt | eval_llm
        res = chain.invoke({
            "query": query, 
            "prediction": prediction, 
            "ground_truth": ground_truth,
            "search_evidence": search_evidence  # 传入搜索结果
        })
        
        content = res.content.strip()
        
        # 清洗 JSON
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
            
        data = json.loads(content)
        
        checks = data.get("checklist", {})
        keys = ["key_terms", "numeric_accuracy", "logic_mechanism", "completeness", "no_hallucination"]
        
        score = sum(2 for k in keys if checks.get(k, False))
        
        return score, checks, data.get("reason", "")
        
    except Exception as e:
        print(f"Checklist 评估出错: {e}")
        empty_checks = {k: False for k in ["key_terms", "numeric_accuracy", "logic_mechanism", "completeness", "no_hallucination"]}
        return 0.0, empty_checks, "Error"

def calculate_final_score(checklist_score):
    return checklist_score * 10.0

# ============================
# 3. Agent 统一调用接口
# ============================

def run_easy_rag(query):
    try: return easy_rag_chain.invoke(query)
    except: return ""

def run_agentic_rag(query):
    try: return agentic_app.invoke({"query": query})["answer"]
    except: return ""

def run_corrective_rag(query):
    try:
        return corrective_app.invoke({"original_query": query, "max_iterations": 3})["best_answer"]
    except: return ""

AGENTS = {
    "EasyRAG": run_easy_rag,
    "AgenticRAG": run_agentic_rag,
    "CorrectiveRAG": run_corrective_rag
}

# ============================
# 4. 主程序
# ============================

def main():
    # 1. 加载数据
    data_file = "agent_genknowledge.json" 
    if not os.path.exists(data_file):
        if os.path.exists("agent_knowledge_base.json"):
            data_file = "agent_knowledge_base.json"
        else:
            print(f"❌ 找不到数据文件: {data_file}")
            # 创建个假数据方便测试运行
            print("⚠️ 创建临时测试数据...")
            dataset = [{"id": "001", "question": "HBM3E的传输速率是多少？", "answer": "HBM3E的数据传输速率最高可达9.6 Gbps。"}]
    else:
        with open(data_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    
    # ⚠️ 调试模式：跑前 3 条
    dataset = dataset[:3]
    
    print(f"🚀 开始评估 {len(dataset)} 条数据 (Checklist + 网络搜索验证)...")
    
    results = []
    checklist_keys = ["key_terms", "numeric_accuracy", "logic_mechanism", "completeness", "no_hallucination"]

    for item in tqdm(dataset):
        query = item["question"]
        ground_truth = item["answer"]
        doc_id = item.get("id", "N/A")
        
        row_data = {"id": doc_id, "query": query} # 保存Query方便人工核对
        
        for agent_name, run_func in AGENTS.items():
            # A. 运行 Agent
            start_t = time.time()
            prediction = run_func(query)
            cost_time = time.time() - start_t
            
            # B. 评估 (LLM + Search)
            score, checks, reason = llm_evaluate_checklist(query, prediction, ground_truth)
            
            # C. 记录数据
            final_score = calculate_final_score(score)
            
            row_data[f"{agent_name}_Score"] = score
            row_data[f"{agent_name}_Final"] = final_score
            row_data[f"{agent_name}_Time"] = round(cost_time, 2)
            row_data[f"{agent_name}_Reason"] = reason # 保存评估理由
            
            for k in checklist_keys:
                row_data[f"{agent_name}_{k}"] = 1 if checks.get(k, False) else 0
            
        results.append(row_data)

    # 4. 生成 DataFrame
    df = pd.DataFrame(results)
    
    # 5. 简单的统计输出
    print("\n" + "="*30)
    print("评估完成，正在生成结果...")
    
    # 计算平均分
    score_cols = [c for c in df.columns if "_Final" in c]
    if score_cols:
        print("各 Agent 平均分:")
        print(df[score_cols].mean())

    # 6. 保存 CSV
    output_csv = "rag_evaluation_with_search_check.csv"
    df.to_csv(output_csv, index=False, encoding="utf_8_sig")
    print(f"✅ 结果已保存至: {output_csv}")

if __name__ == "__main__":
    main()