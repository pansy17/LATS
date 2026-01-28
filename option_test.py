import json
import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ============================
# 0. 路径与环境配置
# ============================
try:
    # 尝试导入 Agent
    from test.easy_rag import rag_chain as easy_rag_chain
    from test.agentic_rag import app as agentic_app
    from test.agentic_corrective_rag import app as corrective_app
    print("✅ 成功导入所有 Agent")
except ImportError:
    try:
        from easy_rag import rag_chain as easy_rag_chain
        from agentic_rag import app as agentic_app
        from agentic_corrective_rag import app as corrective_app
        print("✅ 成功导入所有 Agent (同级目录)")
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        exit(1)

# ============================
# 1. 核心功能函数
# ============================

def format_mcq_query(item):
    """
    将选择题 JSON 对象格式化为 Agent 可理解的输入字符串。
    """
    options_str = "\n".join([f"{k}. {v}" for k, v in item["options"].items()])
    query = f"""{item['question']}

请根据已知信息，从以下选项中选择一个最正确的答案，并简要说明理由：
{options_str}
"""
    return query

def extract_option_with_llm(question, prediction, options_dict):
    """
    使用 LLM 从 Agent 的自然语言回答中提取选项字母 (A/B/C/D)。
    """
    extract_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = ChatPromptTemplate.from_template("""
    你是一个答案提取助手。下面是一道选择题、选项以及模型的回答。
    请判断模型最终选择了哪个选项。
    
    题目: {question}
    选项: {options}
    模型回答: {prediction}
    
    请只输出一个大写字母（A, B, C, 或 D）。如果模型没有做出明确选择或回答无关，请输出 "Unknown"。
    不要输出任何解释，只输出字母。
    """)
    
    try:
        chain = prompt | extract_llm
        res = chain.invoke({
            "question": question,
            "options": str(options_dict),
            "prediction": prediction
        })
        return res.content.strip().upper()
    except:
        return "Unknown"

# ============================
# 2. Agent 统一调用接口
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
# 3. 主程序
# ============================

def main():
    # 1. 加载选择题数据集
    data_file = "option_konwledge.json" 
    if not os.path.exists(data_file):
        print(f"❌ 找不到数据文件: {data_file}")
        return

    with open(data_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    # ⚠️ 调试模式：只跑前 5 题，全量跑请注释下面这行
    # dataset = dataset[:5] 
    
    print(f"🚀 开始评估 {len(dataset)} 道选择题 (仅计算准确率)...")
    
    results = []

    for item in tqdm(dataset):
        # 准备输入
        query_text = format_mcq_query(item)
        correct_opt = item["correct_answer"]
        doc_id = item.get("id", "N/A")
        
        row_data = {"id": doc_id, "Correct_Answer": correct_opt}
        
        for agent_name, run_func in AGENTS.items():
            # A. 运行 Agent
            start_t = time.time()
            raw_prediction = run_func(query_text)
            cost_time = time.time() - start_t
            
            # B. 提取选项 (Extractor)
            extracted_opt = extract_option_with_llm(item["question"], raw_prediction, item["options"])
            
            # C. 判断正误 (Accuracy)
            is_correct = 1 if extracted_opt == correct_opt else 0
            
            # D. 记录数据
            row_data[f"{agent_name}_Pred"] = extracted_opt     # 预测选项
            row_data[f"{agent_name}_Correct"] = is_correct     # 0或1
            row_data[f"{agent_name}_Time"] = round(cost_time, 2)
            
        results.append(row_data)

    # 4. 生成 DataFrame 并计算平均值
    df = pd.DataFrame(results)
    
    # 计算数值列平均值
    means = df.select_dtypes(include=[np.number]).mean()
    
    summary_row = {"id": "AVERAGE"}
    for col in means.index:
        summary_row[col] = round(means[col], 4)
        
    df_final = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

    # 5. 保存 CSV
    output_csv = "mcq_accuracy_only.csv"
    df_final.to_csv(output_csv, index=False, encoding="utf_8_sig")
    
    print("\n" + "="*30)
    print(f"✅ 评估完成！结果已保存至: {output_csv}")
    print("="*30)
    
    # 打印准确率预览
    print("各模型平均准确率 (Correct):")
    acc_cols = [c for c in df_final.columns if "Correct" in c and "Answer" not in c]
    print(df_final[acc_cols].tail(1).to_string(index=False))

if __name__ == "__main__":
    main()