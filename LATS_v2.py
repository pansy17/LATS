import os
import sys

# ==============================================================================
# 🚨 核心修正：环境变量必须在导入 LangChain 组件之前配置，否则追踪可能失效
# ==============================================================================
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "xxxx" 
# ⚠️ 注意：修复了原代码中的中文引号和换行符，请填入你的真实 Key
os.environ["LANGCHAIN_API_KEY"] = "xxxx" 

# ----------------- 标准库导入 -----------------
import math
import ast
import operator
import re
import uuid
from typing import List, TypedDict, Optional, Dict, Any, Union

# ----------------- LangSmith 导入 -----------------
# 用于给函数加装饰器，实现细粒度追踪
from langsmith import traceable

# ----------------- LangChain / LangGraph 导入 -----------------
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig 

# ----------------- 环境配置 -----------------
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# 获取其他 API Key
openai_api_key = os.environ.get("OPENAI_API_KEY")
openai_api_base = os.environ.get("OPENAI_API_BASE")
serper_api_key = os.environ.get("SERPER_API_KEY")

# ----------------- 用户自定义工具导入 -----------------
try:
    from documents.retriever_tools import ensemble_retriever
    print("✅ 成功加载本地 RAG 检索器: ensemble_retriever")
except ImportError:
    print("⚠️ 未找到本地检索器，使用 Mock 替代")
    class MockRetriever:
        def invoke(self, query):
            return [Document(page_content=f"模拟检索内容: 关于 {query} 的本地技术文档数据...", metadata={"source": "local"})]
    ensemble_retriever = MockRetriever()

# ----------------- 全局工具初始化 -----------------
try:
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0.7, 
        openai_api_key=openai_api_key, 
        openai_api_base=openai_api_base
    )
    
    web_search_tool = GoogleSerperAPIWrapper(api_key=serper_api_key)
    print("✅ 成功初始化 Google Serper 工具")
    
except Exception as e:
    print(f"❌ 工具初始化失败: {e}")
    # 为了演示不直接退出，实际使用建议保留 exit(1)
    # exit(1) 


# ==========================================
# 1. MCTS 核心数据结构 (Node & State)
# ==========================================

class MCTSNode:
    """蒙特卡洛树搜索节点"""
    def __init__(self, parent=None, action_thought: str = "Root"):
        self.parent: Optional[MCTSNode] = parent
        self.children: List[MCTSNode] = []
        
        # 节点属性
        # 当前节点对应的“动作/思考”文本，用于描述该节点要解决的问题或子查询
        self.action_thought = action_thought  
        # 存储从检索工具（RAG/Web）返回的原始上下文内容，供后续生成答案使用
        self.content: str = ""                
        # 基于 content 生成的带引用中间答案，也是最终答案的候选片段
        self.generated_answer: str = ""       
        
        # MCTS 统计：节点在模拟阶段被访问的次数，用于计算 UCT 值
        self.visits: int = 0                  
        # MCTS 统计：节点累计获得的奖励/评分总和，用于计算平均价值
        self.value_sum: float = 0.0           
        # 节点在树中的深度，根节点为 0，每下降一层加 1，用于限制搜索深度
        self.depth: int = 0 if parent is None else parent.depth + 1
        
    @property
    def value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def uct_score(self, parent_visits: int, c_puct: float = 1.41) -> float:
        if self.visits == 0:
            return float('inf') 
        
        q_value = self.value
        u_value = c_puct * math.sqrt(math.log(parent_visits) / (1 + self.visits))
        return q_value + u_value

    def add_child(self, child_node):
        self.children.append(child_node)

    def __repr__(self):
        return f"<Node depth={self.depth} visits={self.visits} val={self.value:.2f} thought='{self.action_thought[:10]}...'>"

class TreeState(TypedDict):
    """LangGraph 状态定义"""
    original_query: str     # 用户原始问题
    normalized_query: str   # 归一化后的标准查询
    root: MCTSNode          # 树根
    current_node: MCTSNode  # 当前正在处理的节点
    iterations: int         # 当前迭代轮数
    max_iterations: int     # 最大迭代轮数（预算）
    best_answer: str        # 最终生成的答案

# ==========================================
# 2. LATS 核心节点逻辑 (Nodes)
# ==========================================

@traceable(name="Normalize Query") # ✅ 添加追踪装饰器
def normalize_node(state: TreeState):
    """【查询归一化】"""
    query = state["original_query"]
    print(f"\n🔧 [Normalize] 正在标准化查询: {query}")

    # 1. 规则库归一化 (Regex/Dict)
    term_mapping = {
        r"(?i)stm\s*32": "STM32",
        r"(?i)iic": "I2C",
        r"(?i)spi": "SPI",
        r"(?i)uart": "UART",
        r"(?i)mcu": "MCU",
        r"(?i)datasheet": "Data Sheet",
        r"(?i)spec\s*sheet": "Specification",
        r"(?i)tsmc": "TSMC",
        r"(?i)smic": "SMIC",
        r"(?i)nv(idia)?": "NVIDIA",
        r"(?i)3\s*nm": "3nm",
    }
    
    normalized_query = query
    for pattern, replacement in term_mapping.items():
        normalized_query = re.sub(pattern, replacement, normalized_query)

    # 2. LLM 语义重写
    prompt = f"""
    你是一个半导体领域的术语专家。请对以下用户查询进行标准化处理。
    
    原始查询: "{normalized_query}"
    
    任务：
    1. **纠正拼写**：修复明显的拼写错误。
    2. **术语规范**：将非标准描述转换为行业通用术语（如 "3纳米工艺" -> "3nm Process Node"）。
    3. **实体补全**：如果实体名称模糊，尝试补全（如 "A17" -> "Apple A17 Pro"），但不要改变原意。
    4. **去口语化**：去除无意义的语气词，保留核心搜索意图。
    
    请直接输出标准化后的查询字符串，不要包含任何解释或引号。
    """
    
    try:
        response = llm.invoke(prompt).content.strip()
        final_query = response.strip('"').strip("'")
    except Exception as e:
        print(f"⚠️ 归一化失败，使用规则处理结果: {e}")
        final_query = normalized_query

    print(f"   -> 标准化结果: {final_query}")
    return {"normalized_query": final_query}


def initial_node(state: TreeState):
    """【初始化】创建根节点"""
    q = state.get("normalized_query", state["original_query"])
    print(f"\n🌱 [Start] 初始化 LATS 树搜索，基准查询: {q}")
    root = MCTSNode(action_thought=q)
    return {"root": root, "current_node": root, "iterations": 0}

@traceable(name="MCTS Selection") # ✅ 添加追踪装饰器
def selection_node(state: TreeState):
    """【选择】基于 UCT 选择最有潜力的叶子节点"""
    root = state["root"]
    node = root
    
    # 贪婪选择直到叶子节点
    while node.children:
        node = max(node.children, key=lambda c: c.uct_score(node.visits))
    
    return {"current_node": node}

@traceable(name="MCTS Expansion") # ✅ 添加追踪装饰器
def expansion_node(state: TreeState):
    """【扩展】LLM 生成新的子查询"""
    current_node = state["current_node"]
    query = current_node.action_thought
    
    # 限制深度与扩展条件
    if (current_node.visits > 0 or current_node.depth == 0) and current_node.depth < 3:
        print(f"🌲 [Expand] 正在扩展节点: '{query[:20]}...'")
        
        prompt = f"""
        你是一位资深的半导体行业情报分析师。针对问题: "{query}"
        请生成 2 到 3 个 极具针对性的搜索子查询，旨在挖掘深层数据。
        
        要求：
        1. **精确化度量衡**：如果涉及产能，必须包含 "WPM" (Wafers Per Month), "capacity utilization" (产能利用率) 等关键词。
        2. **区分概念**：明确区分 "Revenue Share" (营收占比) 和 "Wafer Allocation" (晶圆配额/产能占比)，避免混淆。
        3. **具体工艺节点**：对于先进制程，尝试加入具体代号（如 TSMC N3, N3E, N3B, N3P）。
        4. **权威来源导向**：可以加上 "report", "TrendForce", "Digitimes" 等关键词以引导搜索高质量研报。
        
        请严格返回 Python 列表格式字符串。
        示例: ["TSMC N3 capacity WPM 2024 forecast", "Apple vs Nvidia TSMC 3nm wafer allocation 2024"]
        """
        try:
            response = llm.invoke(prompt).content
            start = response.find('[')
            end = response.rfind(']') + 1
            sub_queries = ast.literal_eval(response[start:end])
        except Exception as e:
            print(f"⚠️ 解析扩展查询失败: {e}")
            sub_queries = [f"{query} WPM details", f"{query} market share report"]

        if not sub_queries:
            sub_queries = [query]

        for q in sub_queries:
            child = MCTSNode(parent=current_node, action_thought=q)
            current_node.add_child(child)
        
        if current_node.children:
            return {"current_node": current_node.children[0]}
    
    return {"current_node": current_node}

@traceable(name="Simulation & Fact Check") # ✅ 添加追踪装饰器
def simulation_node(state: TreeState):
    """【模拟 - 事实锚定与拒答机制】"""
    node = state["current_node"]
    
    if node.generated_answer:
        return {"current_node": node}
        
    query = node.action_thought
    print(f"🔍 [Simulate] 执行真实检索: {query}")
    
    raw_docs = [] 
    
    # 1. 执行 RAG 检索
    try:
        rag_docs = ensemble_retriever.invoke(query)
        if rag_docs:
            for d in rag_docs[:2]:
                source = d.metadata.get("source", "Local Doc")
                raw_docs.append(f"【本地-{source}】: {d.page_content}")
    except Exception as e:
        print(f"   -> RAG 错误: {e}")

    # 2. 执行 Web Search
    try:
        web_res = web_search_tool.results(query)
        if "organic" in web_res:
            for item in web_res["organic"][:2]:
                raw_docs.append(f"【网络-{item.get('title')}】: {item.get('snippet')}")
    except Exception as e:
        print(f"   -> Web 错误: {e}")

    # 3. 格式化上下文
    if not raw_docs:
        formatted_context = "未找到任何相关信息。"
        node.content = formatted_context
    else:
        formatted_parts = []
        for i, doc in enumerate(raw_docs):
            formatted_parts.append(f"[Ref: {i}] {doc}")
        formatted_context = "\n\n".join(formatted_parts)
        node.content = formatted_context

    # 4. 生成带引用的中间答案
    print(f"   -> 正在尝试生成中间答案并进行事实锚定...")
    
    anchor_prompt = f"""
    你是一个严谨的半导体产业研究员。请基于下方的【检索上下文】回答问题："{query}"。

    【检索上下文】
    {formatted_context}

    【回答规则】
    1. **事实锚定**：你生成的每一句话，如果引用了上下文，必须在句末标注来源 ID，格式为 [Ref: x]。
    2. **数据敏感性**：
       - 如果找到**具体数字**（如 "60k-70k wpm"），请务必保留并引用。
       - 如果检索到的是**营收占比**（Revenue），严禁将其直接等同于**产能占比**（Capacity/Wafer），必须明确指出是“营收”还是“产能”。
    3. **预测包容性**：如果没有官方披露的精确数据，**允许引用知名分析机构的估算数据**，但必须在回答中明确说明。
    4. **拒答机制**：如果上下文完全没有相关信息，请回答 "无法回答"。

    请生成精炼的回答：
    """
    
    try:
        simulation_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key, openai_api_base=openai_api_base)
        generated_answer = simulation_llm.invoke(anchor_prompt).content
    except Exception as e:
        generated_answer = "无法回答 (生成错误)"

    node.generated_answer = generated_answer
    print(f"   -> 中间答案: {generated_answer[:50]}...")
    
    return {"current_node": node}

@traceable(name="Backpropagation Evaluation") # ✅ 添加追踪装饰器
def evaluation_node(state: TreeState):
    """【评估 & 反向传播】"""
    node = state["current_node"]
    original_query = state["original_query"]
    answer = node.generated_answer
    
    print(f"⚖️ [Evaluate] 正在评估节点质量...")
    
    score = 0.0
    
    # --- 规则 1: 检测拒答 ---
    if "无法回答" in answer or "No information" in answer:
        print("   -> 检测到拒答，给予低分奖励 (0.1) 以鼓励诚实。")
        score = 0.1
    
    # --- 规则 2: 检测引用 (Fact Anchoring Check) ---
    elif "[Ref:" in answer:
        verify_prompt = f"""
        作为半导体数据审核员，请对以下回答进行严格打分 (0.0 到 1.0)。
        
        用户原问题: {original_query}
        当前子查询: {node.action_thought}
        生成的回答: {answer}
        检索到的原文: 
        {node.content[:2000]}
        
        评分标准：
        1. **相关性** (0.4分)：回答是否直接解决了子查询的核心问题？
        2. **真实性** (0.4分)：[Ref: x] 的引用是否真实存在于原文中，且没有歪曲原意？
        3. **精准度** (0.2分)：
           - 如果回答包含了**具体数值**（如 "80k wpm"）而不是模糊描述，加分。
           - 如果回答**混淆了营收（Revenue）和产能（Wafer/Capacity）**，直接扣除 0.5 分（严重错误）。
        
        请综合考虑后，只返回一个数字。
        """
        try:
            response = llm.invoke(verify_prompt).content.strip()
            match = re.search(r"0\.\d+|1\.0|0|1", response)
            if match:
                score = float(match.group())
        except:
            score = 0.5 # 出错回退
        print(f"   -> 引用校验得分: {score}")
        
    # --- 规则 3: 无引用 (Hallucination Risk) ---
    else:
        print("   -> ⚠️ 警告: 回答未包含引用，判定为潜在幻觉，给予 0 分。")
        score = 0.0

    # --- Backpropagation (反向传播) ---
    temp_node = node
    while temp_node:
        temp_node.visits += 1
        temp_node.value_sum += score
        temp_node = temp_node.parent
        
    return {"iterations": state["iterations"] + 1}

@traceable(name="Final Report Generation") # ✅ 添加追踪装饰器
def generation_node(state: TreeState):
    """【生成】汇总高置信度路径"""
    print("\n✍️ [Generate] 搜索结束，正在生成最终回答...")
    root = state["root"]
    original_query = state["original_query"]
    
    all_contexts = []
    
    def collect_contexts(node: MCTSNode):
        if node.visits > 0 and node.value > 0.4 and node.generated_answer:
            if "无法回答" not in node.generated_answer:
                all_contexts.append(f"【来源: {node.action_thought}】\n{node.generated_answer}")
        
        for child in node.children:
            collect_contexts(child)
            
    collect_contexts(root)
    
    unique_contexts = "\n\n".join(list(set(all_contexts))[:10])
    
    if not unique_contexts:
        unique_contexts = "未检索到足够的可信信息。"

    final_prompt = f"""
    你是由字节跳动 RAG 技术支持的半导体首席战略顾问。请基于以下**经过事实核查**的信息，撰写一份关于 "{original_query}" 的简报。
    
    === 经过核查的情报碎片 ===
    {unique_contexts}
    =========================
    
    撰写要求：
    1. **结构化输出**：请使用 Markdown 格式，包含【核心结论】、【数据详解】、【风险/不确定性提示】三个部分。
    2. **数据精准**：优先展示具体数字（如产能 WPM、良率 %）。如果引用的是分析机构（如 TrendForce）的估算值，请明确标注“估算”。
    3. **概念厘清**：在描述占比时，明确区分是“营收贡献占比”还是“晶圆产能占比”，若数据缺失请说明。
    4. **来源标注**：在关键数据后保留 [Ref] 标记（如果有），或说明来源于哪个子查询。
    5. **去伪存真**：如果碎片信息中存在冲突，请对比展示，不要强行合并。
    
    最终简报：
    """
    
    final_answer = llm.invoke(final_prompt).content
    return {"best_answer": final_answer}

# ==========================================
# 3. 路由逻辑
# ==========================================

def should_continue(state: TreeState):
    if state["iterations"] < state["max_iterations"]:
        return "selection"
    return "generate"

# ==========================================
# 4. 构建 LangGraph
# ==========================================

workflow = StateGraph(TreeState)

# 添加节点
workflow.add_node("normalize", normalize_node) 
workflow.add_node("initial", initial_node)
workflow.add_node("selection", selection_node)
workflow.add_node("expansion", expansion_node)
workflow.add_node("simulation", simulation_node)
workflow.add_node("evaluation", evaluation_node)
workflow.add_node("generate", generation_node)

# 设置边
workflow.set_entry_point("normalize")
workflow.add_edge("normalize", "initial")
workflow.add_edge("initial", "selection")
workflow.add_edge("selection", "expansion")
workflow.add_edge("expansion", "simulation")
workflow.add_edge("simulation", "evaluation")

workflow.add_conditional_edges(
    "evaluation",
    should_continue,
    {
        "selection": "selection",
        "generate": "generate"
    }
)

workflow.add_edge("generate", END)
app = workflow.compile()

# ==========================================
# 5. 执行 (带追踪 Metadata)
# ==========================================

if __name__ == "__main__":
    # 测试 Query
    query = "2024年tsmc的最新3nm产能情况如何？苹果和nv的订单占比大概是多少？"
    
    # 生成一个唯一的 run_id 以便在 LangSmith 中通过 ID 搜索（可选）
    run_id = str(uuid.uuid4())
    
    print(f"🚀 [Agent Start] 启动 MCTS Agent (芯知 - 专业半导体版)")
    print(f"📡 LangSmith Trace ID: {run_id}")
    print(f"❓ 原始问题: {query}\n" + "="*50)
    
    inputs = {
        "original_query": query, 
        "max_iterations": 6 
    }
    
    # ===【核心配置】通过 config 传递追踪信息 ===
    config = {
        "recursion_limit": 50,
        "run_name": "MCTS_Semiconductor_Run", # 这将在 LangSmith 列表中显示
        "metadata": {
            "user_id": "test_user_001",
            "environment": "development",
            "algo_type": "MCTS-LATS"
        },
        "configurable": {"thread_id": run_id}
    }
    
    try:
        final_state = app.invoke(inputs, config=config)
        
        print("\n" + "="*50)
        print("✅ [Done] 最终答案：\n")
        print(final_state["best_answer"])
        print("\n" + "="*50)
        
    except Exception as e:
        print(f"\n❌ 执行出错: {e}")
        import traceback
        traceback.print_exc()