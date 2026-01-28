import os
import math
import ast
import operator
from typing import List, TypedDict, Optional, Dict, Any, Union

# ----------------- LangChain / LangGraph Imports -----------------
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from langgraph.graph import StateGraph, END

# ----------------- 系统与环境配置 -----------------
from dotenv import load_dotenv, find_dotenv

# 加载环境变量
load_dotenv(find_dotenv())
openai_api_key = os.environ.get("OPENAI_API_KEY")
openai_api_base = os.environ.get("OPENAI_API_BASE")
serper_api_key = os.environ.get("SERPER_API_KEY")

# ----------------- 用户自定义工具导入 -----------------
# 注意：这里直接使用您提供的检索器
try:
    from documents.retriever_tools import ensemble_retriever
    print("✅ 成功加载本地 RAG 检索器: ensemble_retriever")
except ImportError:
    raise ImportError("❌ 未找到 documents.retriever_tools 模块，请确保文件路径正确。")

# ----------------- 全局工具初始化 -----------------
try:
    # 使用 gpt-4o-mini，temperature 设置为 1.0 以增加树搜索的创造性（MCTS 需要一定的随机性来扩展不同路径）
    # 如果您需要极其严格的输出，可以在生成最终答案时调整回 0
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0.7, 
        openai_api_key=openai_api_key, 
        openai_api_base=openai_api_base
    )
    
    # 网页搜索工具
    web_search_tool = GoogleSerperAPIWrapper(api_key=serper_api_key)
    print("✅ 成功初始化 Google Serper 工具")
    
except Exception as e:
    print(f"❌ 工具初始化失败: {e}")
    exit(1)


# ==========================================
# 1. MCTS 核心数据结构 (Node & State)
# ==========================================

class MCTSNode:
    """蒙特卡洛树搜索节点"""
    def __init__(self, parent=None, action_thought: str = "Root"):
        self.parent: Optional[MCTSNode] = parent
        self.children: List[MCTSNode] = []
        
        # 节点属性
        self.action_thought = action_thought  # 当前节点的“想法”或“子查询”
        self.content: str = ""                # 执行检索/搜索后的结果内容
        self.visits: int = 0                  # 访问次数 N
        self.value_sum: float = 0.0           # 累计分数 V
        self.depth: int = 0 if parent is None else parent.depth + 1
        
    @property# 计算平均价值，通过累计分数 V 和访问次数 N 计算
    def value(self) -> float:
        """平均价值 Q(s, a)"""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def uct_score(self, parent_visits: int, c_puct: float = 1.41) -> float:
        """计算 UCT (Upper Confidence Bound for Trees) 分数"""
        if self.visits == 0:
            return float('inf')  # 优先访问未访问过的节点
        
        q_value = self.value
        # U 项：探索因子
        u_value = c_puct * math.sqrt(math.log(parent_visits) / (1 + self.visits))
        return q_value + u_value

    def add_child(self, child_node):
        self.children.append(child_node)

    def __repr__(self):
        return f"<Node depth={self.depth} visits={self.visits} val={self.value:.2f} thought='{self.action_thought}'>"

class TreeState(TypedDict):
    """LangGraph 状态定义"""
    original_query: str     # 用户原始问题
    root: MCTSNode          # 树根
    current_node: MCTSNode  # 当前正在处理的节点
    iterations: int         # 当前迭代轮数
    max_iterations: int     # 最大迭代轮数（预算）
    best_answer: str        # 最终生成的答案

# ==========================================
# 2. LATS 核心节点逻辑 (Nodes)
# ==========================================

def initial_node(state: TreeState):
    """【初始化】创建根节点"""
    print(f"\n🌱 [Start] 初始化 LATS 树搜索，原问题: {state['original_query']}")
    root = MCTSNode(action_thought=state['original_query'])
    # 根节点不需要内容，只是起点
    return {"root": root, "current_node": root, "iterations": 0}

def selection_node(state: TreeState):
    """【选择】基于 UCT 选择最有潜力的叶子节点"""
    root = state["root"]
    node = root
    
    # 贪婪选择直到叶子节点（没有孩子的节点）
    while node.children:
        # 选择 UCT 分数最高的子节点
        node = max(node.children, key=lambda c: c.uct_score(node.visits))
    
    return {"current_node": node}

def expansion_node(state: TreeState):
    """【扩展】LLM 生成新的子查询或思考步骤"""
    current_node = state["current_node"]
    query = current_node.action_thought
    
    # 如果节点已经访问过（simulation过），或者它是根节点，我们需要扩展出新的子节点
    # 限制深度，防止无限递归，例如深度超过3就不再扩展
    if (current_node.visits > 0 or current_node.depth == 0) and current_node.depth < 3:
        print(f"🌲 [Expand] 正在扩展节点: '{query[:20]}...'")
        
        prompt = f"""
        你是一个专家研究员。针对问题: "{query}"
        请生成 2 到 3 个 不同的、具体的搜索子查询，以便从不同角度获取信息来回答原问题。
        
        要求：
        1. 子查询应该互补，覆盖不同方面（例如：定义、最新数据、技术细节）。
        2. 严格返回 Python 列表格式字符串。
        
        示例格式: ["查询A", "查询B", "查询C"]
        """
        try:
            response = llm.invoke(prompt).content
            # 简单的解析逻辑
            start = response.find('[')
            end = response.rfind(']') + 1
            sub_queries = ast.literal_eval(response[start:end])
        except Exception as e:
            print(f"⚠️ 解析扩展查询失败，使用默认策略: {e}")
            sub_queries = [f"{query} details", f"{query} statistics"]

        # 创建子节点并挂载
        if not sub_queries:
            sub_queries = [query] # Fallback

        for q in sub_queries:
            child = MCTSNode(parent=current_node, action_thought=q)
            current_node.add_child(child)
        
        # 扩展后，立即选择第一个新子节点进入 Simulation
        if current_node.children:
            return {"current_node": current_node.children[0]}
    
    # 如果无法扩展或无需扩展，保持当前节点
    return {"current_node": current_node}

def simulation_node(state: TreeState):
    """【模拟】执行真实的 RAG 和 Web Search"""
    node = state["current_node"]
    
    # 如果节点已有内容（已被模拟过），则跳过
    if node.content:
        return {"current_node": node}
        
    query = node.action_thought
    print(f"🔍 [Simulate] 执行真实检索: {query}")
    
    combined_content = ""
    
    # 1. 执行 RAG 检索 (Local Knowledge)
    try:
        rag_docs = ensemble_retriever.invoke(query)
        if rag_docs:
            rag_text = "\n".join([d.page_content for d in rag_docs[:2]]) # 取前2条最相关的
            combined_content += f"【本地知识库】:\n{rag_text}\n"
            print(f"   -> RAG 检索到 {len(rag_docs)} 条文档")
    except Exception as e:
        print(f"   -> RAG 检索出错: {e}")

    # 2. 执行 Web Search (External Knowledge)
    # LATS 的优势：可以同时结合本地和网络
    try:
        web_res = web_search_tool.results(query)
        if "organic" in web_res:
            web_text = ""
            for item in web_res["organic"][:2]: # 取前2条
                web_text += f"- {item.get('title')}: {item.get('snippet')}\n"
            combined_content += f"【网络搜索】:\n{web_text}\n"
            print(f"   -> Web 搜索到 {len(web_res.get('organic', []))} 条结果")
    except Exception as e:
        print(f"   -> Web 搜索出错: {e}")

    if not combined_content:
        combined_content = "未找到相关信息。"

    node.content = combined_content
    return {"current_node": node}

def evaluation_node(state: TreeState):
    """【评估 & 反向传播】LLM 对当前节点内容打分，并更新路径"""
    node = state["current_node"]
    original_query = state["original_query"]
    content = node.content
    
    print(f"⚖️ [Evaluate] 正在评估节点质量...")
    
    # 构造评分 Prompt
    eval_prompt = f"""
    用户问题: {original_query}
    当前节点的子查询: {node.action_thought}
    检索到的内容:
    {content[:2000]} (截取)
    
    请对上述内容对于回答用户问题的有用性进行打分 (0.0 到 1.0)。
    1.0 表示完美包含答案，0.0 表示完全无关。
    
    请只返回一个数字，例如: 0.8
    """
    
    score = 0.5 # 默认中立分
    try:
        response = llm.invoke(eval_prompt).content.strip()
        # 提取数字
        import re
        match = re.search(r"0\.\d+|1\.0|0|1", response)
        if match:
            score = float(match.group())
    except Exception as e:
        print(f"   -> 评分失败，使用默认分: {e}")

    print(f"   -> 评分结果: {score}")

    # --- Backpropagation (反向传播) ---
    # 从当前节点一直回溯到根节点，更新 visits 和 value_sum
    temp_node = node
    while temp_node:
        temp_node.visits += 1
        temp_node.value_sum += score
        temp_node = temp_node.parent
        
    return {"iterations": state["iterations"] + 1}

def generation_node(state: TreeState):
    """【生成】汇总最佳路径，生成最终答案"""
    print("\n✍️ [Generate] 搜索结束，正在生成最终回答...")
    root = state["root"]
    original_query = state["original_query"]
    
    # 策略：收集树中 visits 次数最多的路径（或者分数最高的路径）
    # 这里我们遍历一层，把所有探索过的内容都作为上下文（只要分数尚可）
    
    all_contexts = []
    
    def collect_contexts(node: MCTSNode):
        # 简单的遍历收集逻辑，收集 value > 0.4 的节点内容
        if node.visits > 0 and node.value > 0.4 and node.content:
             all_contexts.append(f"子查询: {node.action_thought}\n内容: {node.content}")
        for child in node.children:
            collect_contexts(child)
            
    collect_contexts(root)
    
    # 去重并拼接
    unique_contexts = "\n---\n".join(list(set(all_contexts))[:5]) # 限制长度防止 Context Window 溢出
    
    if not unique_contexts:
        unique_contexts = "未检索到有效信息。"

    final_prompt = f"""
    你是一个半导体行业专家。请基于以下经过验证的多步检索信息，回答用户问题。
    
    用户问题: {original_query}
    
    === 检索上下文 ===
    {unique_contexts}
    ==================
    
    请输出逻辑清晰、引用明确的最终回答：
    """
    
    final_answer = llm.invoke(final_prompt).content
    return {"best_answer": final_answer}

# ==========================================
# 3. 路由逻辑 (Conditional Edges)
# ==========================================

def should_continue(state: TreeState):
    """判断是继续搜索还是生成答案"""
    if state["iterations"] < state["max_iterations"]:
        return "selection"
    return "generate"

# ==========================================
# 4. 构建 LangGraph 工作流
# ==========================================

workflow = StateGraph(TreeState)

# 添加节点
workflow.add_node("initial", initial_node)
workflow.add_node("selection", selection_node)
workflow.add_node("expansion", expansion_node)
workflow.add_node("simulation", simulation_node)
workflow.add_node("evaluation", evaluation_node)
workflow.add_node("generate", generation_node)

# 设置边
workflow.set_entry_point("initial")
workflow.add_edge("initial", "selection")
workflow.add_edge("selection", "expansion")
workflow.add_edge("expansion", "simulation")
workflow.add_edge("simulation", "evaluation")

# 条件边：Evaluation 结束后，判断是否达到最大迭代次数
workflow.add_conditional_edges(
    "evaluation",
    should_continue,
    {
        "selection": "selection", # 循环继续搜索
        "generate": "generate"    # 结束搜索，生成答案
    }
)

workflow.add_edge("generate", END)

# 编译应用
app = workflow.compile()

# ==========================================
# 5. 主程序执行入口
# ==========================================

if __name__ == "__main__":
    # 测试问题：这是一个复杂问题，单次检索可能不全，适合 MCTS
    query = "2024年台积电的最新3nm产能情况如何？苹果和英伟达的订单占比大概是多少？"
    
    print(f"🚀 [Agent Start] 启动 LATS Agentic RAG")
    print(f"❓ 问题: {query}\n" + "="*50)
    
    # 这里的 max_iterations 决定了搜索的广度和深度（尝试多少次节点扩展）
    # 建议设置为 5-10 次，为了演示速度这里设为 4
    inputs = {
        "original_query": query, 
        "max_iterations": 10
    }
    
    try:
        final_state = app.invoke(inputs)
        
        print("\n" + "="*50)
        print("✅ [Done] 最终答案：\n")
        print(final_state["best_answer"])
        print("\n" + "="*50)
        
    except Exception as e:
        print(f"\n❌ 执行出错: {e}")
        import traceback
        traceback.print_exc()