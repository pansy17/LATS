import os
import math
import ast
import operator
import re
from typing import List, TypedDict, Optional, Dict, Any, Union

# ----------------- 依赖库导入 -----------------
# LangChain/LangGraph: 用于构建 Agent 工作流和处理文档
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from langgraph.graph import StateGraph, END

# ----------------- 系统与环境配置 -----------------
from dotenv import load_dotenv, find_dotenv

# 加载 .env 文件中的 API Key，确保安全性
load_dotenv(find_dotenv())
openai_api_key = os.environ.get("OPENAI_API_KEY")
openai_api_base = os.environ.get("OPENAI_API_BASE")
serper_api_key = os.environ.get("SERPER_API_KEY")

# ----------------- 用户自定义工具导入 -----------------
# 尝试加载本地定义的混合检索器 (Ensemble Retriever)
# 如果本地环境缺失该文件，为了保证代码可运行，定义一个 Mock (模拟) 检索器
try:
    from documents.retriever_tools import ensemble_retriever
    print("✅ 成功加载本地 RAG 检索器: ensemble_retriever")
except ImportError:
    print("⚠️ 未找到本地检索器，使用 Mock 替代")
    class MockRetriever:
        def invoke(self, query):
            # 模拟返回一个包含元数据的 Document 对象
            return [Document(page_content=f"模拟检索内容: 关于 {query} 的本地技术文档数据...", metadata={"source": "local"})]
    ensemble_retriever = MockRetriever()

# ----------------- 全局模型与工具初始化 -----------------
try:
    # 初始化 LLM
    # temperature=0.7: 在创造性(扩展子查询)和严谨性之间的一般平衡
    # 后续在特定节点(如 Simulation)会动态调整 temp 为 0 以保证事实性
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0.7, 
        openai_api_key=openai_api_key, 
        openai_api_base=openai_api_base
    )
    
    # 初始化 Google 搜索工具 (用于补充外部知识)
    web_search_tool = GoogleSerperAPIWrapper(api_key=serper_api_key)
    print("✅ 成功初始化 Google Serper 工具")
    
except Exception as e:
    print(f"❌ 工具初始化失败: {e}")
    exit(1)


# ==========================================
# 1. MCTS 核心数据结构 (Node & State)
# ==========================================

class MCTSNode:
    """
    蒙特卡洛树搜索 (MCTS) 的节点类。
    每个节点代表搜索树中的一个状态（一个具体的子查询或思考步骤）。
    """
    def __init__(self, parent=None, action_thought: str = "Root"):
        self.parent: Optional[MCTSNode] = parent  # 父节点指针，用于反向传播
        self.children: List[MCTSNode] = []        # 子节点列表
        
        # --- 节点内容属性 ---
        self.action_thought = action_thought  # 当前节点的意图（即生成的子查询）
        self.content: str = ""                # 检索回来的原始文本（上下文）
        self.generated_answer: str = ""       # 基于上下文生成的中间答案（带 [Ref] 引用）
        
        # --- MCTS 统计属性 ---
        self.visits: int = 0                  # N: 被访问/模拟的次数
        self.value_sum: float = 0.0           # V: 累计获得的奖励分数总和
        self.depth: int = 0 if parent is None else parent.depth + 1 # 树深度
        
    @property#这个属性用于获取节点的评估值，即节点的均值。
    def value(self) -> float:
        """计算平均价值 Q(s, a) = V / N"""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def uct_score(self, parent_visits: int, c_puct: float = 1.41) -> float:
        """
        计算 UCT (Upper Confidence Bound for Trees) 分数。
        公式: Q + C * sqrt(ln(N_parent) / N_child)
        作用: 平衡 '利用'(Exploitation, 选高分节点) 和 '探索'(Exploration, 选少访问节点)
        """
        if self.visits == 0:
            return float('inf')  # 没访问过的节点优先级最高，保证广度覆盖
        
        q_value = self.value # 利用项
        # 探索项: 访问越少，分母越小，U值越大
        u_value = c_puct * math.sqrt(math.log(parent_visits) / (1 + self.visits))
        return q_value + u_value

    def add_child(self, child_node):
        self.children.append(child_node)

    def __repr__(self):
        # 打印节点时的调试信息
        return f"<Node depth={self.depth} visits={self.visits} val={self.value:.2f} thought='{self.action_thought[:10]}...'>"

class TreeState(TypedDict):
    """
    LangGraph 的全局状态定义。
    在图的各个节点之间传递数据。
    """
    original_query: str     # 用户最初输入的问题
    normalized_query: str   # 经过归一化处理后的标准查询（如将 stm32 转为 STM32）
    root: MCTSNode          # 整个搜索树的根节点
    current_node: MCTSNode  # 当前工作流正在处理的节点指针
    iterations: int         # 当前已经执行的循环次数
    max_iterations: int     # 最大允许循环次数（预算控制）
    best_answer: str        # 最终生成的 Markdown 研报

# ==========================================
# 2. LATS 核心节点逻辑 (Nodes)
# ==========================================

def normalize_node(state: TreeState):
    """
    【预处理节点】：查询归一化
    目的：解决半导体领域术语书写不规范导致的检索漏召回问题。
    策略：正则匹配 (规则) + LLM 重写 (语义)。
    """
    query = state["original_query"]
    print(f"\n🔧 [Normalize] 正在标准化查询: {query}")

    # 1. 规则库归一化 (Regex/Dict) - 快速处理常见别名
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

    # 2. LLM 语义重写 - 处理复杂语义和实体补全
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
    """
    【初始化节点】：创建树根
    使用归一化后的查询作为 MCTS 的起点。
    """
    q = state.get("normalized_query", state["original_query"])
    print(f"\n🌱 [Start] 初始化 LATS 树搜索，基准查询: {q}")
    
    root = MCTSNode(action_thought=q)
    # 初始化状态：当前节点指向 Root，迭代计数归零
    return {"root": root, "current_node": root, "iterations": 0}

def selection_node(state: TreeState):
    """
    【选择节点 (Selection)】
    基于 UCT 算法，从 Root 开始一直往下走，直到找到一个'最有潜力'的叶子节点。
    """
    root = state["root"]
    node = root
    
    # 贪婪选择：只要有孩子，就选 UCT 分数最高的那个孩子往下走
    while node.children:
        node = max(node.children, key=lambda c: c.uct_score(node.visits))
    
    # 返回选中的节点，准备对其进行扩展
    return {"current_node": node}

def expansion_node(state: TreeState):
    """
    【扩展节点 (Expansion)】
    如果当前节点还没被访问过，或者已经访问过但需要更多信息，就扩展它。
    LLM 负责生成 2-3 个具体的子查询。
    """
    current_node = state["current_node"]
    query = current_node.action_thought
    
    # 防止无限递归：限制树的最大深度为 3
    if (current_node.visits > 0 or current_node.depth == 0) and current_node.depth < 3:
        print(f"🌲 [Expand] 正在扩展节点: '{query[:20]}...'")
        
        # --- Prompt 优化：注入行业知识 ---
        # 强制要求生成带有 WPM (产能单位)、区分营收/产能的查询
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
            # 解析 LLM 返回的 List 字符串
            start = response.find('[')
            end = response.rfind(']') + 1
            sub_queries = ast.literal_eval(response[start:end])
        except Exception as e:
            print(f"⚠️ 解析扩展查询失败: {e}")
            sub_queries = [f"{query} WPM details", f"{query} market share report"]

        if not sub_queries:
            sub_queries = [query]

        # 将生成的子查询挂载为当前节点的子节点
        for q in sub_queries:
            child = MCTSNode(parent=current_node, action_thought=q)
            current_node.add_child(child)
        
        # 扩展完后，立刻选择第一个新生成的子节点进入下一步(Simulation)
        if current_node.children:
            return {"current_node": current_node.children[0]}
    
    return {"current_node": current_node}

def simulation_node(state: TreeState):
    """
    【模拟节点 (Simulation)】
    执行真实的检索动作，并进行'事实锚定'。
    这里包含了 RAG 检索、Web 搜索和中间答案生成。
    """
    node = state["current_node"]
    
    # 缓存机制：如果该节点已经生成过答案，直接跳过
    if node.generated_answer:
        return {"current_node": node}
        
    query = node.action_thought
    print(f"🔍 [Simulate] 执行真实检索: {query}")
    
    raw_docs = [] 
    
    # 1. 本地知识库检索 (RAG)
    try:
        rag_docs = ensemble_retriever.invoke(query)
        if rag_docs:
            for d in rag_docs[:2]:
                source = d.metadata.get("source", "Local Doc")
                raw_docs.append(f"【本地-{source}】: {d.page_content}")
    except Exception as e:
        print(f"   -> RAG 错误: {e}")

    # 2. 网络搜索 (Google Serper)
    try:
        web_res = web_search_tool.results(query)
        if "organic" in web_res:
            for item in web_res["organic"][:2]:
                raw_docs.append(f"【网络-{item.get('title')}】: {item.get('snippet')}")
    except Exception as e:
        print(f"   -> Web 错误: {e}")

    # 3. 格式化上下文并编号，用于引用
    if not raw_docs:
        formatted_context = "未找到任何相关信息。"
        node.content = formatted_context
    else:
        formatted_parts = []
        for i, doc in enumerate(raw_docs):
            formatted_parts.append(f"[Ref: {i}] {doc}")
        formatted_context = "\n\n".join(formatted_parts)
        node.content = formatted_context

    # 4. 生成带引用的中间答案 (关键优化)
    # 使用 temperature=0 强制模型严格遵守引用规则和拒答机制
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

def evaluation_node(state: TreeState):
    """
    【评估节点 (Evaluation) & 反向传播 (Backpropagation)】
    对生成内容的质量进行打分，并将分数回传给所有父节点。
    """
    node = state["current_node"]
    original_query = state["original_query"]
    answer = node.generated_answer
    
    print(f"⚖️ [Evaluate] 正在评估节点质量...")
    
    score = 0.0
    
    # --- 评分策略 ---
    
    # 规则 1: 拒答处理 (给低分但不是0分，鼓励诚实)
    if "无法回答" in answer or "No information" in answer:
        print("   -> 检测到拒答，给予低分奖励 (0.1) 以鼓励诚实。")
        score = 0.1
    
    # 规则 2: 引用检查 (事实锚定验证)
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
        
    # 规则 3: 无引用惩罚 (潜在幻觉)
    else:
        print("   -> ⚠️ 警告: 回答未包含引用，判定为潜在幻觉，给予 0 分。")
        score = 0.0

    # --- 反向传播 (Backpropagation) ---
    # 核心逻辑：将分数加到当前节点及其所有祖先节点的 value_sum 中
    # 这会影响后续 Selection 阶段 UCT 分数的计算
    temp_node = node
    while temp_node:
        temp_node.visits += 1
        temp_node.value_sum += score
        temp_node = temp_node.parent
        
    return {"iterations": state["iterations"] + 1}

def generation_node(state: TreeState):
    """
    【生成节点 (Generation)】
    在所有迭代结束后，收集高分路径的信息，汇总生成最终研报。
    """
    print("\n✍️ [Generate] 搜索结束，正在生成最终回答...")
    root = state["root"]
    original_query = state["original_query"]
    
    all_contexts = []
    
    # 递归收集高价值信息
    def collect_contexts(node: MCTSNode):
        # 过滤条件：必须访问过、得分较高 (>0.4)、且不是拒答内容
        if node.visits > 0 and node.value > 0.4 and node.generated_answer:
            if "无法回答" not in node.generated_answer:
                all_contexts.append(f"【来源: {node.action_thought}】\n{node.generated_answer}")
        
        for child in node.children:
            collect_contexts(child)
            
    collect_contexts(root)
    
    # 去重并拼接，限制上下文长度
    unique_contexts = "\n\n".join(list(set(all_contexts))[:10])
    
    if not unique_contexts:
        unique_contexts = "未检索到足够的可信信息。"

    # 最终生成的 Prompt：要求结构化、研报风格
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
# 3. 路由逻辑 (Control Flow)
# ==========================================

def should_continue(state: TreeState):
    """
    条件边逻辑：判断是继续迭代还是结束。
    由 state['max_iterations'] 决定循环次数。
    """
    if state["iterations"] < state["max_iterations"]:
        return "selection"  # 没达到最大次数，回到 Selection 节点继续搜索
    return "generate"       # 达到最大次数，进入 Generate 节点输出结果

# ==========================================
# 4. 构建 LangGraph 工作流
# ==========================================

workflow = StateGraph(TreeState)

# --- 添加所有节点 ---
workflow.add_node("normalize", normalize_node) # 入口：预处理
workflow.add_node("initial", initial_node)     # 初始化
workflow.add_node("selection", selection_node) # 循环起点
workflow.add_node("expansion", expansion_node)
workflow.add_node("simulation", simulation_node)
workflow.add_node("evaluation", evaluation_node) # 循环终点
workflow.add_node("generate", generation_node)   # 出口：生成

# --- 设置边的连接关系 ---
# 1. 设置入口
workflow.set_entry_point("normalize")

# 2. 线性流程
workflow.add_edge("normalize", "initial")
workflow.add_edge("initial", "selection")
workflow.add_edge("selection", "expansion")
workflow.add_edge("expansion", "simulation")
workflow.add_edge("simulation", "evaluation")

# 3. 循环控制 (Conditional Edge)
# 在 evaluation 结束后，根据 should_continue 的返回值决定去向
workflow.add_conditional_edges(
    "evaluation",
    should_continue,
    {
        "selection": "selection", # 继续循环
        "generate": "generate"    # 结束循环
    }
)

workflow.add_edge("generate", END)

# 编译图
app = workflow.compile()

# ==========================================
# 5. 主程序执行入口
# ==========================================

if __name__ == "__main__":
    # 测试 Query：包含一些非标准写法 (tsmc, 3nm)
    query = "2024年tsmc的最新3nm产能情况如何？苹果和nv的订单占比大概是多少？"
    
    print(f"🚀 [Agent Start] 启动 MCTS Agent (芯知 - 专业半导体版)")
    print(f"❓ 原始问题: {query}\n" + "="*50)
    
    inputs = {
        "original_query": query, 
        # 设置最大迭代次数，建议 5-10 次以保证搜索覆盖度
        "max_iterations": 6 
    }
    
    # 设置递归限制，防止 GraphRecursionError
    config = {"recursion_limit": 50} 
    
    try:
        # 启动图执行
        final_state = app.invoke(inputs, config=config)
        
        print("\n" + "="*50)
        print("✅ [Done] 最终答案：\n")
        print(final_state["best_answer"])
        print("\n" + "="*50)
        
    except Exception as e:
        print(f"\n❌ 执行出错: {e}")
        import traceback
        traceback.print_exc()