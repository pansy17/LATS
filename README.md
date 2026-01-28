# 🧠 Core-RAG: 基于 LATS 策略与混合检索的半导体行业专家 Agent

> **Core-RAG** 是一个高精度的行业研究助手，专为解决复杂、长尾的半导体领域问题而设计。
> 它不仅仅是一个 RAG 系统，更是一个实现了 **LATS (Language Agent Tree Search)** 算法的自主智能体，具备**思维树搜索**、**自我反思**、**术语归一化**和**事实锚定**能力。

---

## 🌟 核心技术亮点 (Key Optimizations)

本项目在标准 RAG 的基础上，针对**数据噪音**、**召回不准**和**大模型幻觉**三大痛点进行了深度优化：

### 1. 数据层：上下文感知的语义分块 (Context-Aware Semantic Chunking)
* **痛点1**：针对传统PDF解析容易断码、无法识别公式等问题，我们使用了“Mineru”实现了更好的PDF解析。
* **痛点2**：传统 RAG 将长文按字符暴力切割，导致“子标题”与“正文”分离，丢失层级上下文。
* **优化方案** (`MarkdownParser.py`)：
* **语义完整性**：采用 `SemanticChunker`（基于 OpenAI Embedding 变化率）而非固定字符数切分，确保切片在语义上的连贯性。
* **父级上下文注入**：实现了 `merge_title_content` 递归算法。在切片时，自动将文档的各级标题（如 `# TSMC -> ## 3nm工艺 -> ### 产能`）注入到正文切片头部。
* **效果**：即使切片是独立的，LLM 也能明确知道该段落属于 "TSMC" 的 "3nm" 章节，极大提升了召回准确率。



### 2. 检索层：稀疏与稠密混合检索 (Hybrid Search)

* **痛点**：Dense Retriever (向量) 擅长语义匹配，但在匹配专有名词（如 "GAA", "N3E"）时往往不如关键词匹配精准。
* **优化方案** (`milvus_db.py`)：
* **双路召回**：同时维护 **Dense Vector** (BGE-Small, 512维) 和 **Sparse Vector** (BM25, 词频权重)。
* **自定义 BM25**：基于 `jieba` 分词手动实现了 BM25 算法，计算 Token 权重并存储为 Milvus 的 `SPARSE_FLOAT_VECTOR`。
* **效果**：在处理半导体型号、工艺代号等精确查询时，召回率显著提升。



### 3. 认知层：LATS 思维树搜索 (Language Agent Tree Search)

* **痛点**：ReAct 或 CoT 是一条路走到黑，一旦中间步出错（如搜索关键词错误），整个回答就会崩塌。
* **优化方案** (`LATS_v3.py`)：
* **MCTS 架构**：构建搜索树，包含 **Selection** (UCT算法选择节点) -> **Expansion** (生成多个子查询) -> **Simulation** (执行检索) -> **Backpropagation** (反向传播评分)。
* **动态探索**：Agent 会尝试多个搜索方向。如果某个方向检索结果差（评分低），它会通过 UCT 算法自动转向其他更有潜力的路径，而非死磕。



### 4. 鲁棒性：事实锚定与防幻觉 (Fact Anchoring & Anti-Hallucination)

* **痛点**：LLM 喜欢编造数据，尤其是在产能、良率等数字问题上。
* **优化方案**：
* **中间答案锚定**：在 Simulation 阶段，强制 LLM 生成带 `[Ref: ID]` 的中间结论。
* **引用校验机制** (`evaluation_node`)：专门的 `verify_prompt` 检查生成的回答是否包含了引用，且引用的内容是否真实存在于检索文中。若发现无引用或引用造假，给予 **0分惩罚**。
* **拒答奖励**：如果 LLM 诚实回答“未找到信息”，系统会给予小额正向奖励 (0.1)，鼓励诚实而非胡编乱造。



---

## 🏗️ 系统架构

用户查询 (User Query)
      ⬇
┌───────────────────────┐
│  术语归一化 (Normalizer) │ 🛠️ 预处理：纠正拼写、统一术语
└───────────┬───────────┘
            ⬇
    [ MCTS 搜索树根节点 ]
            ⬇
┌<─── 🔁 LATS 推理循环 (Max Iterations) ───>┐
│                                             │
│  1. 🧠 选择 (Selection)                     │
│     └── 基于 UCT 算法选择最有潜力的路径       │
│                                             │
│  2. 🌿 扩展 (Expansion)                     │
│     └── 生成多角度子查询 (Sub-queries)        │
│                                             │
│  3. 🔍 模拟 (Simulation) ◀──交互──▶ [知识库] │
│     └── 并行执行：本地混合检索 + 联网搜索      │
│         (Fact Anchoring 事实锚定)            │
│                                             │
│  4. ⚖️ 评估 (Evaluation)                    │
│     └── 对检索结果打分 (Reflexion)           │
│     └── 反向传播更新父节点价值 (Backprop)     │
│                                             │
└───────────┬─────────────────────────────────┘
            ⬇ 🛑 达到最大迭代次数
┌───────────────────────┐
│  最终生成 (Generation)  │ 📝 汇总高分路径，生成结构化研报
└───────────────────────┘

---

## 📂 模块详解

### 1. 智能体核心 (`LATS_v3.py`)

这是系统的大脑，实现了完整的 MCTS 流程：

* **Normalize Node**: 使用正则 + LLM 修正术语（例：`stm 32` -> `STM32`, `3 nm` -> `3nm Process Node`）。
* **Expansion Node**: 注入了半导体行业知识，提示词强制要求区分 "WPM" (晶圆/月) 和 "Revenue" (营收)，避免概念混淆。
* **Simulation Node**:
* 并行调用 `ensemble_retriever` (本地) 和 `GoogleSerper` (网络)。
* 使用 `temperature=0` 确保中间推理的事实性。


* **Generation Node**: 收集所有高分路径 (`value > 0.4`) 的信息，去重后生成结构化研报。

### 2. 高性能数据管道 (`write_milvus.py`)

为了处理大规模文档，设计了 **生产者-消费者** 模型：

* **Process 1 (Parser)**: CPU 密集型。负责 Markdown 解析、Header 聚合、Semantic Chunking。
* **Process 2 (Writer)**: IO 密集型。负责 Embedding 向量化、BM25 计算、Milvus 批量插入。
* **Queue**: 使用 `multiprocessing.Queue` 进行进程间通信，实现流水线作业。

### 3. 爬虫子系统 (`spider_arxiv_project`)

* 基于 `Scrapy` 框架。
* 针对性抓取 `cat:physics` + `semiconductor` + `2024` 的最新论文。
* 自动下载 PDF，为知识库提供最新的学术前沿数据。

---

## 🚀 快速开始

### 环境准备

1. 安装依赖：
```bash
pip install -r requirements.txt

```


2. 启动 Milvus (Docker)：
```bash
docker run -d --name milvus-standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:v2.3.0 standalone

```


3. 配置 `.env`：
```ini
OPENAI_API_KEY=sk-...
SERPER_API_KEY=... (用于 Google Search)
MILVUS_URI=http://localhost:19530

```



### 运行步骤

**Step 1: 构建知识库**
将 Markdown 资料放入 `pdf2md/trans_md` 目录，运行数据管道：

```bash
python write_milvus.py
# 输出：
# [Info] 开始解析目录...
# [Info] 成功写入 20 条文档 (Hybrid Index)...

```

**Step 2: 运行 LATS Agent**
启动主程序进行深度推理：

```bash
python LATS_v3.py

```

*观察控制台输出，你会看到 Agent 如何“思考”：*

1. `[Normalize]` 将你的口语化提问转为专业术语。
2. `[Expand]` 生成了 3 个不同的子查询方向。
3. `[Simulate]` 发现某个方向检索结果为空，自动放弃。
4. `[Evaluate]` 对检索到的产能数据打分，并进行反向传播。
5. `[Generate]` 输出包含引用来源的最终报告。

---

## 📊 效果对比

| 特性 | Easy RAG (Baseline) | Agentic RAG | **Core-RAG (LATS)** |
| --- | --- | --- | --- |
| **检索方式** | 仅 Dense | Dense + Web | **Dense + Sparse (BM25) + Web** |
| **查询理解** | 原文检索 | 基础重写 | **术语归一化 + 多角度扩展** |
| **推理路径** | 线性 (Chain) | 简单循环 (Loop) | **树状搜索 (Tree) + 回溯** |
| **防幻觉** | 无 | 弱 | **强 (事实锚定 + 引用校验)** |
| **长难问题** | 效果差 | 一般 | **优秀 (多步推理)** |

---

## 🛠 技术栈

* **Orchestration**: LangChain, LangGraph
* **Vector DB**: Milvus (Hybrid Search)
* **LLM**: GPT-4o-mini
* **Embedding**: BAAI/bge-small-zh-v1.5
* **NLP Tools**: Jieba (BM25), Unstructured (Parsing)


