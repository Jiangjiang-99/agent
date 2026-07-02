import os
import warnings
import requests
from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma

# 强迫症福音：彻底过滤并静音所有 LangChain 的老版本/过时警告，还你一个绝对干净的控制台
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 清理残留网关环境变量
if "DASHSCOPE_HTTP_BASE" in os.environ:
    del os.environ["DASHSCOPE_HTTP_BASE"]

# 全局IO编码，防止Windows控制台乱码
os.environ["PYTHONIOENCODING"] = "utf-8"

# 加载密钥
load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

# 1. 【知识库】初始化
embedding = DashScopeEmbeddings(model="text-embedding-v2", dashscope_api_key=api_key)
db = Chroma(persist_directory="./chroma_db", embedding_function=embedding)


# 2. 【工具 (Tool)】定义
def call_knowledge_base_tool(query: str) -> str:
    """本地专有图像处理知识库检索工具"""
    docs = db.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in docs])


# 底层核心大脑 (LLM) 调用接口
def get_qwen_answer(prompt_text: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "qwen3.7-plus",
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.1,
        "max_tokens": 1500
    }
    resp = requests.post(base_url, headers=headers, json=data)
    res_json = resp.json()
    if "error" in res_json:
        raise Exception(f"API Error:{res_json['error']}")
    return res_json["choices"][0]["message"]["content"]


# 3. 【工作流 (Workflow)】智能体自动化编排核心逻辑
def ask_cv_agent(question: str) -> str:
    # 步骤 A：意图路由决策
    routing_prompt = f"""你是一个路由决策智能体。请判断以下用户提问是否涉及“计算机视觉、图像质量评估指标（PSNR/SSIM）或边缘场景模型检验”的专业技术知识。
用户问题: {question}
如果是专业技术问题并需要查阅本地知识库，请仅回复 "YES"。如果只是日常打招呼、闲聊或完全无关的问题，请仅回复 "NO"。"""

    route_decision = get_qwen_answer(routing_prompt).strip().upper()

    context_text = ""
    # 步骤 B：工作流分支条件控制
    if "YES" in route_decision:
        print("[Agent Workflow] 智能体自主决策：判定当前为专业问题，正在触发本地知识库检索工具...")
        context_text = call_knowledge_base_tool(question)
    else:
        print("[Agent Workflow] 智能体自主决策：无需检索工具，将直接基于内置常识响应...")

    # 步骤 C：结合召回上下文与自身常识，生成大作业题目5要求的结构化报告
    final_prompt = f"""你是一个专门用于【矫正与检验CV模型】的计算机视觉指标评估智能体。
你的任务是协助工程师评估路面损伤检测等边缘场景下CV模型的输出 quality（完美对齐大作业题目5）。

从本地知识库工具中检索到的专业参考资料：
{context_text}

用户检验提问: {question}

请结合上述参考资料与行业通用常识，严格按照以下 4 个结构化板块输出最终分析报告：
- **定义 (Definition)**：指标的物理意义，并阐述它在【检验及矫正CV模型】时的具体作用。
- **计算公式 (Calculation Formula)**：给出标准的数学表达形式（支持标准的 Markdown 渲染）。
- **优点 (Advantages)**：该指标在边缘场景模型校验中的核心长处。
- **局限性 (Limitations)**：该指标在矫正CV模型时可能存在的局限性（如与人类主观视觉不感知不一致等）。"""

    return get_qwen_answer(final_prompt)


if __name__ == "__main__":
    try:
        query = "What are PSNR and SSIM as image quality evaluation metrics?"
        result = ask_cv_agent(query)
        print("\n===== Qwen3.7-Plus Agent Output =====")
        print(result)
    except Exception as err:
        print("Call Failed Info:", str(err))
