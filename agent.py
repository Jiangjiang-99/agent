import os
import requests
import base64
from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma

# 清理可能冲突的网关环境变量
if "DASHSCOPE_HTTP_BASE" in os.environ:
    del os.environ["DASHSCOPE_HTTP_BASE"]

# 全局环境编码，防止 Windows 终端渲染乱码
os.environ["PYTHONIOENCODING"] = "utf-8"

# 载入环境变量密钥
load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

# 1. 【专业知识库】离线语义向量存储初始化
embedding = DashScopeEmbeddings(model="text-embedding-v2", dashscope_api_key=api_key)
db = Chroma(persist_directory="./chroma_db", embedding_function=embedding)


# 2. 【智能体工具 (Tool)】封装：本地知识检索
def call_knowledge_base_tool(query: str) -> str:
    """本地计算机视觉专家文献知识库检索工具"""
    docs = db.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in docs])


# 图像处理辅助工具：将本地缓存图片文件安全转换为 Base64 规范编码流
def get_base64_image(image_path: str) -> str:
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


# 底层 LLM 核心网关通用适配器
def call_qwen_api(model_name: str, messages: list) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 1500
    }
    resp = requests.post(base_url, headers=headers, json=data)
    res_json = resp.json()
    if "error" in res_json:
        raise Exception(f"API 响应异常: {res_json['error']}")
    return res_json["choices"][0]["message"]["content"]


# 3. 【多模态自动化工作流 (Workflow)】智能体编排核心逻辑
def ask_cv_agent(question: str, image_path: str = None) -> str:
    # 步骤 A：意图路由前置决策
    routing_prompt = f"请判断以下用户提问是否涉及图像质量评估指标（PSNR/SSIM）或视觉模型检验专业技术。用户问题: {question}"
    messages = [{"role": "user", "content": routing_prompt}]
    route_decision = call_qwen_api("qwen3.7-plus", messages).strip().upper()

    context_text = ""
    if "YES" in route_decision or image_path is not None:
        print("[Agent Workflow] 智能体自主决策：当前属于垂直技术任务，正在激活专业工具链检索本地库...")
        context_text = call_knowledge_base_tool(question)

    # 步骤 B：重构提示词，彻底打破固定模板，让模型根据用户需求自由发挥
    final_prompt_text = f"""你是一个工业级计算机视觉（CV）模型质检专家智能体。
现在请你忘记之前任何死板的输出模版。请根据工程师提出的【具体提问需求】，结合本地知识库内容以及输入的图像特征（如果提供了图片），进行针对性、定制化的解答。

【答复核心指导原则】
1. **千人千面，按需回答**：用户问什么，你就核心回答什么。
   - 如果用户让你“分析左侧图像缺陷”，你就专注于分析图像里的红框、掩码和具体视觉局限。
   - 如果用户单纯问你一个“理论概念、公式或PyTorch调优代码”，你就直接给出高密度的干货，不要硬扯图像缺陷。
2. **严禁废话与教条**：拒绝任何没有营养的寒暄、前言和总结。不强制要求每次都输出固定板块。只有当问题确实需要时，才列出相应的 LaTeX 公式或代码。
3. **真实性原则**：严禁捏造未经像素矩阵真实计算的假定定量降噪数值或得分数字。

本地知识库检索到的参考资料：
{context_text}

工程师当前的实际提问需求：{question}
"""

    # 步骤 C：依据输入形态，动态路由执行器形态
    if image_path and os.path.exists(image_path):
        print(f"[Agent Workflow] 多模态模块激活：正在调用官方旗舰 qwen-vl-max 进行动态审图...")
        base64_str = get_base64_image(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": final_prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_str}"}}
                ]
            }
        ]
        return call_qwen_api("qwen-vl-max", messages)  # 使用大作业规定的视觉大模型

    else:
        messages = [{"role": "user", "content": final_prompt_text}]
        return call_qwen_api("qwen3.7-plus", messages)  # 使用大作业规定的语言大模型


if __name__ == "__main__":
    try:
        print(ask_cv_agent("如何利用SSIM指标矫正视觉模型？"))
    except Exception as e:
        print("测试未通过:", e)
