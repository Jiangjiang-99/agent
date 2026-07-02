import os
import requests
import base64
from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma

if "DASHSCOPE_HTTP_BASE" in os.environ:
    del os.environ["DASHSCOPE_HTTP_BASE"]

os.environ["PYTHONIOENCODING"] = "utf-8"

load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

# 统一维护的大模型基座常量
MODEL_TEXT = "qwen3.7-plus"
MODEL_VISION = "qwen-vl-max"

# 离线语义向量存储初始化
embedding = DashScopeEmbeddings(model="text-embedding-v2", dashscope_api_key=api_key)
db = Chroma(persist_directory="./chroma_db", embedding_function=embedding)

def call_knowledge_base_tool(query: str) -> str:
    """本地图像质量评估文献知识库检索工具"""
    docs = db.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in docs])

def get_base64_image(image_path: str) -> str:
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

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

def ask_cv_agent(question: str, image_path: str = None) -> str:
    # 步骤 A：意图路由前置决策
    routing_prompt = f"请判断以下用户提问是否涉及图像质量评估指标（PSNR/SSIM）或视觉模型检验专业技术。用户问题: {question}"
    messages = [{"role": "user", "content": routing_prompt}]
    route_decision = call_qwen_api(MODEL_TEXT, messages).strip().upper()

    context_text = ""
    if "YES" in route_decision or image_path is not None:
        print("[Agent Workflow] 智能体自主决策：当前属于垂直技术任务，正在激活专业工具链检索本地库...")
        context_text = call_knowledge_base_tool(question)

    # 步骤 B：重构提示词，严格收敛系统认知身份
    final_prompt_text = f"""你是一个【图像质量评估与算法缺陷分析智能体】。
请根据工程师提出的【具体提问需求】，结合本地知识库内容以及输入的图像特征（如果提供了图片），进行针对性、定制化的解答。

【答复核心指导原则】
1. **千人千面，按需回答**：用户问什么，你就核心回答什么。
   - 如果用户让你“分析图像缺陷”，你就专注于分析图像里的红框、算法预测掩码和具体视觉局限。
   - 如果用户单纯问你一个“图像指标理论（PSNR/SSIM）、公式或PyTorch调优代码”，你就直接给出高密度的干货，不要硬扯图像缺陷。
2. **严禁废话与教条**：拒绝任何没有营养的寒暄、前言和总结。不强制要求每次都输出固定板块。只有当问题确实需要时，才列出相应的 LaTeX 公式或代码。
3. **真实性原则**：严禁捏造未经像素矩阵真实计算的假定定量降噪数值或得分数字。

本地知识库检索到的参考资料：
{context_text}

工程师当前的实际提问需求：{question}
"""

    # 步骤 C：依据输入形态，动态路由执行器形态
    if image_path and os.path.exists(image_path):
        print(f"[Agent Workflow] 多模态模块激活：正在调用官方旗舰 {MODEL_VISION} 进行动态审图...")
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
        return call_qwen_api(MODEL_VISION, messages)

    else:
        messages = [{"role": "user", "content": final_prompt_text}]
        return call_qwen_api(MODEL_TEXT, messages)

if __name__ == "__main__":
    try:
        print(ask_cv_agent("如何利用SSIM指标矫正视觉模型？"))
    except Exception as e:
        print("测试未通过:", e)
