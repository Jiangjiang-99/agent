import os
import requests
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma


# 清理残留网关环境变量
if "DASHSCOPE_HTTP_BASE" in os.environ:
    del os.environ["DASHSCOPE_HTTP_BASE"]


# 全局IO编码
os.environ["PYTHONIOENCODING"] = "utf-8"

# 加载密钥
load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


# 向量库初始化
embedding = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=api_key
)
db = Chroma(persist_directory="./chroma_db", embedding_function=embedding)


# 调用 qwen3.7-plus 兼容接口
def get_qwen_answer(prompt_text: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "qwen3.7-plus",
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.1,
        "max_tokens": 1000
    }
    resp = requests.post(base_url, headers=headers, json=data)
    res_json = resp.json()
    if "error" in res_json:
        raise Exception(f"API Error:{res_json['error']}")
    return res_json["choices"][0]["message"]["content"]


# RAG问答
def ask_cv_agent(question: str) -> str:
    docs: list[Document] = db.similarity_search(question, k=3)
    context_text = "\n".join([doc.page_content for doc in docs])
    prompt = """You are a professional computer vision assistant.
Rule 1: Give priority to information extracted from reference materials.
Rule 2: If complete mathematical formulas are not covered in documents, you can add the standard industry formulas of PSNR and SSIM, and clearly mark them as general industry common knowledge.
Reference materials:
{context_text}
User question: {question}
Output complete content with 4 parts: Definition, Calculation Formula, Advantages, Limitations.
""".format(context_text=context_text, question=question)
    return get_qwen_answer(prompt)


if __name__ == "__main__":
    try:
        query = "What are PSNR and SSIM as image quality evaluation metrics?"
        result = ask_cv_agent(query)
        print("===== Qwen3.7-Plus RAG Output =====")
        print(result)
    except Exception as err:
        safe_err = str(err).encode("utf-8", errors="replace").decode("utf-8")
        print("Call Failed Info:", safe_err)