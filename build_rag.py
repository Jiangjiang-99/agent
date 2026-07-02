from dotenv import load_dotenv
import os
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")


# 加载docs全部PDF
def load_all_pdf(docs_dir="./docs"):
    pages = []
    # 先判断文件夹是否存在，防止路径报错
    if not os.path.exists(docs_dir):
        print(f"错误：文件夹 {docs_dir} 不存在！")
        return pages
    for filename in os.listdir(docs_dir):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(docs_dir, filename)
            loader = PyPDFLoader(file_path)
            pages.extend(loader.load())
            print(f"已加载文件：{filename}")
    return pages


# 缩短单段文本，减少token消耗额度
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=80,
    separators=["\n\n", "\n", "。", "，"]
)

# 阿里云向量
embedding = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=api_key
)


# 每批20条，休眠2秒平缓调用API
def batch_add_to_chroma(docs, batch_size=20):
    # 关键点：实例化时指定持久化目录，自动保存，删除persist()
    vector_store = Chroma(
        embedding_function=embedding,
        persist_directory="./chroma_db"
    )
    total = len(docs)
    print(f"总文本块：{total}，分批写入向量库")
    for i in range(0, total, batch_size):
        batch = docs[i:i + batch_size]
        vector_store.add_documents(batch)
        current = min(i + batch_size, total)
        print(f"进度 {current}/{total}")
        time.sleep(2)
    return vector_store


if __name__ == "__main__":
    raw_docs = load_all_pdf()
    if len(raw_docs) == 0:
        print("docs文件夹下没有任何PDF文件，请检查！")
    else:
        split_docs = text_splitter.split_documents(raw_docs)
        print(f"文档切分完成，共{len(split_docs)}个文本块")
        vector_store = batch_add_to_chroma(split_docs, batch_size=20)
        print("✅ 向量知识库构建完成")