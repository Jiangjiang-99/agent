import gradio as gr
from agent import ask_cv_agent

def chat_func(question):
    return ask_cv_agent(question)

with gr.Blocks(title="图像质量指标RAG智能体") as demo:
    gr.Markdown("# Qwen3.7-Plus RAG CV问答智能体")
    gr.Markdown("知识库：图像处理专业PDF资料，仅基于参考资料作答")
    txt_input = gr.Textbox(label="输入你的问题", value="What are PSNR and SSIM as image quality evaluation metrics?")
    txt_output = gr.Textbox(label="回答结果", lines=15)
    btn = gr.Button("开始问答")
    btn.click(chat_func, inputs=[txt_input], outputs=[txt_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")