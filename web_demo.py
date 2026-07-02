import gradio as gr
from agent import ask_cv_agent

def chat_func(question):
    return ask_cv_agent(question)

with gr.Blocks(title="CV模型校验指标智能体") as demo:
    gr.Markdown("# 🤖 Qwen3.7-Plus 边缘场景 CV 模型检验与矫正智能体")
    gr.Markdown("**大作业选题**：题目5 - 泛化的路面损伤检测与边缘场景下的CV模型矫正检验 。本智能体通过外挂专业数字图像处理知识库，自主调用检索工具，提供高置信度的指标分析报告。")
    txt_input = gr.Textbox(label="输入你要查询的模型检验指标问题", value="在边缘损伤检测场景下，如何利用PSNR和SSIM指标来检验和矫正CV模型的泛化误差？")
    txt_output = gr.Textbox(label="智能体结构化分析报告", lines=15)
    btn = gr.Button("开始智能体分析")
    btn.click(chat_func, inputs=[txt_input], outputs=[txt_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)
