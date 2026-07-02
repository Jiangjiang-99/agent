import gradio as gr
from agent import ask_cv_agent

def chat_func(question, image):
    try:
        return ask_cv_agent(question, image)
    except Exception as e:
        return f"### ❌ 智能体系统调度失败\n\n**原因分析**：{str(e)}\n\n*提示：请检查本地网络连通性或核对 .env 文件中的 API Key 是否失效。*"

# 组装与论文题目完全对齐的统一专家系统UI
with gr.Blocks(title="图像质量评估与算法缺陷分析智能体") as demo:
    gr.Markdown("# 🔬 基于多模态大模型的图像质量评估与算法缺陷分析智能体系统")
    gr.Markdown(
        "**系统定位**：本系统是一套垂直于机器视觉质检领域的智能体（Agent）。"
        "融合多模态视觉大模型与离线学术文献知识库，根据工程师的实际指令，"
        "动态提供高置信度的图像质量评估指标（PSNR/SSIM）咨询或算法预测结果的缺陷分析。"
    )

    with gr.Row():
        with gr.Column(scale=1):
            txt_input = gr.Textbox(
                label="可信度检验与技术咨询输入（可自由输入任意视觉指标/代码/审图需求）",
                placeholder=(
                    "你可以这样提问系统：\n"
                    "1. [算法缺陷分析] 仔细看看左边这张图，红色的算法预测掩码有什么问题吗？\n"
                    "2. [代码方案生成] 在 PyTorch 里面，怎么把 SSIM 损失函数加进模型的训练循环里？直接给我核心代码片段。\n"
                    "3. [质量评估咨询] 一句话说清楚，为什么在边缘场景下只看 PSNR 往往是不可信的？"
                ),
                lines=5
            )
            img_input = gr.Image(label="上传待分析的算法预测图像（可选）", type="filepath")
            btn = gr.Button("🚀 触发多模态智能体深度分析", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 📋 智能体综合分析报告")
            txt_output = gr.Markdown("等待接收智能体多模态分析输入...")

    btn.click(chat_func, inputs=[txt_input, img_input], outputs=[txt_output])

if __name__ == "__main__":
    demo.launch()
