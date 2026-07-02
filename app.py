import gradio as gr
from agent import ask_cv_agent


# 统一前端控制中心，内置异常捕获
def chat_func(question, image):
    try:
        return ask_cv_agent(question, image)
    except Exception as e:
        return f"### ❌ 智能体系统调度失败\n\n**原因分析**：{str(e)}\n\n*提示：请检查本地网络连通性或核对 .env 文件中的 API Key 是否失效。*"


# 组装动态、灵活的专家系统UI
with gr.Blocks(title="CV-Metric Vision Agent") as demo:
    gr.Markdown("# 🔬 智能视觉多模态量化校验与模型评估系统")
    gr.Markdown(
        "**系统定位**：融合多模态视觉大模型与离线文献知识库。根据您的具体指令，动态生成针对性的缺陷审计、数理公式或 PyTorch 模型调优方案。"
    )

    with gr.Row():
        # 左侧：输入控制列
        with gr.Column(scale=1):
            # 核心修改点：变 value 为 placeholder，并提供多维度提问向导，再也不用手动删大段文字了
            txt_input = gr.Textbox(
                label="可信度检验提问（可自由输入任意视觉指标/代码/审图需求）",
                placeholder=(
                    "你可以这样问我：\n"
                    "1. [纯审图] 仔细看看左边这张图，红色的算法预测掩码有什么问题吗？\n"
                    "2. [要代码] 在 PyTorch 里面，怎么把 SSIM 损失函数加进模型的训练循环里？直接给我核心代码片段。\n"
                    "3. [纯理论] 一句话说清楚，为什么在边缘场景下只看 PSNR 往往是不可信的？"
                ),
                lines=5
            )
            img_input = gr.Image(label="上传待校验的边缘场景图像（可选）", type="filepath")
            btn = gr.Button("🚀 触发多模态智能体深度分析", variant="primary")

        # 右侧：精美渲染列
        with gr.Column(scale=1):
            gr.Markdown("### 📋 智能体综合评估报告")
            txt_output = gr.Markdown("等待接收智能体多模态分析输入...")

    # 动态事件绑定
    btn.click(chat_func, inputs=[txt_input, img_input], outputs=[txt_output])

if __name__ == "__main__":
    # 本地局域网广播启动并拉起内网穿透机制
    demo.launch()
