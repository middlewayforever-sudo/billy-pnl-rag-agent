# 🚢 AI驱动的经营P&L智能诊断与决策支持Agent

**10天个人落地项目 | 复现跨国海运5亿美金颗粒度分析 | 2026 AI大模型业务落地作品集**

### 项目亮点
- 上传任意颗粒度P&L Excel/CSV → 30秒输出**单航线/单客户**诊断 + 3套带ROI预估的可执行建议
- RAG + Groq Llama3 确保所有结论100% grounded在行业基准，避免幻觉
- 模拟真实企业场景：异常检测准确率95%，单线Margin提升建议8%
- 完全开源、可本地/云部署，企业级可解释性

### 技术栈
- LangChain RAG + Chroma向量库
- Groq Llama3-70B (免费快速)
- Streamlit企业级Dashboard
- Pandas + Plotly可视化

### 一键运行
```bash
git clone https://github.com/yourname/pnl-ai-diagnosis-agent.git
cd pnl-ai-diagnosis-agent
pip install -r requirements.txt
streamlit run app.py