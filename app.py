import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import plotly.express as px
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="P&L AI诊断Agent", layout="wide")
st.title("🚢 AI驱动的经营P&L智能诊断与决策支持Agent")
st.markdown("**复现海运5亿美金颗粒度分析 | 30秒出诊断+可执行方案**")

# Groq API Key is automatically loaded from the .env file
if "GROQ_API_KEY" not in os.environ:
    st.error("❌ 未在环境变量或 .env 中找到 GROQ_API_KEY")
    st.stop()

# RAG知识库
@st.cache_resource
def get_rag_chain():
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    kb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_base.txt")
    with open(kb_path, "r", encoding="utf-8") as f:
        docs = f.read().split("\n\n")
    vectorstore = Chroma.from_texts(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    template = """你是海运经营分析专家。根据以下P&L数据和行业知识，完成诊断：
    {context}
    
    数据：
    {data_summary}
    
    请严格按以下格式输出：
    1. 总体诊断（毛利、异常线）
    2. 低毛利根因分析
    3. 3套可执行建议（含预计Margin提升%）
    4. 风险提示
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": retriever, "data_summary": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

chain = get_rag_chain()

# 文件上传
uploaded_file = st.file_uploader("上传P&L Excel/CSV（支持多航线/客户）", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # 校验必要列是否存在
    required_cols = ["收入(万美元)", "成本(万美元)", "航线", "客户"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"❌ 上传的数据格式有误！缺少以下必要列：**{', '.join(missing_cols)}**")
        st.info("💡 提示：请检查文件第一行是否是正确的表头。如果是CSV文件，请确保第一行没有多余的逗号或空行。")
        st.stop()
        
    st.subheader("原始数据预览")
    st.dataframe(df.head(10))
    
    # 自动计算指标
    df["毛利率"] = (df["收入(万美元)"] - df["成本(万美元)"]) / df["收入(万美元)"] * 100
    summary = df.describe().to_string()
    low_margin = df[df["毛利率"] < 10]
    
    st.plotly_chart(px.bar(df, x="航线", y="毛利率", color="客户", title="各航线毛利率分布"), use_container_width=True)
    
    if st.button("🚀 一键AI诊断 & 生成决策方案"):
        with st.spinner("Agent正在诊断中（RAG检索行业基准 + LLM推理）..."):
            result = chain.invoke(summary)
            st.success("诊断完成！")
            st.markdown(result)
            
            # 下载基础报告 (TXT)
            st.download_button("📥 下载基础诊断报告 (TXT)", result, file_name="P&L_AI_Diagnosis_Report.txt")
            
            # 导出网页全景为彩色PDF (使用浏览器原生打印功能)
            st.markdown(
                """
                <style>
                @media print {
                    /* 1. 隐藏多余元素：侧边栏、顶部栏、页脚 */
                    header, [data-testid="stSidebar"], footer { 
                        display: none !important; 
                    }
                    
                    /* 2. 让核心组件（图表、Markdown）在打印时尽量不被从中间截断 */
                    .element-container, .stMarkdown, .stPlotlyChart {
                        page-break-inside: avoid !important;
                    }
                    
                    /* 3. 强制保留所有 CSS 背景色，确保彩色打印 */
                    * {
                        -webkit-print-color-adjust: exact !important;
                        print-color-adjust: exact !important;
                    }
                }
                </style>
                """, unsafe_allow_html=True
            )
            
            st.components.v1.html(
                """
                <div style="text-align: center; margin-top: 20px;">
                    <button onclick="window.parent.print()" style="
                        background-color: #ff4b4b; 
                        color: white; 
                        padding: 12px 24px; 
                        border: none; 
                        border-radius: 6px; 
                        cursor: pointer; 
                        font-weight: bold; 
                        font-size: 16px;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                        transition: 0.3s;">
                        🖨️ 导出全景彩色报告 (PDF)
                    </button>
                    <p style="font-size: 13px; color: #666; margin-top: 8px;">
                        💡 <b>操作指南:</b> 点击上方按钮，在弹出的打印设置窗口中，将目标打印机选择为 <b>「另存为 PDF」</b>。
                    </p>
                </div>
                """,
                height=120
            )

st.caption("技术栈：LangChain RAG + Groq Llama3.7 + Streamlit | 完全本地可控，企业级可解释性")