import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()  # 加载.env里的GROQ_API_KEY
os.makedirs("chroma_db", exist_ok=True)  # 用os创建持久化目录

# ============== 初始化（只运行一次）==============
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    return Chroma(persist_directory="chroma_db", embedding_function=embeddings)

@st.cache_resource
def load_llm():
    return ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)

vectorstore = load_vectorstore()
llm = load_llm()
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ============== 页面布局 ===============
st.title("🌊 Miles海运知识RAG Agent")
st.caption("结合Groq极速推理 + Chroma向量库，3秒出港拥堵分析报告")

# 侧边栏配置（用dotenv安全管理）
with st.sidebar:
    st.header("⚙️ 配置")
    temperature = st.slider("温度", 0.0, 1.0, 0.7)
    llm.temperature = temperature

# 聊天历史（session_state自动保存）
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入
if prompt := st.chat_input("问我任何海运问题，例如：2026年新加坡港拥堵情况？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Groq正在极速思考..."):
            # RAG链（LCEL）
            template = """基于以下上下文回答问题：
            {context}
            
            问题: {question}
            答案（用中文，专业且简洁）:"""
            prompt_template = ChatPromptTemplate.from_template(template)
            
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt_template
                | llm
                | StrOutputParser()
            )
            
            response = chain.invoke(prompt)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

import plotly.express as px
# ... 在Agent返回数据后
fig = px.line(df_port, x="日期", y="拥堵指数", title="港口拥堵趋势")
st.plotly_chart(fig, use_container_width=True)