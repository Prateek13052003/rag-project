import os
import streamlit as st
from dotenv import load_dotenv

# LangChain 1.x compatible imports
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# -----------------------------------------
# Load environment variables
# -----------------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("groq_api_key")


# -----------------------------------------
# Initialize LLM (Groq)
# -----------------------------------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile"
)


# Prompt template (LangChain 1.x)
prompt = ChatPromptTemplate.from_template("""
Use ONLY the following context to answer the question.

<context>
{context}
</context>

Question: {input}
""")


# -----------------------------------------
# Vector DB Creation
# -----------------------------------------
def create_vector_db():

    if "vectors" in st.session_state:
        st.info("Vector DB already created.")
        return

    st.write("📂 Loading PDFs from: research_papers/")
    loader = PyPDFDirectoryLoader("research_papers")
    docs = loader.load()

    st.write("✂️ Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    st.write("🔢 Creating embeddings (HuggingFace)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    st.write("📦 Building FAISS vectorstore...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save in session
    st.session_state.vectors = vectorstore
    st.success("🎉 Vector DB ready!")


# -----------------------------------------
# STREAMLIT UI
# -----------------------------------------
st.title("📘 Research Paper RAG (Groq + LangChain 1.x)")

query = st.text_input("Ask a question from the research papers:")

if st.button("Build Vector Database"):
    create_vector_db()

if query:

    if "vectors" not in st.session_state:
        st.error("⚠️ Please build the vector database first!")
    else:

        # Build retriever
        retriever = st.session_state.vectors.as_retriever()

        # Convert retrieved docs to a single text block
        def combine_docs(docs):
            return "\n\n".join([d.page_content for d in docs])

        # Build RAG chain (new LC 1.x way)
        rag_chain = (
            {
                "context": retriever | combine_docs,
                "input": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        # Run RAG pipeline
        answer = rag_chain.invoke({"input": query})

        # Display answer
        st.subheader("🧠 Answer")
        st.write(answer)

        # Show retrieved docs
        with st.expander("📄 Retrieved Context"):
            retrieved_docs = retriever.get_relevant_documents(query)
            for d in retrieved_docs:
                st.write(d.page_content)
                st.write("---")
