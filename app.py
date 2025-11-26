import os
import streamlit as st
from dotenv import load_dotenv

# LangChain 1.x components
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Vector + Embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -----------------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# -----------------------------------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY missing. Add it in Streamlit Cloud → Settings → Secrets.")
    st.stop()


# -----------------------------------------------------------
# INITIALIZE GROQ LLM
# -----------------------------------------------------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile"
)

prompt = ChatPromptTemplate.from_template("""
Use ONLY the following context to answer the question concisely.

<context>
{context}
</context>

Question: {input}
""")


# -----------------------------------------------------------
# CREATE VECTOR DATABASE
# -----------------------------------------------------------
def create_vector_db():

    if "vectors" in st.session_state:
        st.info("Vector DB already created.")
        return

    # Check folder exists
    if not os.path.exists("research_papers"):
        st.error("❌ Folder `research_papers/` does not exist in the repo!")
        return

    files = os.listdir("research_papers")
    st.write("📂 Files found:", files)

    if len(files) == 0:
        st.error("❌ No PDF files found in `research_papers/`.")
        return

    st.write("📄 Loading PDFs...")
    loader = PyPDFDirectoryLoader("research_papers")
    docs = loader.load()

    if len(docs) == 0:
        st.error("❌ PDFs loaded but contain NO extractable text. They may be scanned images.")
        return

    st.write(f"Loaded {len(docs)} documents.")

    st.write("✂️ Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    if len(chunks) == 0:
        st.error("❌ Documents produced ZERO text chunks. Possibly scanned PDFs or empty text.")
        return

    st.write(f"Generated {len(chunks)} chunks.")

    st.write("🔢 Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    st.write("📦 Building FAISS vectorstore...")
    try:
        vectorstore = FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        st.error(f"❌ Error building vectorstore: {e}")
        return

    st.session_state.vectors = vectorstore
    st.success("🎉 Vector DB created successfully!")


# -----------------------------------------------------------
# BUILD RAG CHAIN
# -----------------------------------------------------------
def build_rag_chain():

    retriever = st.session_state.vectors.as_retriever()

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    rag_chain = (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# -----------------------------------------------------------
# STREAMLIT APP UI
# -----------------------------------------------------------
st.title("📘 Research Paper RAG (Groq + LangChain 1.x)")

question = st.text_input("Ask a question from the research papers:")

if st.button("Build Vector Database"):
    create_vector_db()

if question:

    if "vectors" not in st.session_state:
        st.error("⚠️ Please build the vector DB first.")
        st.stop()

    rag_chain = build_rag_chain()
    answer = rag_chain.invoke({"input": question})

    st.subheader("🧠 Answer")
    st.write(answer)

    # Retrieve docs
    with st.expander("📄 Retrieved Context"):
        retrieved_docs = st.session_state.vectors.as_retriever().get_relevant_documents(question)
        for d in retrieved_docs:
            st.write(d.page_content)
            st.write("---")
