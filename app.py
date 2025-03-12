import os
import gradio as gr
import gradio.themes as themes
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

# Access secrets from environment variables
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL")

# Initialize Google Generative AI
system_message = "You are a helpful AI assistant expert in Sri Lankan Tea industry. Use the following pieces of context to answer the users question about Sri Lankan Tea industry.  If you don't know the answer based on the context, just say 'I am sorry, but I cannot answer this question from the given documents.' and do not try to make up an answer"
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-002", streaming=True)

# System message
system_message = """You are a highly knowledgeable AI assistant specializing in the Sri Lankan Tea industry. Your expertise is based on analyzing official documents and reports related to this industry.

When a user asks a question about the Sri Lankan Tea industry, you MUST use ONLY the information provided in the context I will give you to formulate your answer.

Provide informative and detailed answers, drawing directly from the context. If the context contains the answer, please provide it in a clear and structured way.

If the provided documents do not contain the answer to the user's question, or if the context is insufficient to answer accurately, respond with: 'I am sorry, but I cannot provide a definitive answer to this question based on the information in the documents provided. The documents do not contain information to answer this question.'  Do not invent or make up information. Focus strictly on the content of the provided documents."""

# Connect to Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = PINECONE_INDEX_NAME
if index_name not in pc.list_indexes().names():
    raise ValueError(f"Index '{index_name}' not found. Please create the index first.")
index = pc.Index(index_name)

# Integrating with LangChain
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Initialize Pinecone vector store and retrieval chain
vector_store = PineconeVectorStore(index=index, embedding=embeddings)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)

# Utility function to limit output
def limit_words(text, max_words=500):
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + "..."
    return text

# Chat response function
def respond(message, history):
    print(f"Input: {message}. History: {history}\n")
    response = qa_chain.run(message)
    limited_response = limit_words(response, 500)
    history.append((message, limited_response))
    return "", history

# Custom CSS
custom_css = """
body {
    background-color: #f0efe9;
    font-family: 'Georgia', serif;
}

header {
    background-color: #d2b48c; 
    color: #fff;
    padding: 20px;
    text-align: center;
}

header h1 {
    font-size: 36px;
}

.gradio-container {
    border-radius: 10px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
}

.message {
    background-color: #fff;
    border-radius: 10px;
    padding: 10px;
    margin: 5px;
}
"""

# Build Gradio interface
with gr.Blocks(css=custom_css) as demo_interface: # Removed themes.Monochrome()
    with gr.Row():
        gr.Markdown("<header><h1>Sri Lankan Tea Industry Assistant</h1></header>")
    gr.Markdown("### Bridging the gap between tea estate owners and essential documents/information.")
    chat = gr.Chatbot()
    state = gr.State([])
    txt = gr.Textbox(
        placeholder="Ask me anything about the Sri Lankan Tea Industry",
        label="Your Query"
    )
    txt.submit(respond, [txt, state], [txt, chat])

# Launch the interface
demo_interface.launch(share=True, debug=True)