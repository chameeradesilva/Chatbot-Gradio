import os
import gradio as gr
import gradio.themes as themes
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone

# Access secrets from environment variables
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL")

# Initialize Google Generative AI with a system prompt
system_message = """
You are a highly knowledgeable AI assistant specializing in the Sri Lankan Tea industry.
Your expertise is based on analyzing official documents and reports related to this industry.
When a user asks a question, you MUST use ONLY the provided context to answer.
If the documents do not contain the answer, respond with:
'I am sorry, but I cannot provide a definitive answer to this question based on the provided information.'
Do not invent or make up details.
"""
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-002", streaming=True)

# Connect to Pinecone and check for the index
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = PINECONE_INDEX_NAME
if index_name not in pc.list_indexes().names():
    raise ValueError(f"Index '{index_name}' not found. Please create the index first.")
index = pc.Index(index_name)

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Create the retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)

# Utility function to limit output to roughly 500 words.
def limit_words(text, max_words=500):
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + "..."
    return text

# Chat response function using the QA chain
def respond(message, history):
    print(f"Input: {message}. History: {history}\n")
    response = qa_chain.run(message)
    limited_response = limit_words(response, 500)
    history.append((message, limited_response))
    return "", history

# Custom CSS for improved UI/UX and to ensure message visibility.
custom_css = """
body {
    background-color: #f0efe9;
    font-family: 'Georgia', serif;
}
header {
    background-image: url('https://openceylon.com/images/plantation-manager.jpg');
    background-size: cover;
    background-position: center;
    height: 200px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #fff;
}
header h1 {
    font-size: 36px;
    background-color: rgba(0, 0, 0, 0.5);
    padding: 10px 20px;
}

/* Enhance the container appearance */
.gradio-container {
    border-radius: 10px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
}

/* Override chatbot message styles for better readability */
.chatbot .message,
.chatbot .message.user,
.chatbot .message.ai {
    color: #333 !important; /* Force dark text */
    background-color: #e8f5e9 !important;
    border-radius: 10px;
    padding: 10px;
    margin: 5px;
}
.chatbot .message.user {
    background-color: #dcedc8 !important;
}

/* Ensure all text elements inside the chatbot are visible */
.gradio-container, .chatbot, .chatbot * {
    color: #333 !important;
}
"""

# Build a custom Gradio Blocks interface suitable for HuggingFace Spaces.
with gr.Blocks(css=custom_css, theme=themes.Soft()) as demo_interface:
    with gr.Row():
        gr.Markdown("<header><h1>Sri Lankan Tea Plantation Assistant</h1></header>")
    gr.Markdown("### Bridging the gap between tea estate owners and essential documents/information.")
    
    # Chatbot component with custom class for CSS targeting.
    chat = gr.Chatbot(elem_classes=["chatbot"])
    state = gr.State([])
    txt = gr.Textbox(
        placeholder="Ask me anything about the Sri Lankan Tea Plantation Guidelines",
        label="Your Query"
    )
    
    # Wire the textbox submit action.
    txt.submit(respond, [txt, state], [txt, chat])

# Launch the interface. This setup works well on HuggingFace Spaces.
demo_interface.launch(share=True, debug=True)