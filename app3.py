import gradio as gr
import pickle
import urllib.parse
import numpy as np
from fastapi.staticfiles import StaticFiles
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

import shutil
import os

STATIC_DIR = "static"  # This folder must exist

# Make sure static directory exists
os.makedirs(STATIC_DIR, exist_ok=True)

# Load RAG data
with open("vector_store.pkl", "rb") as f:
    rag_data = pickle.load(f)
    chunks = rag_data["chunk_texts"]
    page_numbers = rag_data["page_numbers"]
    index = rag_data["faiss_index"]

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
genai.configure(api_key="AIzaSyBVhhMXHbuGeAyEN85nB2cIWQC_n5UUWAQ")
gemini = genai.GenerativeModel("gemini-2.0-flash")

def ask_question(message, history):
    # Retrieve context with page numbers
    query_embed = embed_model.encode([message])
    _, I = index.search(np.array(query_embed), k=5)
    
    # Collect pages and context
    unique_pages = set()
    context_chunks = []
    for i in I[0]:
        page = page_numbers[i]
        unique_pages.add(page)
        context_chunks.append(f"[Page {page}] {chunks[i]}")

    context = "\n".join(context_chunks)
    page_refs = sorted(unique_pages)
    
    # Build conversation history
    chat_history = "\n".join([f"User: {msg[0]}\nBot: {msg[1]}" for msg in history])
    
    # Generate answer
    prompt = f"""CONTEXT:
{context}

CONVERSATION HISTORY:
{chat_history}

QUESTION: {message}
ANSWER WITH PAGE REFERENCES:"""
    
    response = gemini.generate_content(prompt)
    answer = response.text.strip()
    
    # Add fallback references
    if "Page" not in answer and page_refs:
        answer += f"\n\n(Related pages: {', '.join(map(str, page_refs))})"
    
    return answer, page_refs

# Custom CSS for split view
css = """
#book-viewer {height: 80vh !important; overflow-y: auto !important;}
.chat-panel {height: 80vh !important;}
"""
def show_pdf(file):
    if file is None:
        return ""
    
    # Sanitize filename
    original_name = os.path.basename(file.name)
    safe_name = original_name.replace(" ", "_").replace("%", "")  # Basic sanitization
    static_path = os.path.join(STATIC_DIR, safe_name)
    
    # Copy file to static directory
    shutil.copy(file.name, static_path)
    
    # URL-encode the filename
    encoded_name = urllib.parse.quote(safe_name)
    
    # Return viewer HTML
    return f"""
    <iframe src="/static/{encoded_name}" 
            width="0%" 
            height="6px" 
            style="border: none"></iframe>
    """
with gr.Blocks(css=css) as demo:
    gr.Markdown("# 📚 Book Analysis Suite - Split View")
    
    with gr.Row():
        # Left Panel - PDF Viewer
        with gr.Column(scale=0):
            gr.Markdown("## Book PDF Viewer")
            file_input = gr.File(label="📄 Upload your PDF", file_types=[".pdf"])
            pdf_viewer = gr.HTML()
            file_input.change(
                fn=show_pdf,
                inputs=file_input,
                outputs=pdf_viewer
                )   
            
        # Right Panel - Chat Interface
        with gr.Column(scale=1, elem_id="chat-panel"):
            chatbot = gr.Chatbot(height=600)
            msg = gr.Textbox(label="Your Question")
            clear = gr.ClearButton([msg, chatbot])
    
    # Handle interactions
    def respond(message, chat_history):
        answer, pages = ask_question(message, chat_history)
        
        # Format page references for display
        page_text = "\n".join([f"• Page {p}: {chunks[i]}" for i, p in enumerate(pages)])
        page_annotations = [(f"Page {p}", str(p)) for p in pages]
        
        chat_history.append((message, answer))
        return "", chat_history, page_text, page_annotations
    
    msg.submit(
        respond,
        [msg, chatbot],
        [msg, chatbot],
    )
# Configure CORS to prevent postMessage errors
demo.app.add_middleware(
    "CORSMiddleware",
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
demo.app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Add verification endpoint
@demo.app.get("/verify-static")
async def verify_static():
    return {"files": os.listdir(STATIC_DIR)}

demo.launch(pwa=True)