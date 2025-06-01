import gradio as gr
import pickle
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# Load RAG data with page numbers
with open("vector_store.pkl", "rb") as f:
    rag_data = pickle.load(f)
    chunks = rag_data["chunk_texts"]
    page_numbers = rag_data["page_numbers"]  # Assume this exists in your pkl
    index = rag_data["faiss_index"]

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Gemini setup
genai.configure(api_key="#########################")
gemini = genai.GenerativeModel("gemini-2.0-flash")

def ask_question(message, history):
    # Retrieve context with page numbers
    query_embed = embed_model.encode([message])
    _, I = index.search(np.array(query_embed), k=5)
    
    # Collect unique pages and format context
    unique_pages = set()
    context_chunks = []
    for i in I[0]:
        page = page_numbers[i]
        unique_pages.add(page)
        context_chunks.append(f"[Page {page}] {chunks[i]}")

    context = "\n".join(context_chunks)
    page_refs = ", ".join(map(str, sorted(unique_pages)))

    # Build conversation history
    chat_history = "\n".join(
        [f"{msg[0].capitalize()}: {msg[1]}"  # Access tuple elements instead of dict
         for msg in (history[-4:] if history else [])]
    )

    # Create citation-aware prompt
    prompt = f"""Use this context and conversation history to answer. Always cite pages using [Page X].

CONTEXT:
{context}

CONVERSATION HISTORY:
{chat_history}

NEW QUESTION: {message}
ANSWER:"""

    # Generate response
    response = gemini.generate_content(prompt)
    answer = response.text.strip()
    
    # Add page references if missing
    if "Page" not in answer and page_refs:
        answer += f"\n\n(References: Pages {page_refs})"
    
    return answer

# Gradio Chat UI
gr.ChatInterface(
    fn=ask_question,
    title="📚 Book Q&A with Page Citations",
    chatbot=gr.Chatbot(label="Book Expert", render_markdown=True),
    textbox=gr.Textbox(placeholder="Ask about the book...", 
                      label="Your Question"),
    examples=["Explain Fourier analysis with page references",
             "What does page 45 say about circuit design?"],
).launch(pwa=True)  # PWA and share options for easy access
# Note: The above code is a simplified version of the original app.py file. It focuses on the core functionality of the chatbot, including the RAG context retrieval and Gemini response generation. The Gradio interface is set up to allow users to interact with the chatbot easily.
