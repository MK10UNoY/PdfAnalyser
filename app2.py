import gradio as gr
import pickle
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# Load RAG data
chunks, index = pickle.load(open("offline_rag.pkl", "rb"))
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Gemini setup
genai.configure(api_key="AIzaSyBVhhMXHbuGeAyEN85nB2cIWQC_n5UUWAQ")
gemini = genai.GenerativeModel("gemini-2.0-flash")

# Query logic
def ask_question(message, history):
    # Step 1: Retrieve context using RAG (no history modification)
    query_embed = embed_model.encode([message])
    _, I = index.search(np.array(query_embed), k=5)
    context = "\n\n".join([chunks[i] for i in I[0]])

    # Step 2: Build prompt using history for context (but don't modify it)
    chat_history = ""
    for msg in history:
        if msg["role"] == "user":
            chat_history += f"Q: {msg['content']}\n"
        elif msg["role"] == "assistant":
            chat_history += f"A: {msg['content']}\n"

    prompt = f"""**Context**: {context}

**Chat History**:
{chat_history}

**New Question**: {message}
"""
    # Step 3: Get Gemini response
    response = gemini.generate_content(prompt)
    return response.text.strip()  # Return only the answer string
# Gradio UI
gr.ChatInterface(
    fn=ask_question,
    title="Book Q&A Chatbot",
    chatbot=gr.Chatbot(label="BookBot", type="messages"),  # Use 'messages' type
    textbox=gr.Textbox(placeholder="Ask about the book...", label="Your Question"),
    type="messages",  # IMPORTANT: Tell ChatInterface what format we're using
).launch(pwa=True, share=True)  # PWA and share options for easy access
# Note: The above code is a simplified version of the original app.py file. It focuses on the core functionality of the chatbot, including the RAG context retrieval and Gemini response generation. The Gradio interface is set up to allow users to interact with the chatbot easily.

