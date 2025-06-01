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
def ask_question(question):
    query_embed = embed_model.encode([question])
    _, I = index.search(np.array(query_embed), k=5)
    context = "\n\n".join([chunks[i] for i in I[0]])

    prompt = f"""Use the following context to answer the question:

Context:
{context}

Question: {question}
"""

    response = gemini.generate_content(prompt)
    return response.text

# Gradio UI
gr.Interface(
    fn=ask_question,
    inputs=gr.Textbox(label="Ask a question from the book"),
    outputs=gr.Textbox(label="Gemini Answer"),
    title="Offline Book QA with Gemini",
    description="Ask questions using the book and Gemini 1.5 Flash"
).launch()
