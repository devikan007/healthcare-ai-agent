import os
import re
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.runnables import RunnableSequence
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import LlamaCpp

from langchain.memory import ConversationBufferMemory
from knowledge_base import load_or_create_knowledge_base

# Load environment variables
load_dotenv()

# Load LLaMA model
llm = LlamaCpp(
    model_path="models/llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=1024,
    temperature=0.3,
    max_tokens=512,
    stop=["</s>", "User:", "Assistant:"],
    verbose=False,
)

# Embeddings and knowledge base
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = load_or_create_knowledge_base()

# Initialize memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# Filter unsafe queries
def is_unsafe_query(text: str) -> bool:
    patterns = [
        r"how.{1,20}(make|create|produce|synthesize|manufacture).{1,20}(drug|cocaine|heroin|meth|fentanyl)",
        r"(suicide|kill myself|end my life|take my own life)",
        r"how.{1,20}(harm|hurt|injure|damage).{1,20}(people|someone|patient|child)"
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)

# Simplified prompt (LLaMA-friendly)
def get_prompt():
    return ChatPromptTemplate.from_template("""
You are a friendly AI healthcare assistant. Use the medical context to help answer the user's question.
Keep it general, safe, and helpful.

Context:
{context}

Question:
{question}

Helpful Answer:
""")

# Pipeline
def get_pipeline():
    return RunnableSequence(get_prompt() | llm)

# Chatbot core logic
def generate_response(user_input: str):
    cleaned_input = re.sub(r"[^\w\s.,?!;:()\[\]{}'\"-]", "", user_input)

    if is_unsafe_query(cleaned_input):
        return {
            "response": "I'm sorry, I can't help with that topic. Please contact medical professionals or emergency services if needed.",
            "context": "Filtered due to safety policy.",
            "confidence": 10
        }

    relevant_docs = vector_store.similarity_search(cleaned_input, k=2)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    if not context.strip():
        context = "No medical context found."

    pipeline = get_pipeline()
    formatted_prompt = {
        "context": context,
        "question": cleaned_input
    }

    # Save user message to memory
    st.session_state.memory.chat_memory.add_message(HumanMessage(content=cleaned_input))

    # LLM Invocation
    try:
        raw_response = pipeline.invoke(formatted_prompt)
        print("Raw Response from LLM:", raw_response)  # DEBUG
        if hasattr(raw_response, 'content'):
            final_response = raw_response.content.strip()
        else:
            final_response = str(raw_response).strip()

        if not final_response:
            final_response = "I'm not sure how to respond. Please consult a healthcare professional."
    except Exception as e:
        print("LLM Invocation Error:", str(e))
        final_response = "Sorry, I couldn't process your question right now. Please try again later."

    # Save assistant message
    st.session_state.memory.chat_memory.add_message(AIMessage(content=final_response))

    return {
        "response": final_response,
        "context": context,
        "confidence": 7  # You can add a real confidence score later if needed
    }

# Streamlit UI
def main():
    st.set_page_config(page_title="Healthcare AI Assistant", layout="centered")
    st.title("🩺 Healthcare Assistant (LLaMA)")
    st.write("Ask me general health questions. I'm not a substitute for professional advice!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_query := st.chat_input("Enter a health-related question..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                result = generate_response(user_query)
                st.markdown(result["response"])
                with st.expander("More Info"):
                    st.write(f"Confidence: {result['confidence']}/10")
                    st.code(result["context"])

        st.session_state.messages.append({"role": "assistant", "content": result["response"]})

if __name__ == "__main__":
    main()
