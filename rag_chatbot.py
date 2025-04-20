import os
from dotenv import load_dotenv
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.llms import LlamaCpp
from knowledge_base import load_or_create_knowledge_base

# Load environment variables
load_dotenv()

# ======= ✅ Load LLaMA model from GGUF =======
MODEL_PATH = r"C:\Users\Devika\Desktop\Langchain\RAG\Exercise_2\models\llama-2-7b-chat.Q4_K_M.gguf"

llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.7,
    max_tokens=512,
    top_p=0.95,
    n_ctx=2048,
    verbose=True
)

# ======= ✅ Memory & Knowledge Base =======
memory = ConversationBufferMemory(return_messages=True)
vector_store = load_or_create_knowledge_base()

def get_rag_response(user_input):
    """Get a RAG-enhanced response by retrieving relevant documents first"""
    memory.chat_memory.add_message(HumanMessage(content=user_input))
    
    relevant_docs = vector_store.similarity_search(user_input, k=2)
    
    context = "\n".join([
        f"Topic: {doc.metadata.get('topic', 'General Health')}\n{doc.page_content}" 
        for doc in relevant_docs
    ])
    
    messages = memory.chat_memory.messages
    
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        messages.insert(0, SystemMessage(content=""" 
        You are a helpful healthcare assistant. Provide information based on the medical context provided.
        Always remind users that you're not a substitute for professional medical advice.
        If you're unsure or don't have enough information, acknowledge this clearly.
        """))
    
    prompt_template = """
    Answer the user's question based on the following medical context. If the answer is not in the context, 
    provide general health information but make it clear that you don't have specific information on that topic.
    
    Medical Context:
    {context}
    
    Recent conversation history:
    {chat_history}
    
    User Question: {question}
    """
    
    chat_history = ""
    for msg in messages[1:]:  # Skip system message
        if isinstance(msg, HumanMessage):
            chat_history += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            chat_history += f"Assistant: {msg.content}\n"
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    formatted_prompt = prompt.format(
        context=context,
        chat_history=chat_history,
        question=user_input
    )
    
    # Directly generate response
    response = llm.invoke(formatted_prompt)
    memory.chat_memory.add_message(AIMessage(content=response))
    
    return {
        "response": response,
        "retrieved_context": context
    }

def chat_loop():
    print("Healthcare Assistant: Hello! I'm your healthcare assistant with access to medical information. How can I help you today?")
    print("(Type 'exit' to end the conversation)")

    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("\nHealthcare Assistant: Goodbye! Take care of your health.")
            break
            
        result = get_rag_response(user_input)
        print(f"\nHealthcare Assistant: {result['response']}")

        # Optional debug
        # print(f"\n[DEBUG] Retrieved Context: {result['retrieved_context']}")

if __name__ == "__main__":
    chat_loop()
