from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Initialize the Ollama-based LLaMA model
llm = ChatOllama(model="llama2")

# Set up memory for conversation
memory = ConversationBufferMemory(return_messages=True)

# Create the conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)

# Initial system message
def print_welcome():
    print("Healthcare Assistant: Hello! I'm your healthcare assistant. How can I help you today?")
    print("(Type 'exit' to end the conversation)\n")

def get_chatbot_response(user_input):
    # Feed the user input into the conversation chain
    return conversation.predict(input=user_input)

def chat_loop():
    print_welcome()
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Healthcare Assistant: Take care! Goodbye. 😊")
            break
        response = get_chatbot_response(user_input)
        print(f"Healthcare Assistant: {response}\n")

if __name__ == "__main__":
    chat_loop()
