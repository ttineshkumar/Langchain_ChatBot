
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# 1. Load variables from the .env file
load_dotenv()


# 2. Check whether the API key is available
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY is missing. "
        "Please add it to your .env file."
    )


# 3. Configure the AI model
model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

llm = ChatOpenAI(
    model=model_name,
    temperature=0.7,
    api_key=api_key,
)


# 4. Create a prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant. "
            "Answer clearly and use simple English."
        ),
        ("human", "{question}"),
    ]
)


# 5. Convert the model response into plain text
output_parser = StrOutputParser()


# 6. Build the LangChain pipeline
chain = prompt | llm | output_parser


# 7. Create the chatbot function
def ask_question(question: str) -> str:
    """Send a question to the AI model and return its answer."""

    question = question.strip()

    if not question:
        return "Please enter a question."

    response = chain.invoke({"question": question})
    return response


# 8. Run the chatbot
def main():
    print("=" * 45)
    print("Welcome to the LangChain Chatbot!")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 45)

    while True:
        question = input("\nYou: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Chatbot: Goodbye!")
            break

        try:
            answer = ask_question(question)
            print(f"\nChatbot: {answer}")

        except Exception as error:
            print(f"\nError: {error}")
            print(
                "Check your API key, internet connection, "
                "model access, and API usage limits."
            )


if __name__ == "__main__":
    main()