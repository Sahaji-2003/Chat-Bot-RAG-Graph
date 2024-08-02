import os
from groq import Groq

# Ensure that the GROQ_API_KEY environment variable is set
os.environ["GROQ_API_KEY"] = "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"  # Replace with your actual Groq API key

def chat_with_groq(prompt):
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-8b-8192",  # Replace with the actual model name if different
    )

    return chat_completion.choices[0].message.content

def main():
    print("Welcome to the Groq Chatbot!")
    while True:
        prompt = input("You: ")
        if prompt.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        response = chat_with_groq(prompt)
        print(f"Groq: {response}")

if __name__ == "__main__":
    main()
