# ollama_feeling_safe.py
# Enhanced version with error handling and cleaner exit

import ollama

# Define the emotional persona
persona_prompt = """
You are a warm, empathetic mentor.
Always respond with encouragement, positivity, and emotional awareness.
Use supportive language, motivational tone, and show understanding.
"""

def chat_with_feeling():
    print("💡 Emotional Ollama Chat Started")
    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("👋 Ending session.")
                break

            try:
                response = ollama.chat(
                    model="llama3.2",   # use "llama3.2" or "llama3.1" for newer versions
                    messages=[
                        {"role": "system", "content": persona_prompt},
                        {"role": "user", "content": user_input}
                    ]
                )
                print("Ollama:", response["message"]["content"])
            except Exception as e:
                print("⚠️ Error communicating with Ollama:", str(e))

    except KeyboardInterrupt:
        print("\n👋 Session interrupted. Goodbye!")

if __name__ == "__main__":
    chat_with_feeling()
