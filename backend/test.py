import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

print("API key present:", bool(api_key))

if not api_key:
    raise ValueError("GROQ_API_KEY not found")

client = Groq(api_key=api_key)

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": "Explain what OCR is in exactly 3 sentences."
        }
    ]
)

print("\nResponse:")
print(response.choices[0].message.content)

print("\nUsage:")
print(response.usage)