from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

models = client.models.list()

print("Available Groq Models:\n")
for m in models.data:
    print(m.id)
