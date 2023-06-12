import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role":"user", "content":"Tell me a joke about opentelemetry"}],
)

openai.Embedding.create(
    model="text-embedding-ada-002",
    input=["poop","pee"],
    user="test",
)
