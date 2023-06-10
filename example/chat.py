import os
import openai
from dotenv import load_dotenv
from opentelemetry.instrumentation.openai import OpenAIInstrumentor

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

OpenAIInstrumentor().instrument()

openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role":"user", "content":"Tell me a joke about opentelemetry"}],
)
