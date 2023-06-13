import os
import openai
from dotenv import load_dotenv
from opentelemetry import trace

# If you don't want to use full autoinstrumentation, just add this:
#
# from opentelemetry.instrumentation.openai import OpenAIInstrumentor
# OpenAIInstrumentor().instrument()

tracer = trace.get_tracer("chat.demo")

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

with tracer.start_as_current_span("example") as span:
    span.set_attribute("attr1", 12)
    openai.Embedding.create(
        model="text-embedding-ada-002",
        input=["some text"],
        user="test",
    )

    openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    openai.Completion.create(
        model="text-davinci-003",
        prompt="Tell me a joke about opentelemetry",
    )

    openai.Edit.create(
        model="text-davinci-edit-001",
        input="What day of the wek is it?",
        instruction="Fix the spelling mistakes",
    )

    mod_response = openai.Moderation.create(
        model="text-moderation-latest",
        input=["This text shouldn't be flagged, right?"],
    )
