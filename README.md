# opentelemetry-instrument-openai

It's OpenTelemetry instrumentation (python) for OpenAI's library.

Project site: https://github.com/cartermp/opentelemetry-instrument-openai-py

Supported APIs:

- [x] Chat
- [x] Embeddings
- [x] Moderation
- [x] Image (generation, edit, variation)
- [x] Audio (transcribe, translate)
- [x] Completion (GPT-3)
- [x] Edit (GPT-3)

## How to use it

Simple! First, install this package.

### Usage

With autoinstrumentation agent:

```
poetry add opentelemetry-instrument-openai
poetry run opentelemetry-bootstrap -a install
poetry run opentelemetry-instrument python your_app.py
```

If you prefer to do it in code, you can do that too:

```python
import openai
from dotenv import load_dotenv
from opentelemetry.instrument.openai import OpenAIInstrumentor

OpenAIInstrumentor().instrument()

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role":"user", "content":"Tell me a joke about opentelemetry"}],
)
```

You can then run your app normally with your own opentelemetry initialization.

## How to develop

Get [poetry](https://python-poetry.org/).

Now install and run tests:

```
poetry install
poetry run pytest
```

Now you can develop and run tests as you go!

## How to run the example

Click the example folder and read the README.