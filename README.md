# opentelemetry-instrument-openai

It's OpenTelemetry instrumentation (python) for OpenAI's library.

Pproject site: https://github.com/cartermp/opentelemetry-instrument-openai-py

## How to use it

Simple! First, install this package.

### Autoinstrumentation

```
opentelemetry-bootstrap -a install
opentelemetry-instrument \
  --traces_exporter console \
  --metrics_exporter none \
  --logs_exporter none \
  python chat.py
```

If you're using poetry, put `poetry run` before each command.

### In code

It's one line of code too:

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

## How to develop

Get [poetry](https://python-poetry.org/). Python build and dependency management is fucked but this is about as good as it gets.

Now install and run tests:

```
poetry install
poetry run pytest
```

Now you can develop and run tests as you go!

## How to run the example

Click the example folder and read the README.