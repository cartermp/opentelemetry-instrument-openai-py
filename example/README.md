How to run:

1. Reame `env.example` to `.env`
2. Put your openai API key in there

Now:

```shell
poetry install
poetry run opentelemetry-bootstrap
poetry run opentelemetry-instrument --traces_exporter console --metrics_exporter none --logs_exporter none python chat.py
```

You will see a response like this:

```json
{
    "name": "openai.chat",
    "context": {
        "trace_id": "0xe87062cbe18067301d34cb87d09da4c7",
        "span_id": "0x4dacd658adc7ee5d",
        "trace_state": "[]"
    },
    "kind": "SpanKind.INTERNAL",
    "parent_id": null,
    "start_time": "2023-06-10T18:10:01.869155Z",
    "end_time": "2023-06-10T18:10:03.030913Z",
    "status": {
        "status_code": "UNSET"
    },
    "attributes": {
        "openai.chat.model": "gpt-3.5-turbo",
        "openai.chat.temperature": 1.0,
        "openai.chat.top_p": 1.0,
        "openai.chat.n": 1,
        "openai.chat.stream": false,
        "openai.chat.stop": "",
        "openai.chat.max_tokens": Infinity,
        "openai.chat.presence_penalty": 0.0,
        "openai.chat.frequency_penalty": 0.0,
        "openai.chat.logit_bias": "",
        "openai.chat.name": "",
        "openai.chat.messages.0.role": "user",
        "openai.chat.messages.0.content": "Tell me a joke about opentelemetry",
        "openai.chat.response.id": "chatcmpl-7PxKEDOVgBHYQPxZqposuEU93SLbd",
        "openai.chat.response.object": "chat.completion",
        "openai.chat.response.created": 1686420602,
        "openai.chat.response.choices.0.message.role": "assistant",
        "openai.chat.response.choices.0.message.content": "Why did Opentelemetry cross the road? To trace its steps!",
        "openai.chat.response.choices.0.finish_reason": "stop",
        "openai.chat.response.usage.prompt_tokens": 16,
        "openai.chat.response.usage.completion_tokens": 14,
        "openai.chat.response.usage.total_tokens": 30
    },
    "events": [],
    "links": [],
    "resource": {
        "attributes": {
            "telemetry.sdk.language": "python",
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.version": "1.18.0",
            "telemetry.auto.version": "0.39b0",
            "service.name": "unknown_service"
        },
        "schema_url": ""
    }
}
```

To see it in honeycomb:

```
export OTEL_EXPORTER_OTLP_ENDPOINT="https://api.honeycomb.io"
export OTEL_SERVICE_NAME="your-service-name"
poetry run opentelemetry-instrument python chat.py
```

And then you can query it and see a trace (in this case it's just one span so I'm not showing that):

![](honeycomb-openai-query.png)