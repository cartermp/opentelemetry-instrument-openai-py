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
    "name": "HTTP POST",
    "context": {
        "trace_id": "0x972ad0fce08102c888e2813a3fa0e927",
        "span_id": "0xa0d21a44d6704eec",
        "trace_state": "[]"
    },
    "kind": "SpanKind.CLIENT",
    "parent_id": "0xbf271dae7f8f92d4",
    "start_time": "2023-06-10T17:54:29.606964Z",
    "end_time": "2023-06-10T17:54:30.990708Z",
    "status": {
        "status_code": "UNSET"
    },
    "attributes": {
        "http.method": "POST",
        "http.url": "https://api.openai.com/v1/chat/completions",
        "http.status_code": 200
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
{
    "name": "openai.chat",
    "context": {
        "trace_id": "0x972ad0fce08102c888e2813a3fa0e927",
        "span_id": "0xbf271dae7f8f92d4",
        "trace_state": "[]"
    },
    "kind": "SpanKind.INTERNAL",
    "parent_id": null,
    "start_time": "2023-06-10T17:54:29.589859Z",
    "end_time": "2023-06-10T17:54:30.995015Z",
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
        "openai.chat.messages": "user: Tell me a joke about opentelemetry\n",
        "openai.chat.response.id": "chatcmpl-7Px5Bhbd1neL9Zqivfd9nYStycMDL",
        "openai.chat.response.object": "chat.completion",
        "openai.chat.response.created": 1686419669,
        "openai.chat.response.choices.0.message.role": "assistant",
        "openai.chat.response.choices.0.message.content": "Why was the developer always happy when working with OpenTelemetry? Because he could always trace his steps back to where he went wrong!",
        "openai.chat.response.choices.0.finish_reason": "stop",
        "openai.chat.response.usage.prompt_tokens": 16,
        "openai.chat.response.usage.completion_tokens": 27,
        "openai.chat.response.usage.total_tokens": 43
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

That's a trace with two spans!