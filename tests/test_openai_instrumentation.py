import openai
from unittest import mock
from openai.api_resources.abstract.engine_api_resource import EngineAPIResource
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.test.test_base import TestBase


class MockChatCompletion(EngineAPIResource):
    @classmethod
    def create(cls, *args, **kwargs):
        return {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "\n\nOpenTelemetry is easy to use",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 10, "total_tokens": 19},
        }


class TestOpenAIInstrumentation(TestBase):
    def assert_spans(self, num_spans: int):
        finished_spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(num_spans, len(finished_spans))
        if num_spans == 0:
            return None
        if num_spans == 1:
            return finished_spans[0]
        return finished_spans

    def call_chat(self):
        return openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "tell me a joke about opentelemetry"}
            ],
            temperature=0.0,
            max_tokens=150,
            name="test",
        )

    @mock.patch(
        "openai.api_resources.abstract.engine_api_resource.EngineAPIResource.create",
        new=MockChatCompletion.create,
    )
    def test_instrument(self):
        OpenAIInstrumentor().instrument()
        result = self.call_chat()

        span = self.assert_spans(1)
        name = "openai.chat"
        self.assertEqual(span.name, name)
        self.assertEqual(span.attributes[f"{name}.model"], "gpt-3.5-turbo")
        self.assertEqual(
            span.attributes[f"{name}.messages"],
            "user: tell me a joke about opentelemetry\n",
        )
        self.assertEqual(span.attributes[f"{name}.temperature"], 0.0)
        self.assertEqual(span.attributes[f"{name}.top_p"], 1.0)
        self.assertEqual(span.attributes[f"{name}.n"], 1)
        self.assertEqual(span.attributes[f"{name}.stream"], False)
        self.assertEqual(span.attributes[f"{name}.stop"], "")
        self.assertEqual(span.attributes[f"{name}.max_tokens"], 150)
        self.assertEqual(span.attributes[f"{name}.presence_penalty"], 0.0)
        self.assertEqual(span.attributes[f"{name}.frequency_penalty"], 0.0)
        self.assertEqual(span.attributes[f"{name}.logit_bias"], "")
        self.assertEqual(span.attributes[f"{name}.name"], "test")

        self.assertEqual(span.attributes[f"{name}.response.id"], result["id"])
        self.assertEqual(span.attributes[f"{name}.response.object"], result["object"])
        self.assertEqual(span.attributes[f"{name}.response.created"], result["created"])
        self.assertEqual(
            span.attributes[f"{name}.response.choices.0.message.role"],
            result["choices"][0]["message"]["role"],
        )
        self.assertEqual(
            span.attributes[f"{name}.response.choices.0.message.content"],
            result["choices"][0]["message"]["content"],
        )
        self.assertEqual(
            span.attributes[f"{name}.response.choices.0.finish_reason"],
            result["choices"][0]["finish_reason"],
        )
        self.assertEqual(
            span.attributes[f"{name}.response.usage.prompt_tokens"],
            result["usage"]["prompt_tokens"],
        )
        self.assertEqual(
            span.attributes[f"{name}.response.usage.completion_tokens"],
            result["usage"]["completion_tokens"],
        )
        self.assertEqual(
            span.attributes[f"{name}.response.usage.total_tokens"],
            result["usage"]["total_tokens"],
        )

    @mock.patch(
        "openai.api_resources.abstract.engine_api_resource.EngineAPIResource.create",
        new=MockChatCompletion.create,
    )
    def uninstrument(self):
        OpenAIInstrumentor().uninstrument()
        self.call_chat()
        self.assert_spans(0)

        OpenAIInstrumentor().instrument()
        self.call_chat()
        self.assert_spans(1)
