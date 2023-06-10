import openai
from unittest import mock
from openai.api_resources.abstract.engine_api_resource import EngineAPIResource
from opentelemetry.instrumentation.openai import OpenAIInstrumentator
from opentelemetry.test.test_base import TestBase


class MockChatCompletion(EngineAPIResource):
    @classmethod
    def create(cls, *args, **kwargs):
        return "mocked"


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
            messages=[{"role": "user", "content": "tell me a joke about opentelemetry"}],
            temperature=0.0,
            max_tokens=150,
            name="test",
        )

    @mock.patch(
        "openai.api_resources.abstract.engine_api_resource.EngineAPIResource.create",
        new=MockChatCompletion.create,
    )
    def test_instrumentat(self):
        OpenAIInstrumentator().instrument()
        self.call_chat()

        span = self.assert_spans(1)
        name = "openai.chat"
        self.assertEqual(span.name, name)
        self.assertEqual(span.attributes[f"{name}.model"], "gpt-3.5-turbo")
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

    @mock.patch(
        "openai.api_resources.abstract.engine_api_resource.EngineAPIResource.create",
        new=MockChatCompletion.create,
    )
    def uninstrument(self):
        OpenAIInstrumentator().uninstrument()
        self.call_chat()
        self.assert_spans(0)

        OpenAIInstrumentator().instrument()
        self.call_chat()
        self.assert_spans(1)
