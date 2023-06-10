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

    @mock.patch(
        "openai.api_resources.abstract.engine_api_resource.EngineAPIResource.create",
        new=MockChatCompletion.create,
    )
    def test_instrumentat(self):
        OpenAIInstrumentator().instrument()
        openai.ChatCompletion.create()

        span = self.assert_spans(1)
        self.assertEqual(span.name, "openai.chat")

    @mock.patch(
        "openai.api_resources.abstract.engine_api_resource.EngineAPIResource.create",
        new=MockChatCompletion.create,
    )
    def uninstrument(self):
        OpenAIInstrumentator().uninstrument()
        openai.ChatCompletion.create()
        self.assert_spans(0)

        OpenAIInstrumentator().instrument()
        openai.ChatCompletion.create()
        self.assert_spans(1)
