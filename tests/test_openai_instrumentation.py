import openai
from opentelemetry.instrumentation.openai import OpenAIInstrumentation
from opentelemetry.test.test_base import TestBase


# mock the openai ChatCompletion class
class MockChatCompletion:
    def create(self):
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

    def test_instrumentation(self):
        instrumentation = OpenAIInstrumentation()
        instrumentation.instrument()

        # mock the openai ChatCompletion class
        openai.ChatCompletion = MockChatCompletion
        openai.ChatCompletion().create()

        self.assert_spans(1)
