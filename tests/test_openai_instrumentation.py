import openai
import math
from openai import util
from unittest import mock
from unittest.mock import create_autospec
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


class MockCompletion(EngineAPIResource):
    @classmethod
    def create(cls, *args, **kwargs):
        return {
            "id": "cmpl-123",
            "object": "text_completion",
            "created": 1677652288,
            "choices": [
                {
                    "text": "OpenTelemetry is easy to use",
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 10, "total_tokens": 19},
        }


class MockEmbedding(EngineAPIResource):
    @classmethod
    def create(cls, *args, **kwargs):
        data = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [
                        0.0023064255,
                        -0.009327292,
                        -0.0028842222,
                        # Eliding 1536 other floats
                    ],
                    "index": 0,
                }
            ],
            "model": "text-embedding-ada-002",
            "usage": {
                "prompt_tokens": 8,
                "total_tokens": 8,
            },
        }
        return util.convert_to_openai_object(data)


class TestOpenAIInstrumentation(TestBase):
    def _assert_spans(self, num_spans: int):
        finished_spans = self.memory_exporter.get_finished_spans()
        self.assertEqual(num_spans, len(finished_spans))
        if num_spans == 0:
            return None
        if num_spans == 1:
            return finished_spans[0]
        return finished_spans

    def _call_chat(self):
        return openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "tell me a joke about opentelemetry"}
            ],
            temperature=0.0,
            max_tokens=150,
            user="test",
        )

    # @mock.patch(
    #     # "openai.api_resources.abstract.engine_api_resource.EngineAPIResource.create",
    #     "openai.ChatCompletion.create",
    #     autospec=True,
    #     # new_callable=MockChatCompletion.create,
    # )
    def test_instrument_chat(self):
        mock_chat_completion = create_autospec(openai.ChatCompletion)
        mock_chat_completion.create = MockChatCompletion.create
        with mock.patch("openai.ChatCompletion", new=mock_chat_completion):
            OpenAIInstrumentor().instrument()
            result = self._call_chat()

            span = self._assert_spans(1)
            name = "openai.chat"
            self.assertEqual(span.name, name)
            self.assertEqual(span.attributes[f"{name}.model"], "gpt-3.5-turbo")
            self.assertEqual(
                span.attributes[f"{name}.messages.0.role"],
                "user",
            )
            self.assertEqual(
                span.attributes[f"{name}.messages.0.content"],
                "tell me a joke about opentelemetry",
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
            self.assertEqual(span.attributes[f"{name}.user"], "test")

            self.assertEqual(span.attributes[f"{name}.response.id"], result["id"])
            self.assertEqual(
                span.attributes[f"{name}.response.object"], result["object"]
            )
            self.assertEqual(
                span.attributes[f"{name}.response.created"], result["created"]
            )
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

            OpenAIInstrumentor().uninstrument()

    def test_instrument_completion(self):
        mock_completion = create_autospec(openai.Completion)
        mock_completion.create = MockCompletion.create
        with mock.patch("openai.Completion", new=mock_completion):
            OpenAIInstrumentor().instrument()
            result = openai.Completion.create(
                model="text-davinci-003",
                prompt="tell me a joke about opentelemetry",
                user="test",
            )

            span = self._assert_spans(1)
            name = "openai.completion"
            self.assertEqual(span.name, name)
            self.assertEqual(span.attributes[f"{name}.model"], "text-davinci-003")
            self.assertEqual(
                span.attributes[f"{name}.prompt"],
                "tell me a joke about opentelemetry",
            )
            self.assertEqual(span.attributes[f"{name}.temperature"], 1.0)
            self.assertEqual(span.attributes[f"{name}.top_p"], 1.0)
            self.assertEqual(span.attributes[f"{name}.n"], 1)
            self.assertEqual(span.attributes[f"{name}.stream"], False)
            self.assertEqual(span.attributes[f"{name}.stop"], "")
            self.assertEqual(span.attributes[f"{name}.max_tokens"], math.inf)
            self.assertEqual(span.attributes[f"{name}.presence_penalty"], 0.0)
            self.assertEqual(span.attributes[f"{name}.frequency_penalty"], 0.0)
            self.assertEqual(span.attributes[f"{name}.logit_bias"], "")
            self.assertEqual(span.attributes[f"{name}.user"], "test")

            self.assertEqual(span.attributes[f"{name}.response.id"], result["id"])
            self.assertEqual(
                span.attributes[f"{name}.response.object"], result["object"]
            )
            self.assertEqual(
                span.attributes[f"{name}.response.created"], result["created"]
            )
            self.assertEqual(
                span.attributes[f"{name}.response.choices.0.text"],
                result["choices"][0]["text"],
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

            OpenAIInstrumentor().uninstrument()

    def test_instrument_embedding(self):
        mock_embedding = create_autospec(openai.Embedding)
        mock_embedding.create = MockEmbedding.create
        with mock.patch("openai.Embedding", new=mock_embedding):
            OpenAIInstrumentor().instrument()
            result = openai.Embedding.create(
                model="text-embedding-ada-002",
                input="The food was delicious and the waiter...",
                user="test",
            )

            span = self._assert_spans(1)
            name = "openai.embedding"
            self.assertEqual(span.name, name)
            self.assertEqual(span.attributes[f"{name}.model"], "text-embedding-ada-002")
            self.assertEqual(span.attributes[f"{name}.input_count"], 40)
            self.assertEqual(span.attributes[f"{name}.user"], "test")
            # TODO: lol this doesn't work
            # the patching required for how openai's thing works here is too much for my brain
            # print(result)
            # self.assertEqual(
            #     span.attributes[f"{name}.response.embeddings_count"], len(result.data)
            # )
            # self.assertEqual(
            #     span.attributes[f"{name}.response.usage.promt_tokens"],
            #     result.usage.prompt_token,
            # )
            # self.assertEqual(
            #     span.attributes[f"{name}.response.usage.total_tokens"],
            #     result.usage.total_tokens,
            # )

            OpenAIInstrumentor().uninstrument()
