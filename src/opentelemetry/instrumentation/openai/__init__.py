# Copyright Phillip Carter
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
OpenTelemetry instrumentation for OpenAI's client library.

Usage
-----
Instrument all OpenAI client calls:

.. code-block:: python

    import openai
    from opentelemetry.instrumentation.openai import OpenAIInstrumentor

    # Enable instrumentation
    OpenAIInstrumentor().instrument()

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "tell me a joke about opentelemetry"}],
    )
"""
import math
from typing import Collection

import wrapt
import openai

from opentelemetry import context as context_api
from opentelemetry.trace import get_tracer, Tracer, SpanKind

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)
from opentelemetry.instrumentation.openai.package import _instruments
from opentelemetry.instrumentation.openai.version import __version__


def _instrument_chat(tracer: Tracer):
    def _instrumented_create(wrapped, instance, args, kwargs):
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return

        name = "openai.chat"
        with tracer.start_as_current_span(name, kind=SpanKind.CLIENT) as span:
            span.set_attribute(f"{name}.model", kwargs["model"])
            span.set_attribute(
                f"{name}.temperature",
                kwargs["temperature"] if "temperature" in kwargs else 1.0,
            )
            span.set_attribute(
                f"{name}.top_p", kwargs["top_p"] if "top_p" in kwargs else 1.0
            )
            span.set_attribute(f"{name}.n", kwargs["n"] if "n" in kwargs else 1)
            span.set_attribute(
                f"{name}.stream", kwargs["stream"] if "stream" in kwargs else False
            )
            span.set_attribute(
                f"{name}.stop", kwargs["stop"] if "stop" in kwargs else ""
            )
            span.set_attribute(
                f"{name}.max_tokens",
                kwargs["max_tokens"] if "max_tokens" in kwargs else math.inf,
            )
            span.set_attribute(
                f"{name}.presence_penalty",
                kwargs["presence_penalty"] if "presence_penalty" in kwargs else 0.0,
            )
            span.set_attribute(
                f"{name}.frequency_penalty",
                kwargs["frequency_penalty"] if "frequency_penalty" in kwargs else 0.0,
            )
            span.set_attribute(
                f"{name}.logit_bias",
                kwargs["logit_bias"] if "logit_bias" in kwargs else "",
            )
            span.set_attribute(
                f"{name}.user", kwargs["user"] if "user" in kwargs else ""
            )
            for index, message in enumerate(kwargs["messages"]):
                span.set_attribute(f"{name}.messages.{index}.role", message["role"])
                span.set_attribute(
                    f"{name}.messages.{index}.content", message["content"]
                )

            response = wrapped(*args, **kwargs)

            span.set_attribute(f"{name}.response.id", response["id"])
            span.set_attribute(f"{name}.response.object", response["object"])
            span.set_attribute(f"{name}.response.created", response["created"])
            for index, choice in enumerate(response["choices"]):
                span.set_attribute(
                    f"{name}.response.choices.{index}.message.role",
                    choice["message"]["role"],
                )
                span.set_attribute(
                    f"{name}.response.choices.{index}.message.content",
                    choice["message"]["content"],
                )
                span.set_attribute(
                    f"{name}.response.choices.{index}.finish_reason",
                    choice["finish_reason"],
                )

            span.set_attribute(
                f"{name}.response.usage.prompt_tokens",
                response["usage"]["prompt_tokens"],
            )
            span.set_attribute(
                f"{name}.response.usage.completion_tokens",
                response["usage"]["completion_tokens"],
            )
            span.set_attribute(
                f"{name}.response.usage.total_tokens", response["usage"]["total_tokens"]
            )

        return response

    wrapt.wrap_function_wrapper(openai.ChatCompletion, "create", _instrumented_create)


def _instrument_embedding(tracer: Tracer):
    def _instrumented_create(wrapped, instance, args, kwargs):
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return

        name = "openai.embedding"
        with tracer.start_as_current_span(name, kind=SpanKind.CLIENT) as span:
            print(kwargs)
            span.set_attribute(f"{name}.model", kwargs["model"])
            span.set_attribute(f"{name}.input_count", len(kwargs["input"]))
            span.set_attribute(
                f"{name}.user", kwargs["user"] if "user" in kwargs else ""
            )

            response = wrapped(*args, **kwargs)

            span.set_attribute(f"{name}.response.embeddings_count", len(response.data))
            span.set_attribute(
                f"{name}.response.usage.prompt_tokens", response.usage.prompt_tokens
            )
            span.set_attribute(
                f"{name}.response.usage.total_tokens", response.usage.total_tokens
            )

    wrapt.wrap_function_wrapper(openai.Embedding, "create", _instrumented_create)


def _instrument_completions(tracer: Tracer):
    def _instrumented_create(wrapped, instance, args, kwargs):
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return

        name = "openai.completion"
        with tracer.start_as_current_span(name, kind=SpanKind.CLIENT) as span:
            span.set_attribute(f"{name}.model", kwargs["model"])
            if "prompt" in kwargs:
                if type(kwargs["prompt"]) == str:
                    span.set_attribute(f"{name}.prompt", kwargs["prompt"])
                else:
                    for index, prompt in enumerate(kwargs["prompt"]):
                        span.set_attribute(f"{name}.prompt.{index}", prompt)
            span.set_attribute(
                f"{name}.suffix",
                kwargs["suffix"] if "suffix" in kwargs else "",
            )
            span.set_attribute(
                f"{name}.temperature",
                kwargs["temperature"] if "temperature" in kwargs else 1.0,
            )
            span.set_attribute(
                f"{name}.top_p", kwargs["top_p"] if "top_p" in kwargs else 1.0
            )
            span.set_attribute(f"{name}.n", kwargs["n"] if "n" in kwargs else 1)
            span.set_attribute(
                f"{name}.stream", kwargs["stream"] if "stream" in kwargs else False
            )
            # TODO: logprobs is optional, not a concept in otel?
            if "logprobs" in kwargs and kwargs["logprobs"] is not None:
                span.set_attribute(
                    f"{name}.logprobs",
                    kwargs["logprobs"] if "logprobs" in kwargs else -1,
                )
            span.set_attribute(
                f"{name}.echo", kwargs["echo"] if "echo" in kwargs else False
            )
            span.set_attribute(
                f"{name}.stop", kwargs["stop"] if "stop" in kwargs else ""
            )
            span.set_attribute(
                f"{name}.max_tokens",
                kwargs["max_tokens"] if "max_tokens" in kwargs else math.inf,
            )
            span.set_attribute(
                f"{name}.presence_penalty",
                kwargs["presence_penalty"] if "presence_penalty" in kwargs else 0.0,
            )
            span.set_attribute(
                f"{name}.frequency_penalty",
                kwargs["frequency_penalty"] if "frequency_penalty" in kwargs else 0.0,
            )
            span.set_attribute(
                f"{name}.best_of", kwargs["best_of"] if "best_of" in kwargs else 1
            )
            span.set_attribute(
                f"{name}.logit_bias",
                kwargs["logit_bias"] if "logit_bias" in kwargs else "",
            )
            span.set_attribute(
                f"{name}.user", kwargs["user"] if "user" in kwargs else ""
            )

            response = wrapped(*args, **kwargs)

            span.set_attribute(f"{name}.response.id", response["id"])
            span.set_attribute(f"{name}.response.object", response["object"])
            span.set_attribute(f"{name}.response.created", response["created"])
            for index, choice in enumerate(response["choices"]):
                span.set_attribute(
                    f"{name}.response.choices.{index}.text",
                    choice["text"],
                )
                # TODO: logprobs is optional, not a concept in otel?
                if choice["logprobs"] is not None:
                    span.set_attribute(
                        f"{name}.response.choices.{index}.logprobs",
                        choice["logprobs"],
                    )
                span.set_attribute(
                    f"{name}.response.choices.{index}.finish_reason",
                    choice["finish_reason"],
                )

            span.set_attribute(
                f"{name}.response.usage.prompt_tokens",
                response["usage"]["prompt_tokens"],
            )
            span.set_attribute(
                f"{name}.response.usage.completion_tokens",
                response["usage"]["completion_tokens"],
            )
            span.set_attribute(
                f"{name}.response.usage.total_tokens", response["usage"]["total_tokens"]
            )

        return response

    wrapt.wrap_function_wrapper(openai.Completion, "create", _instrumented_create)


def _instrument_edit(tracer: Tracer):
    def _instrumented_create(wrapped, instance, args, kwargs):
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return

        name = "openai.edit"
        with tracer.start_as_current_span(name, kind=SpanKind.CLIENT) as span:
            span.set_attribute(f"{name}.model", kwargs["model"])
            span.set_attribute(f"{name}.instruction", kwargs["instruction"])
            span.set_attribute(
                f"{name}.input", kwargs["input"] if "input" in kwargs else ""
            )
            span.set_attribute(f"{name}.n", kwargs["n"] if "n" in kwargs else 1)
            span.set_attribute(
                f"{name}.temperature",
                kwargs["temperature"] if "temperature" in kwargs else 1.0,
            )
            span.set_attribute(
                f"{name}.top_p", kwargs["top_p"] if "top_p" in kwargs else 1.0
            )

            response = wrapped(*args, **kwargs)

            span.set_attribute(f"{name}.response.object", response["object"])
            span.set_attribute(f"{name}.response.created", response["created"])
            for index, choice in enumerate(response["choices"]):
                span.set_attribute(
                    f"{name}.response.choices.{index}.text", choice["text"]
                )

            span.set_attribute(
                f"{name}.response.usage.prompt_tokens",
                response["usage"]["prompt_tokens"],
            )
            span.set_attribute(
                f"{name}.response.usage.completion_tokens",
                response["usage"]["completion_tokens"],
            )
            span.set_attribute(
                f"{name}.response.usage.total_tokens", response["usage"]["total_tokens"]
            )

        return response

    wrapt.wrap_function_wrapper(openai.Edit, "create", _instrumented_create)


def _instrument_moderation(tracer: Tracer):
    def _instrumented_create(wrapped, instance, args, kwargs):
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return

        name = "openai.moderation"
        with tracer.start_as_current_span(name, kind=SpanKind.CLIENT) as span:
            span.set_attribute(
                f"{name}.model",
                kwargs["model"] if "model" in kwargs else "text-moderation-latest",
            )
            span.set_attribute(f"{name}.input", kwargs["input"])

            response = wrapped(*args, **kwargs)

            span.set_attribute(f"{name}.response.id", response["id"])
            span.set_attribute(
                f"{name}.response.results.categories.hate",
                response["results"][0]["categories"]["hate"],
            )
            span.set_attribute(
                f"{name}.response.results.categories.hate/threatening",
                response["results"][0]["categories"]["hate/threatening"],
            )
            span.set_attribute(
                f"{name}.response.results.categories.self-harm",
                response["results"][0]["categories"]["self-harm"],
            )
            span.set_attribute(
                f"{name}.response.results.categories.sexual",
                response["results"][0]["categories"]["sexual"],
            )
            span.set_attribute(
                f"{name}.response.results.categories.sexual/minors",
                response["results"][0]["categories"]["sexual/minors"],
            )
            span.set_attribute(
                f"{name}.response.results.categories.violence",
                response["results"][0]["categories"]["violence"],
            )
            span.set_attribute(
                f"{name}.response.results.categories.violence/graphic",
                response["results"][0]["categories"]["violence/graphic"],
            )
            span.set_attribute(
                f"{name}.response.results.category_scores.hate",
                response["results"][0]["category_scores"]["hate"],
            )
            span.set_attribute(
                f"{name}.response.results.category_scores.hate/threatening",
                response["results"][0]["category_scores"]["hate/threatening"],
            )
            span.set_attribute(
                f"{name}.response.results.category_scores.self-harm",
                response["results"][0]["category_scores"]["self-harm"],
            )
            span.set_attribute(
                f"{name}.response.results.category_scores.sexual",
                response["results"][0]["category_scores"]["sexual"],
            )
            span.set_attribute(
                f"{name}.response.results.category_scores.sexual/minors",
                response["results"][0]["category_scores"]["sexual/minors"],
            )
            span.set_attribute(
                f"{name}.response.results.category_scores.violence",
                response["results"][0]["category_scores"]["violence"],
            )
            span.set_attribute(
                f"{name}.response.results.category_scores.violence/graphic",
                response["results"][0]["category_scores"]["violence/graphic"],
            )
            span.set_attribute(
                f"{name}.response.results.flagged", response["results"][0]["flagged"]
            )

        return response

    wrapt.wrap_function_wrapper(openai.Moderation, "create", _instrumented_create)


def _instrument_image_generate(tracer: Tracer):
    def _instrumented_create(wrapped, instance, args, kwargs):
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return

        name = "openai.image.generate"
        with tracer.start_as_current_span(name, kind=SpanKind.CLIENT) as span:
            span.set_attribute(f"{name}.prompt", kwargs["prompt"])
            span.set_attribute(f"{name}.n", kwargs["n"] if "n" in kwargs else 1)
            span.set_attribute(
                f"{name}.size", kwargs["size"] if "size" in kwargs else "1024x1024"
            )
            span.set_attribute(
                f"{name}.response_format",
                kwargs["response_format"] if "response_format" in kwargs else "url",
            )
            span.set_attribute(
                f"{name}.user", kwargs["user"] if "user" in kwargs else ""
            )

            response = wrapped(*args, **kwargs)

            span.set_attribute(f"{name}.response.created", response["created"])
            for index, choice in enumerate(response["data"]):
                if (
                    "response_format" not in kwargs
                    or kwargs["response_format"] == "url"
                ):
                    span.set_attribute(
                        f"{name}.response.data.{index}.url", choice["url"]
                    )
                # Not going to instrument the b64_json response because it's huge

        return response

    wrapt.wrap_function_wrapper(openai.Image, "create", _instrumented_create)


def _instrument_image_edit(tracer: Tracer):
    def _instrumented_create(wrapped, instance, args, kwargs):
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return

        name = "openai.image.edit"
        with tracer.start_as_current_span(name, kind=SpanKind.CLIENT) as span:
            span.set_attribute(f"{name}.prompt", kwargs["prompt"])
            span.set_attribute(f"{name}.image", kwargs["image"])
            span.set_attribute(
                f"{name}.mask", kwargs["mask"] if "mask" in kwargs else ""
            )
            span.set_attribute(f"{name}.n", kwargs["n"] if "n" in kwargs else 1)
            span.set_attribute(
                f"{name}.size", kwargs["size"] if "size" in kwargs else "1024x1024"
            )
            span.set_attribute(
                f"{name}.response_format",
                kwargs["response_format"] if "response_format" in kwargs else "url",
            )
            span.set_attribute(
                f"{name}.user", kwargs["user"] if "user" in kwargs else ""
            )

            response = wrapped(*args, **kwargs)

            span.set_attribute(f"{name}.response.created", response["created"])
            for index, choice in enumerate(response["data"]):
                if (
                    "response_format" not in kwargs
                    or kwargs["response_format"] == "url"
                ):
                    span.set_attribute(
                        f"{name}.response.data.{index}.url", choice["url"]
                    )
                # Not going to instrument the b64_json response because it's huge

        return response

    wrapt.wrap_function_wrapper(openai.Image, "create_edit", _instrumented_create)


def _instrument_image_variation(tracer: Tracer):
    def _instrumented_create(wrapped, instance, args, kwargs):
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return

        name = "openai.image.variation"
        with tracer.start_as_current_span(name, kind=SpanKind.CLIENT) as span:
            span.set_attribute(f"{name}.image", kwargs["image"])
            span.set_attribute(f"{name}.n", kwargs["n"] if "n" in kwargs else 1)
            span.set_attribute(
                f"{name}.size", kwargs["size"] if "size" in kwargs else "1024x1024"
            )
            span.set_attribute(
                f"{name}.response_format",
                kwargs["response_format"] if "response_format" in kwargs else "url",
            )
            span.set_attribute(
                f"{name}.user", kwargs["user"] if "user" in kwargs else ""
            )

            response = wrapped(*args, **kwargs)

            span.set_attribute(f"{name}.response.created", response["created"])
            for index, choice in enumerate(response["data"]):
                if (
                    "response_format" not in kwargs
                    or kwargs["response_format"] == "url"
                ):
                    span.set_attribute(
                        f"{name}.response.data.{index}.url", choice["url"]
                    )
                # Not going to instrument the b64_json response because it's huge

        return response

    wrapt.wrap_function_wrapper(openai.Image, "create_variation", _instrumented_create)


def _instrument_audio_transcription(tracer: Tracer):
    def _instrumented_create(wrapped, instance, args, kwargs):
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return

        name = "openai.audio.transcribe"
        with tracer.start_as_current_span(name, kind=SpanKind.CLIENT) as span:
            span.set_attribute(f"{name}.file", kwargs["file"])
            span.set_attribute(f"{name}.model", kwargs["model"])
            span.set_attribute(
                f"{name}.prompt", kwargs["prompt"] if "prompt" in kwargs else ""
            )
            span.set_attribute(
                f"{name}.response_format",
                kwargs["response_format"] if "response_format" in kwargs else "json",
            )
            span.set_attribute(
                f"{name}.temperature",
                kwargs["temperature"] if "temperature" in kwargs else 0.0,
            )
            span.set_attribute(
                f"{name}.language", kwargs["language"] if "language" in kwargs else ""
            )

            response = wrapped(*args, **kwargs)

            span.set_attribute(f"{name}.response.text", response["text"])

        return response

    wrapt.wrap_function_wrapper(openai.Audio, "transcribe", _instrumented_create)


def _instrument_audio_translate(tracer: Tracer):
    def _instrumented_create(wrapped, instance, args, kwargs):
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return

        name = "openai.audio.translate"
        with tracer.start_as_current_span(name, kind=SpanKind.CLIENT) as span:
            span.set_attribute(f"{name}.file", kwargs["file"])
            span.set_attribute(f"{name}.model", kwargs["model"])
            span.set_attribute(
                f"{name}.prompt", kwargs["prompt"] if "prompt" in kwargs else ""
            )
            span.set_attribute(
                f"{name}.response_format",
                kwargs["response_format"] if "response_format" in kwargs else "json",
            )
            span.set_attribute(
                f"{name}.temperature",
                kwargs["temperature"] if "temperature" in kwargs else 0.0,
            )

            response = wrapped(*args, **kwargs)

            span.set_attribute(f"{name}.response.text", response["text"])

        return response

    wrapt.wrap_function_wrapper(openai.Audio, "translate", _instrumented_create)


class OpenAIInstrumentor(BaseInstrumentor):
    """An instrumenter for OpenAI's client library."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)
        _instrument_chat(tracer)
        _instrument_embedding(tracer)
        _instrument_completions(tracer)
        _instrument_edit(tracer)
        _instrument_moderation(tracer)
        _instrument_image_generate(tracer)
        _instrument_image_edit(tracer)
        _instrument_image_variation(tracer)
        _instrument_audio_transcription(tracer)
        _instrument_audio_translate(tracer)

    def _uninstrument(self, **kwargs):
        unwrap(openai.ChatCompletion, "create")
        unwrap(openai.Embedding, "create")
        unwrap(openai.Completion, "create")
        unwrap(openai.Edit, "create")
        unwrap(openai.Moderation, "create")
        unwrap(openai.Image, "create")
        unwrap(openai.Image, "create_edit")
        unwrap(openai.Image, "create_variation")
        unwrap(openai.Audio, "transcribe")
        unwrap(openai.Audio, "translate")
