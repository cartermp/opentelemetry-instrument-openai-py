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
import logging
from typing import Collection

from wrapt import wrap_function_wrapper
import openai

from opentelemetry import context as context_api
from opentelemetry.trace import get_tracer, SpanKind
from opentelemetry.trace.status import Status, StatusCode

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)
from opentelemetry.instrumentation.openai.package import _instruments
from opentelemetry.instrumentation.openai.version import __version__


logger = logging.getLogger(__name__)


TO_WRAP = [
    {
        "object": "ChatCompletion",
        "method": "create",
        "span_name": "openai.chat",
        "default_inputs": {
            "temperature": 1.0,
            "top_p": 1.0,
            "n": 1,
            "stream": False,
            "stop": "",
            "max_tokens": float('inf'),
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "logit_bias": "",
            "user": "",
        }
    },
    {
        "object": "Completion",
        "method": "create",
        "span_name": "openai.completion",
        "default_inputs": {
            "suffix": "",
            "temperature": 1.0,
            "top_p": 1.0,
            "n": 1,
            "stream": False,
            "logprobs": -1,
            "echo": False,
            "stop": "",
            "max_tokens": float('inf'),
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "best_of": 1,
            "logit_bias": "",
            "user": "",
        }
    },
    {
        "object": "Embedding",
        "method": "create",
        "span_name": "openai.embedding",
        "default_inputs": {
            "user": "",
        }
    },
    {
        "object": "Edit",
        "method": "create",
        "span_name": "openai.edit",
        "default_inputs": {
            "input": "",
            "temperature": 1.0,
            "top_p": 1.0,
            "n": 1,
        }
    },
    {
        "object": "Moderation",
        "method": "create",
        "span_name": "openai.moderation",
        "default_inputs": {
            "model": "text-moderation-latest"
        }
    },
    {
        "object": "Image",
        "method": "create",
        "span_name": "openai.image.generate",
        "default_inputs": {
            "n": 1,
            "size": "1024x1024",
            "response_format": "url",
            "user": "",
        }
    },
    {
        "object": "Image",
        "method": "create_edit",
        "span_name": "openai.image.edit",
        "default_inputs": {
            "mask": "",
            "n": 1,
            "size": "1024x1024",
            "response_format": "url",
            "user": "",
        }
    },
    {
        "object": "Image",
        "method": "create_variation",
        "span_name": "openai.image.variation",
        "default_inputs": {
            "n": 1,
            "size": "1024x1024",
            "response_format": "url",
            "user": "",
        }
    },
    {
        "object": "Audio",
        "method": "transcribe",
        "span_name": "openai.audio.transcribe",
        "default_inputs": {
            "prompt": "",
            "response_format": "json",
            "temperature": 0.0,
            "language": "",
        }
    },
    {
        "object": "Audio",
        "method": "translate",
        "span_name": "openai.audio.translate",
        "default_inputs": {
            "prompt": "",
            "response_format": "json",
            "temperature": 0.0,
        }
    }
]


def no_none(value):
    """
    OTEL Attributes cannot be NoneType. 
    If NoneType return string 'None'.
    """
    if value is None:
        return str(value)
    return value


def _set_attributes(span, name, attributes, nestings=[]):
    """
    For every nested field, set its fields as attributes.
    Then set the remianing non-nested fields. 
    """
    attr_copy = attributes.copy()  # don't change shape of inputs!
    for nesting in nestings:
        nested = attr_copy.pop(nesting, None)
        if nested:
            for key, value in nested.items():
                span.set_attribute(
                    f"{name}.{nesting}.{key}",
                    no_none(value)
                )
    for key, value in attr_copy.items():
        span.set_attribute(
            f"{name}.{key}",
            no_none(value)
        )
    return


def _set_attributes_from_array(span, name, attributes, array_field, nestings=[]):
    """
    For a given field that contains arrays of fields, set each index's
    fields as a unique set of attributes.
    """
    attr_array = attributes.pop(array_field, None)
    if attr_array:
        for index, attr_item in enumerate(attr_array):
            _set_attributes(
                span=span,
                name=f"{name}.{array_field}.{index}",
                attributes=attr_item,
                nestings=nestings
            )
    return


def _set_input_attributes(span, name, to_wrap, kwargs):
    """
    Capture input params as span attributes. 
    Unpacks 'message' input fields to separate attributes.
    For embeddings, captures count of 'input' values.
    """

    params = to_wrap.get("default_inputs")
    params.update(kwargs)

    # "input" fields can be very large, so we handle them specially 
    # depending on which api object they belong to. 
    _input = params.pop("input", None)
    if _input:
        if name in ["openai.embedding"]:
            # input values for Embedding objects can be too
            # long so for that we only capture len(input)
            span.set_attribute(
                f"{name}.input_count",
                len(_input)
            )
        else:
            # but input values for other objects are interesting
            span.set_attribute(
                f"{name}.input",
                no_none(_input)
            )

    # set attributes from input fields nested under arrays 
    # {messages[]}
    _set_attributes_from_array(
        span=span,
        name=name,
        attributes=params,
        array_field="messages"
    )

    # set attributes from remaining input fields
    _set_attributes(
        span=span,
        name=name,
        attributes=params
    )
    return


def _set_response_attributes(span, name, response):
    """
    Captures response fields as span attributes. 
    Unpacks 'choices', 'usage', and 'results' fields to separate attributes.
    For embeddings, catpures count of data values instead of data.
    Otherwise captures data values that are not b64_json.
    """

    resp_attributes = response.copy()

    # "data" fields can be very large, so we handle them specially 
    # depending on which api object they belong to. 
    resp_data = resp_attributes.pop("data", None)
    if resp_data:
        if name in ["openai.embedding"]:
            # data values for Embedding objects can be too
            # long so for that we only capture len(data)
            span.set_attribute(
                f"{name}.response.embeddings_count",
                len(resp_data)
            )
        else:
            # but data values for other objects are interesting
            for index, datum in enumerate(resp_data):
                for key, value in datum.items():
                    if key not in ["b64_json"]:
                        # b64_json values are too big, skip them
                        span.set_attribute(
                            f"{name}.response.data.{index}.{key}",
                            no_none(value)
                        )

    # set attributes from response fields nested under arrays 
    # {choices[], results[]}
    _set_attributes_from_array(
        span=span,
        name=f"{name}.response",
        attributes=resp_attributes,
        array_field="choices",
        nestings=["message"]
    )
    _set_attributes_from_array(
        span=span,
        name=f"{name}.response",
        attributes=resp_attributes,
        array_field="results",
        nestings=["categories", "category_scores"]
    )

    # set attributes from remaining response fields
    _set_attributes(
        span=span,
        name=f"{name}.response",
        attributes=resp_attributes,
        nestings=["usage"]
    )
    return


def _set_api_attributes(span):
    """Capture attributes about the api endpoint."""
    span.set_attribute("openai.api_base", no_none(openai.api_base))
    span.set_attribute("openai.api_type", no_none(openai.api_type))
    span.set_attribute("openai.api_version", no_none(openai.api_version))
    return


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            # prevent double wrapping
            if hasattr(wrapped, "__wrapped__"):
                return wrapped(*args, **kwargs)

            return func(tracer, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


@_with_tracer_wrapper
def _wrap(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")
    with tracer.start_as_current_span(
        name, kind=SpanKind.CLIENT, attributes={}
    ) as span:
        if span.is_recording():
            _set_api_attributes(span)
        try:
            if span.is_recording():
                _set_input_attributes(span, name, to_wrap, kwargs)

        except Exception as ex:  # pylint: disable=broad-except
            logger.warning(
                "Failed to set input attributes for openai span, error: %s", 
                str(ex)
            )

        response = wrapped(*args, **kwargs)

        if response:
            try:
                if span.is_recording():
                    _set_response_attributes(span, name, response)

            except Exception as ex:  # pylint: disable=broad-except
                logger.warning(
                    "Failed to set response attributes for openai span, error: %s", 
                    str(ex)
                )
            if span.is_recording():
                span.set_status(Status(StatusCode.OK))
            
        return response


class OpenAIInstrumentor(BaseInstrumentor):
    """An instrumenter for OpenAI's client library."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)
        for to_wrap in TO_WRAP:
            wrap_object = to_wrap.get("object")
            wrap_method = to_wrap.get("method")
            wrap_function_wrapper(
                "openai",
                f"{wrap_object}.{wrap_method}",
                _wrap(tracer, to_wrap)
            )

    def _uninstrument(self, **kwargs):
        for to_wrap in TO_WRAP:
            wrap_object = to_wrap.get("object")
            unwrap(
                f"openai.{wrap_object}",
                to_wrap.get("method")
            )
