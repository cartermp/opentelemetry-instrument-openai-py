# Copyright your mom
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
    from opentelemetry.instrumentation.openai import OpenAIInstrumentation

    # Enable instrumentation
    OpenAIInstrumentation().instrument()

    openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "tell me a joke about opentelemetry"}],
    )
"""

from typing import Collection

import wrapt
import openai
from openai.api_resources.abstract.engine_api_resource import EngineAPIResource

from opentelemetry import context as context_api
from opentelemetry.trace import get_tracer, Tracer

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

        with tracer.start_as_current_span("openai.chat"):
            result = wrapped(*args, **kwargs)

        return result

    wrapt.wrap_function_wrapper(openai.ChatCompletion, "create", _instrumented_create)


def _uninstrument():
    unwrap(openai.ChatCompletion, "create")


class OpenAIInstrumentator(BaseInstrumentor):
    """An instrumenter for OpenAI's client library."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        """Instruments all OpenAI client calls.

        Args:
            **kwargs: Optional arguments
                ``tracer_provider``: a TracerProvider, defaults to global
        """
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)
        _instrument_chat(tracer)

    def _uninstrument(self, **kwargs):
        _uninstrument()
