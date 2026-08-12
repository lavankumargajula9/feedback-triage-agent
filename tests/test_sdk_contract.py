"""Offline contract test binding the mock seam to the real anthropic SDK.

The whole suite mocks `client.messages.parse`, so a drift between the mock's
shape and the SDK's would let every other test pass against a fiction. These
tests read the installed SDK's own signature and model — no network, no key.
"""

import inspect

import anthropic
from anthropic.types.parsed_message import ParsedMessage, ParsedTextBlock

from triage.tools.llm import DEFAULT_MAX_TOKENS, call_with_schema
from triage.tools.schemas import CategorizeResult

PARSED = CategorizeResult(label="General Inquiry", rationale="ok")


def real_parsed_message(parsed):
    """A genuine SDK ParsedMessage whose `parsed_output` property yields `parsed`.

    Built from content blocks because parsed_output is computed over them, not
    a settable field — the exact shape a stub would otherwise get wrong.
    """
    return ParsedMessage.construct(
        content=[
            ParsedTextBlock.construct(
                type="text", text=parsed.model_dump_json(), parsed_output=parsed
            )
        ],
        stop_reason="end_turn",
        usage=None,
    )


class RecordingMessages:
    def __init__(self):
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return real_parsed_message(PARSED)


class RecordingClient:
    def __init__(self):
        self.messages = RecordingMessages()


def real_parse_parameters() -> set[str]:
    client = anthropic.Anthropic(api_key="not-a-real-key")
    return set(inspect.signature(client.messages.parse).parameters)


class TestCallContract:
    def test_every_kwarg_we_send_is_accepted_by_the_real_sdk(self):
        client = RecordingClient()
        call_with_schema(
            "categorize",
            model="claude-haiku-4-5",
            system="sys",
            user_text="hello",
            schema=CategorizeResult,
            client=client,
        )
        (sent,) = client.messages.calls
        unknown = set(sent) - real_parse_parameters()
        assert not unknown, f"kwargs the installed SDK would reject: {sorted(unknown)}"

    def test_required_call_shape(self):
        client = RecordingClient()
        call_with_schema(
            "categorize",
            model="claude-haiku-4-5",
            system="sys",
            user_text="hello",
            schema=CategorizeResult,
            client=client,
        )
        (sent,) = client.messages.calls
        assert sent["model"] == "claude-haiku-4-5"
        assert sent["output_format"] is CategorizeResult
        assert sent["max_tokens"] == DEFAULT_MAX_TOKENS
        assert sent["messages"] == [{"role": "user", "content": "hello"}]


class TestResponseContract:
    def test_sdk_response_exposes_the_attributes_the_wrapper_reads(self):
        assert hasattr(ParsedMessage, "parsed_output")
        assert "stop_reason" in ParsedMessage.model_fields
        assert "usage" in ParsedMessage.model_fields

    def test_wrapper_reads_a_real_parsed_message_object(self):
        # The response the wrapper unwraps is the SDK's own model, not a stub.
        result = call_with_schema(
            "categorize",
            model="claude-haiku-4-5",
            system="sys",
            user_text="hello",
            schema=CategorizeResult,
            client=RecordingClient(),
        )
        assert isinstance(result, CategorizeResult)
        assert result.label == "General Inquiry"
