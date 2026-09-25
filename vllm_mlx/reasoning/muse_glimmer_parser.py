# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser for Muse Glimmer models (ATEM channel format).

Muse Glimmer addresses each assistant message to a recipient:
    <|start|>assistant to=self<|message|>[reasoning]<|eom|>
    <|start|>assistant to=user<|message|>[content]<|eot|>
or, for a tool call, to the tool itself:
    <|start|>assistant to=get_weather<|message|><atem:function_calls>...

The generation prompt ends with ``<|start|>assistant``, so output begins with
`` to=self``. ``<|start|>`` and ``<|message|>`` are special tokens: streamed text
still contains them, but the MLLM engine strips them from complete output via
``clean_output_text``. Both forms are normalized by dropping those tokens, which
leaves a plain start/end pair (``to=self`` ... ``<|eom|>``) followed by a
channel header (``assistant to=user``) that is removed from the content. A tool
channel header is removed too, leaving ``<atem:function_calls>`` for the
``muse_glimmer`` tool parser.

The chat template renders one ``to=self`` block per assistant message, so only
the first block is treated as reasoning. A recipient followed by
``<atem:function_calls>`` is a tool channel even when its name starts with
``user`` (``to=user_lookup``); otherwise ``user`` is the user channel and what
follows it is the reply. ``self`` and ``user`` are themselves reserved
recipients per the ATEM protocol and cannot be tool names; the server rejects
those names at request-validation time (``_validate_muse_glimmer_tool_names``
in ``server.py``) rather than relying on this lookahead to resolve every case.
"""

import re

from .base import DeltaMessage
from .think_parser import BaseThinkingReasoningParser

_FRAMING_RE = re.compile(r"<\|start\|>|<\|message\|>")
_TOOL_CALLS = "<atem:function_calls>"
_HEADER = "assistant to="
# Tool recipient first, so ``to=user_lookup<atem:...`` is not read as the user
# channel followed by ``_lookup``.
_HEADER_RE = re.compile(r"assistant to=(?:[\w.\-]+(?=\s*<atem:function_calls>)|user)")
_RECIPIENT_RE = re.compile(r"[\w.\-]*")


def _strip_header(content: str) -> str:
    return _HEADER_RE.sub("", content.lstrip(), count=1)


def _header_incomplete(text: str) -> bool:
    """True while ``text`` could still grow into a channel header."""
    if _HEADER.startswith(text):
        return True
    if not text.startswith(_HEADER):
        return False
    recipient = text[len(_HEADER) :]
    # Until the text after the name rules out <atem:function_calls>, a
    # ``user``-prefixed name may still be a tool (``user_lookup``).
    rest = recipient[_RECIPIENT_RE.match(recipient).end() :].lstrip()
    return _TOOL_CALLS.startswith(rest) and rest != _TOOL_CALLS


class MuseGlimmerReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for Muse Glimmer's ATEM channels.

    Example (engine-cleaned output):
        Input: " to=selfThe user wants a greeting.<|eom|>assistant to=userHello"
        Output: reasoning="The user wants a greeting.", content="Hello"
    """

    @property
    def start_token(self) -> str:
        return "to=self"

    @property
    def end_token(self) -> str:
        return "<|eom|>"

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        text = _FRAMING_RE.sub("", model_output)
        if self.end_token not in text:
            return super().extract_reasoning(text)
        # The reasoning message ends at <|eom|>; one reply or tool message follows.
        reasoning, _, content = text.partition(self.end_token)
        reasoning = reasoning.strip().removeprefix(self.start_token).strip()
        return reasoning or None, _strip_header(content).strip() or None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        return super().extract_reasoning_streaming(
            _FRAMING_RE.sub("", previous_text),
            _FRAMING_RE.sub("", current_text),
            _FRAMING_RE.sub("", delta_text),
        )

    def _content_delta(self, delta_text: str) -> DeltaMessage | None:
        if self._content_started:
            return super()._content_delta(delta_text)
        # Withhold the channel header until it can be removed whole.
        buffer = (self._content_buffer + delta_text).lstrip()
        self._content_buffer = ""
        if _header_incomplete(buffer):
            self._content_buffer = buffer
            return None
        return super()._content_delta(_strip_header(buffer))

    def finalize_stream(self) -> DeltaMessage | None:
        if not self._content_started and self._content_buffer:
            leftover, self._content_buffer = self._content_buffer, ""
            # A held ``to=user<word>`` reply still loses its header here.
            return DeltaMessage(content=_strip_header(leftover) or None)
        return super().finalize_stream()
