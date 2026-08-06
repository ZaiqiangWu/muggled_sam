"""Serialization helpers for video prompt/tracking state files.

The first version of the video scripts saved the object-index mapping directly.
Newer files wrap that mapping so optional metadata, such as a SAM3 text prompt,
can be persisted without changing the stored tracking objects.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_FORMAT_VERSION = 2
_OBJECTS_KEY = "objects"
_TEXT_PROMPT_KEY = "text_prompt"
_TEXT_PROMPTS_KEY = "text_prompts"
_VERSION_KEY = "format_version"


def make_tracking_state(
    objects: Mapping[int, Any], text_prompts: Mapping[int, str] | str | None = None
) -> dict[str, Any]:
    """Build the on-disk representation for video tracking prompts.

    ``objects`` is copied into a plain dictionary so callers can safely keep
    mutating their in-memory object-buffer mapping after saving.
    """
    if not isinstance(objects, Mapping):
        raise TypeError("objects must be a mapping from buffer index to tracking state")
    # Accept the original single-prompt argument for compatibility with files
    # created while text prompting was global rather than per object.
    if isinstance(text_prompts, str):
        text_prompts = {0: text_prompts}
    if text_prompts is not None and not isinstance(text_prompts, Mapping):
        raise TypeError("text_prompts must be a mapping from buffer index to string")

    normalized_prompts = {} if text_prompts is None else dict(text_prompts)
    if not all(isinstance(index, int) and isinstance(prompt, str) for index, prompt in normalized_prompts.items()):
        raise TypeError("text_prompts must map integer buffer indices to strings")

    return {
        _VERSION_KEY: _FORMAT_VERSION,
        _OBJECTS_KEY: dict(objects),
        _TEXT_PROMPTS_KEY: normalized_prompts,
    }


def unpack_tracking_state(saved_state: Any) -> tuple[dict[Any, Any], dict[int, str]]:
    """Return ``(objects, text_prompts)`` from new or legacy prompt files.

    Legacy files contain only the object-index mapping, so they naturally load
    with no text metadata.
    """
    if not isinstance(saved_state, Mapping):
        raise ValueError("Prompt file must contain a mapping of tracking states")

    if _OBJECTS_KEY not in saved_state:
        # Files written before text prompts were supported.
        return dict(saved_state), {}

    objects = saved_state[_OBJECTS_KEY]
    if not isinstance(objects, Mapping):
        raise ValueError("Prompt file has an invalid 'objects' mapping")

    text_prompts = saved_state.get(_TEXT_PROMPTS_KEY)
    if text_prompts is None:
        # Version 1 files stored one global text prompt.  Its tracked object was
        # seeded into buffer 0, so preserve that association on load.
        legacy_text_prompt = saved_state.get(_TEXT_PROMPT_KEY)
        if legacy_text_prompt is None:
            text_prompts = {}
        elif isinstance(legacy_text_prompt, str):
            text_prompts = {0: legacy_text_prompt}
        else:
            raise ValueError("Prompt file has an invalid 'text_prompt' value")

    if not isinstance(text_prompts, Mapping) or not all(
        isinstance(index, int) and isinstance(prompt, str) for index, prompt in text_prompts.items()
    ):
        raise ValueError("Prompt file has an invalid 'text_prompts' mapping")

    return dict(objects), dict(text_prompts)
