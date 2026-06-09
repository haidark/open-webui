from open_webui.utils.payload import apply_anthropic_cache_control


EPHEMERAL = {"type": "ephemeral"}


def _cache_controls(messages):
    """Collect every cache_control marker found across messages."""
    found = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and "cache_control" in part:
                    found.append((m.get("role"), part["cache_control"]))
    return found


def test_non_anthropic_models_untouched():
    for model in ["openai/gpt-5.5-pro", "x-ai/grok-4.3", "google/gemini-2.5-pro"]:
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ],
        }
        out = apply_anthropic_cache_control(body)
        # String content stays a plain string; no breakpoints injected.
        assert out["messages"][0]["content"] == "sys"
        assert out["messages"][1]["content"] == "hi"
        assert _cache_controls(out["messages"]) == []


def test_string_content_converted_and_marked():
    body = {
        "model": "anthropic/claude-opus-4.8",
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "hello"},
        ],
    }
    out = apply_anthropic_cache_control(body)

    sys_content = out["messages"][0]["content"]
    assert sys_content == [
        {"type": "text", "text": "system prompt", "cache_control": EPHEMERAL}
    ]
    user_content = out["messages"][1]["content"]
    assert user_content == [
        {"type": "text", "text": "hello", "cache_control": EPHEMERAL}
    ]


def test_marks_system_and_last_two_messages_only():
    body = {
        "model": "anthropic/claude-opus-4.8",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "content": "answer 1"},
            {"role": "user", "content": "turn 2"},
        ],
    }
    out = apply_anthropic_cache_control(body)
    marks = _cache_controls(out["messages"])

    # system + last two (assistant "answer 1", user "turn 2") == 3 breakpoints,
    # within Anthropic's 4-breakpoint limit. "turn 1" stays unmarked.
    assert len(marks) == 3
    assert out["messages"][1]["content"] == "turn 1"  # untouched, still a string
    roles = {role for role, _ in marks}
    assert roles == {"system", "assistant", "user"}


def test_list_content_marks_last_part():
    body = {
        "model": "anthropic/claude-opus-4.8",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            }
        ],
    }
    out = apply_anthropic_cache_control(body)
    parts = out["messages"][0]["content"]
    # Breakpoint goes on the last dict part; earlier part is left clean.
    assert "cache_control" not in parts[0]
    assert parts[1]["cache_control"] == EPHEMERAL


def test_idempotent_breakpoint_count():
    body = {
        "model": "anthropic/claude-opus-4.8",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ],
    }
    once = apply_anthropic_cache_control(body)
    n_once = len(_cache_controls(once["messages"]))
    twice = apply_anthropic_cache_control(once)
    n_twice = len(_cache_controls(twice["messages"]))
    # Re-running must not multiply breakpoints (Anthropic caps at 4).
    assert n_once == n_twice == 2


def test_empty_and_missing_messages_safe():
    assert apply_anthropic_cache_control(
        {"model": "anthropic/claude-opus-4.8"}
    ) == {"model": "anthropic/claude-opus-4.8"}
    assert apply_anthropic_cache_control(
        {"model": "anthropic/claude-opus-4.8", "messages": []}
    )["messages"] == []


def test_single_user_message_marked_once():
    body = {
        "model": "anthropic/claude-opus-4.8",
        "messages": [{"role": "user", "content": "only message"}],
    }
    out = apply_anthropic_cache_control(body)
    assert len(_cache_controls(out["messages"])) == 1
