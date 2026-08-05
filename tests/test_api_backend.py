from probe.backends.api_backend import _completion_text


def test_normal_completion():
    assert _completion_text({"message": {"content": "define i32 @f()"}}) == "define i32 @f()"


def test_reasoning_content_is_not_the_answer():
    # gpt-oss/Qwen3 split chain-of-thought into `reasoning_content`. It is never the rewrite,
    # so it must not leak into the IR we hand to the verifier.
    choice = {
        "message": {"reasoning_content": "The user wants...", "content": "define i32 @f()"},
    }
    assert _completion_text(choice) == "define i32 @f()"


def test_truncated_reasoning_yields_empty_not_a_crash():
    # The shape that broke the run: reasoning ate the token budget, so `content` is absent.
    # Previously this raised KeyError inside the retry loop and burned two pointless retries.
    choice = {"message": {"role": "assistant", "reasoning_content": "hi"}, "finish_reason": "length"}
    assert _completion_text(choice) == ""


def test_null_content_is_treated_as_empty():
    assert _completion_text({"message": {"content": None}}) == ""
