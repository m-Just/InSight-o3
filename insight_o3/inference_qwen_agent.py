"""Qwen Agent inference backend for evaluate.py.

Provides query_qwen_agent_vqa() with the same return signature as query_api_vqa(),
so the two can be swapped in the evaluation loop.
"""

import asyncio
import os
import tempfile
from dataclasses import dataclass
from time import perf_counter

from PIL import Image

from insight_o3.inference import (
    Conversation,
    InferenceResult,
    maybe_rescale_image,
    maybe_resize_image,
    summarize_image_prep_profiles,
)
from insight_o3.utils.format import _format_reasoning


def format_qwen_agent_messages(
    messages: list[dict],
    omit_image_data: bool = True,
    max_reasoning_chars: int = 1000,
) -> str:
    """Format Qwen Agent messages for display.

    Handles Qwen Agent content conventions (``{"image": path}``,
    ``{"text": ...}``, ``function_call``, role ``"function"``) in addition
    to the standard OpenAI-style content parts.  Reasoning content, if
    present, is shown before the main content (middle-truncated when long).
    """
    s = ""
    for message in messages:
        role = message.get("role", "unknown").upper()
        name = message.get("name", "")
        if name:
            s += f"[{role}: {name}]\n"
        else:
            s += f"[{role}]\n"

        s += _format_reasoning(message, max_reasoning_chars)

        content = message.get("content")
        if content is None:
            fn_call = message.get("function_call")
            if fn_call:
                s += f"<function_call: {fn_call.get('name', '?')}({fn_call.get('arguments', '')})>\n"
            else:
                s += "<content missing>\n"
            continue

        if isinstance(content, str):
            s += content
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if "text" in part and "type" not in part:
                        s += part["text"]
                    elif "image" in part and "type" not in part:
                        if omit_image_data:
                            s += "<image>"
                        else:
                            s += f"[image]({part['image']})"
                    elif part.get("type") == "text":
                        s += part.get("text", "")
                    elif "image_url" in part:
                        if omit_image_data:
                            s += "<image>"
                        else:
                            url = part["image_url"]
                            if isinstance(url, dict):
                                url = url.get("url", "")
                            s += f"[image_url]({url})"
                    else:
                        s += f"<{part.get('type', 'unknown')}>"
                elif isinstance(part, str):
                    s += part
                else:
                    s += f"<unexpected part type: {type(part)}>"
        else:
            s += f"<unexpected content type: {type(content)}>"

        s += "\n"

    return s


@dataclass
class QwenAgentConversation(Conversation):
    """Conversation subclass that formats Qwen Agent messages properly."""

    def __str__(self):
        s = ""
        s += f">>>>>>>>>>>>>>>>>>> {self.agent} messages start >>>>>>>>>>>>>>>>>>>\n"
        s += format_qwen_agent_messages(self.messages)
        s += f"<<<<<<<<<<<<<<<<<<< {self.agent} messages stop <<<<<<<<<<<<<<<<<<<<\n"
        return s


def prepare_image_as_file(
    image_path: str,
    min_pixels: int | None,
    max_pixels: int | None,
    rescale_ratio: float | None = None,
    image_format: str = "png",
    tmpdir: str | None = None,
) -> tuple[str, dict[str, float]]:
    """Preprocess an image and save it to a temporary file.

    Applies the same rescale / resize pipeline as ``prepare_image`` but
    writes to disk instead of base64-encoding (Qwen Agent expects file paths).

    Returns (temp_file_path, profile_dict).
    """
    load_start = perf_counter()
    image = Image.open(image_path).convert("RGB")
    load_seconds = perf_counter() - load_start

    resize_start = perf_counter()
    image = maybe_rescale_image(image, rescale_ratio)
    image = maybe_resize_image(image, min_pixels, max_pixels)
    resize_seconds = perf_counter() - resize_start

    save_start = perf_counter()
    suffix = {"png": ".png", "jpg": ".jpg"}[image_format]
    pil_format = {"png": "PNG", "jpg": "JPEG"}[image_format]
    fd = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=tmpdir)
    try:
        image.save(fd, format=pil_format)
        temp_path = fd.name
    finally:
        fd.close()
    save_seconds = perf_counter() - save_start

    profile = {
        "load_seconds": load_seconds,
        "resize_seconds": resize_seconds,
        "encode_seconds": save_seconds,
        "total_seconds": load_seconds + resize_seconds + save_seconds,
    }
    return temp_path, profile


def _run_qwen_agent_sync(
    model: str,
    api_base_url: str,
    api_key: str,
    model_type: str,
    system_prompt: str | None,
    tools: list[str],
    generate_cfg: dict,
    image_file_paths: list[str],
    user_text: str,
    max_retries: int = 3,
) -> tuple[str | None, list[dict] | None]:
    """Run Qwen Agent synchronously (called via ``asyncio.to_thread``).

    Returns ``(last_message_content, ret_messages)`` on success.
    Raises the last exception after *max_retries* consecutive failures.
    """
    from qwen_agent.agents import Assistant  # lazy: only needed for this backend

    llm_cfg = {
        "model_type": model_type,
        "model": model,
        "model_server": api_base_url,
        "api_key": api_key,
        "generate_cfg": generate_cfg,
    }

    agent = Assistant(
        llm=llm_cfg,
        function_list=tools,
        system_message=system_prompt or "",
    )

    content_parts: list[dict] = [{"image": p} for p in image_file_paths]
    content_parts.append({"text": user_text})
    messages = [{"role": "user", "content": content_parts}]

    last_error: Exception | None = None
    for attempt in range(max_retries):
        ret_messages = None
        try:
            for ret_messages in agent.run(messages):
                pass
            if ret_messages is None:
                raise RuntimeError("Qwen Agent returned no messages")
            last_message = ret_messages[-1]
            if last_message["role"] != "assistant":
                raise RuntimeError(
                    f"Last message role is '{last_message['role']}', expected 'assistant'"
                )
            return last_message["content"], [*messages, *ret_messages]
        except Exception as exc:
            last_error = exc
            print(
                f"WARNING: Qwen Agent attempt {attempt + 1}/{max_retries} failed: {exc}"
            )
    raise last_error  # type: ignore[misc]


def _normalize_agent_content(content: object) -> str | None:
    """Extract plain text from a Qwen Agent assistant message content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif "text" in item:
                    text_parts.append(item["text"])
            elif isinstance(item, str):
                text_parts.append(item)
        return "".join(text_parts) if text_parts else None
    return str(content) if content else None


def _empty_token_usage() -> dict[str, int]:
    return {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "api_calls_with_usage": 0,
    }


async def query_qwen_agent_vqa(
    user_prompt: str,
    model: str,
    api_base_url: str,
    api_key: str,
    image_paths: list[str] | None = None,
    image_path: str | None = None,
    image_rescale_ratio: float | None = None,
    image_format: str = "png",
    image_max_pixels: int | None = None,
    system_prompt: str | None = None,
    model_type: str = "qwenvl_oai",
    tools: list[str] | None = None,
    generate_cfg: dict | None = None,
    max_retries: int = 3,
) -> tuple[list[InferenceResult], dict[str, int]]:
    """Query a Qwen Agent for VQA.

    Return signature matches ``query_api_vqa`` so the two can be swapped in
    the evaluation loop.  Always returns a single-element result list (the
    agent does not support ``n > 1``).
    """
    # ---- validate image paths ------------------------------------------------
    if image_paths is None:
        if image_path is None:
            raise ValueError("Either image_paths or image_path must be provided")
        image_paths = [image_path]
    elif image_path is not None:
        raise ValueError("Provide either image_paths or image_path, not both")
    if not image_paths:
        raise ValueError("image_paths must not be empty")

    if tools is None:
        tools = ["image_zoom_in_tool"]
    if generate_cfg is None:
        generate_cfg = {}

    # ---- preprocess images → temp files --------------------------------------
    temp_file_paths: list[str] = []
    image_prep_profiles: list[dict[str, float]] = []
    tmpdir = os.environ.get("TMPDIR")

    try:
        for current_image_path in image_paths:
            temp_path, profile = prepare_image_as_file(
                current_image_path,
                None,
                image_max_pixels,
                image_rescale_ratio,
                image_format,
                tmpdir=tmpdir,
            )
            temp_file_paths.append(temp_path)
            image_prep_profiles.append(profile)
        image_prep_profile = summarize_image_prep_profiles(image_prep_profiles)

        # ---- run agent -------------------------------------------------------
        request_start = perf_counter()
        try:
            last_content, ret_messages = await asyncio.to_thread(
                _run_qwen_agent_sync,
                model=model,
                api_base_url=api_base_url,
                api_key=api_key,
                model_type=model_type,
                system_prompt=system_prompt,
                tools=tools,
                generate_cfg=generate_cfg,
                image_file_paths=temp_file_paths,
                user_text=user_prompt,
                max_retries=max_retries,
            )
        except Exception as exc:
            request_wait_seconds = perf_counter() - request_start
            print(f"WARNING: Qwen Agent failed after retries: {exc}")
            profile = {
                "image_preprocess": image_prep_profile,
                "request_wait_seconds": request_wait_seconds,
                "response_parse_seconds": 0.0,
                "total_seconds": image_prep_profile["total_seconds"] + request_wait_seconds,
            }
            return [
                InferenceResult(
                    success=False,
                    fail_reason="qwen_agent_error",
                    fail_detail=str(exc),
                    profile=profile,
                )
            ], _empty_token_usage()

        request_wait_seconds = perf_counter() - request_start

        # ---- parse result ----------------------------------------------------
        parse_start = perf_counter()
        try:
            last_message_content = _normalize_agent_content(last_content)
            if not last_message_content:
                raise ValueError(
                    f"Empty agent response content (raw={last_content!r})"
                )
        except Exception as exc:
            parse_seconds = perf_counter() - parse_start
            profile = {
                "image_preprocess": image_prep_profile,
                "request_wait_seconds": request_wait_seconds,
                "response_parse_seconds": parse_seconds,
                "total_seconds": image_prep_profile["total_seconds"] + request_wait_seconds + parse_seconds,
            }
            return [
                InferenceResult(
                    success=False,
                    fail_reason="failed_to_parse_agent_response",
                    fail_detail=str(exc),
                    profile=profile,
                )
            ], _empty_token_usage()

        parse_seconds = perf_counter() - parse_start

        conversations: list[Conversation] = []
        if ret_messages:
            conversations = [QwenAgentConversation(agent=model, messages=ret_messages)]

        profile = {
            "image_preprocess": image_prep_profile,
            "request_wait_seconds": request_wait_seconds,
            "response_parse_seconds": parse_seconds,
            "total_seconds": (
                image_prep_profile["total_seconds"]
                + request_wait_seconds
                + parse_seconds
            ),
        }

        result = InferenceResult(
            success=True,
            finish_reason="stop",
            conversations=conversations,
            last_message_content=last_message_content,
            profile=profile,
        )
        return [result], _empty_token_usage()

    finally:
        for temp_path in temp_file_paths:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
