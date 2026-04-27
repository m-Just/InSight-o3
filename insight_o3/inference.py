from dataclasses import dataclass, field
from math import ceil, floor
import base64
import io
import os
from time import perf_counter

from PIL import Image

from openai import AsyncOpenAI

from insight_o3.utils.api import query_api
from insight_o3.utils.format import format_messages


@dataclass
class Conversation:
    agent: str             # 'vreasoner' | 'vsearcher' | ... (agent/model name),
    messages: list[dict]   # list of messages in the conversation

    def __str__(self):
        s = ""
        s += f">>>>>>>>>>>>>>>>>>> {self.agent} messages start >>>>>>>>>>>>>>>>>>>\n"
        s += format_messages(self.messages)
        s += f"<<<<<<<<<<<<<<<<<<< {self.agent} messages stop <<<<<<<<<<<<<<<<<<<<\n"
        return s


@dataclass
class InferenceResult:
    success: bool
    fail_reason: str = ""
    fail_detail: str = ""
    finish_reason: str = ""
    conversations: list[Conversation] = field(default_factory=list)
    last_message_content: str = ""
    profile: dict[str, float | dict[str, float]] = field(default_factory=dict)


def extract_token_usage(response: object) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "api_calls_with_usage": 0,
        }

    prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
    completion_tokens_details = getattr(usage, "completion_tokens_details", None)
    return {
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "cached_input_tokens": int(getattr(prompt_tokens_details, "cached_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "reasoning_tokens": int(getattr(completion_tokens_details, "reasoning_tokens", 0) or 0),
        "api_calls_with_usage": 1,
    }


def normalize_message_content(content: object) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        return "".join(text_parts) if text_parts else None
    return str(content)


def summarize_raw_value(value: object, max_chars: int = 500) -> str:
    text = repr(value)
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def classify_response_parse_error(error: Exception, finish_reason: str = "") -> str:
    if isinstance(error, ValueError) and str(error).startswith("assistant message content is empty"):
        if finish_reason == "length":
            return "empty_assistant_message_content:length"
        if finish_reason == "stop":
            return "empty_assistant_message_content:stop"
        return "empty_assistant_message_content"
    return "failed_to_parse_response"


def build_response_parse_fail_detail(
    error: Exception,
    finish_reason: str = "",
    raw_message_content: object = None,
    raw_reasoning_content: object = None,
) -> str:
    detail = str(error)
    if isinstance(error, ValueError) and str(error).startswith("assistant message content is empty"):
        if finish_reason == "length":
            detail += (
                " | likely_cause=model hit max_completion_tokens before it finished thinking "
                "and emitted the final answer"
            )
        elif finish_reason == "stop":
            detail += (
                " | likely_cause=stop sequence or chat template interaction cut generation "
                "before </think> or before answer text"
            )
        detail += (
            f" | finish_reason={finish_reason or '[missing]'}"
            f" | reasoning_content_present={raw_reasoning_content is not None}"
            f" | reasoning_content={summarize_raw_value(raw_reasoning_content)}"
            f" | content={summarize_raw_value(raw_message_content)}"
        )
    return detail


def maybe_resize_image(image: Image.Image, min_pixels: int | None, max_pixels: int | None) -> Image.Image:
    """ Maybe resize image to fit the minimum and maximum number of pixels.
    When both min_pixels and max_pixels are provided but cannot be satisfied at the same time, max_pixels will be used.
    """
    if not min_pixels and not max_pixels:
        return image

    w, h = image.size
    if min_pixels and w * h < min_pixels:
        ratio = (min_pixels / (w * h)) ** 0.5
        new_w = ceil(w * ratio)
        new_h = ceil(h * ratio)
        image = image.resize((new_w, new_h), Image.LANCZOS)

    w, h = image.size
    if max_pixels and w * h > max_pixels:
        ratio = (max_pixels / (w * h)) ** 0.5
        new_w = floor(w * ratio)
        new_h = floor(h * ratio)
        image = image.resize((new_w, new_h), Image.LANCZOS)

    return image


def maybe_rescale_image(image: Image.Image, ratio: float | None) -> Image.Image:
    if ratio is None or ratio == 1.0:
        return image

    w, h = image.size
    new_w = max(1, round(w * ratio))
    new_h = max(1, round(h * ratio))
    return image.resize((new_w, new_h), Image.LANCZOS)


def prepare_image(
    image_path: str,
    min_pixels: int | None,
    max_pixels: int | None,
    rescale_ratio: float | None = None,
    image_format: str = "png",
) -> tuple[str, dict[str, float]]:
    """ Prepare image in base64 format. Resize the image if necessary. """
    load_start = perf_counter()
    image = Image.open(image_path).convert("RGB")
    load_seconds = perf_counter() - load_start

    resize_start = perf_counter()
    image = maybe_rescale_image(image, rescale_ratio)
    image = maybe_resize_image(image, min_pixels, max_pixels)
    resize_seconds = perf_counter() - resize_start

    encode_start = perf_counter()
    buffered = io.BytesIO()
    image.save(buffered, format={"png": "PNG", "jpg": "JPEG"}[image_format])
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    encode_seconds = perf_counter() - encode_start

    profile = {
        "load_seconds": load_seconds,
        "resize_seconds": resize_seconds,
        "encode_seconds": encode_seconds,
        "total_seconds": load_seconds + resize_seconds + encode_seconds,
    }
    return image_base64, profile


def summarize_image_prep_profiles(profiles: list[dict[str, float]]) -> dict[str, float]:
    return {
        "image_count": float(len(profiles)),
        "load_seconds": float(sum(profile["load_seconds"] for profile in profiles)),
        "resize_seconds": float(sum(profile["resize_seconds"] for profile in profiles)),
        "encode_seconds": float(sum(profile["encode_seconds"] for profile in profiles)),
        "total_seconds": float(sum(profile["total_seconds"] for profile in profiles)),
    }


async def query_api_vqa(
    user_prompt: str,
    model: str,
    client: AsyncOpenAI,
    image_paths: list[str] | None = None,
    image_path: str | None = None,
    image_rescale_ratio: float | None = None,
    image_format: str = "png",
    image_max_pixels: int | None = None,
    image_url_extra_settings: dict | None = None,
    system_prompt: str | None = None,
    n: int = 1,
    **kwargs,
) -> tuple[list[InferenceResult], dict[str, int]]:
    if image_paths is None:
        if image_path is None:
            raise ValueError("Either image_paths or image_path must be provided")
        image_paths = [image_path]
    elif image_path is not None:
        raise ValueError("Provide either image_paths or image_path, not both")

    if not image_paths:
        raise ValueError("image_paths must not be empty")

    image_urls = []
    image_prep_profiles = []
    for current_image_path in image_paths:
        image_base64, image_profile = prepare_image(
            current_image_path,
            None,
            image_max_pixels,
            image_rescale_ratio,
            image_format,
        )
        image_urls.append(f"data:image/{image_format};base64,{image_base64}")
        image_prep_profiles.append(image_profile)
    image_prep_profile = summarize_image_prep_profiles(image_prep_profiles)

    request_start = perf_counter()
    try:
        query_messages, response = await query_api(
            query=user_prompt,
            model=model,
            client=client,
            image_urls=image_urls,
            image_url_extra_settings=image_url_extra_settings,
            context=[{"role": "system", "content": system_prompt}] if system_prompt else [],
            n=n,
            **kwargs,
        )
    except Exception as e:
        request_wait_seconds = perf_counter() - request_start
        print(f"WARNING: Failed to query API: {e}")
        profile = {
            "image_preprocess": image_prep_profile,
            "request_wait_seconds": request_wait_seconds,
            "response_parse_seconds": 0.0,
            "total_seconds": image_prep_profile["total_seconds"] + request_wait_seconds,
        }
        return [InferenceResult(success=False, fail_reason='api_error', profile=profile) for _ in range(n)], extract_token_usage(None)
    request_wait_seconds = perf_counter() - request_start
    token_usage = extract_token_usage(response)

    assert isinstance(response.choices, list), f"Unexpected response choices type: {type(response.choices)}"
    assert len(response.choices) == n, f"Unexpected number of choices: {len(response.choices)}!={n}"
    
    results = []
    for choice in response.choices:
        parse_start = perf_counter()
        finish_reason = ""
        raw_message_content = None
        raw_reasoning_content = None
        try:
            finish_reason = choice.finish_reason
            message = choice.message
            assert message.role == "assistant", f"Unexpected message role: {message.role}"
            raw_message_content = message.content
            raw_reasoning_content = getattr(message, "reasoning", None)
            if raw_reasoning_content is None:
                raw_reasoning_content = getattr(message, "reasoning_content", None)
            last_message_content = normalize_message_content(raw_message_content)
            if last_message_content is None:
                raise ValueError(
                    "assistant message content is empty "
                    f"(raw_type={type(raw_message_content).__name__}, raw_value={raw_message_content!r})"
                )
        except Exception as e:
            print(f"WARNING: Failed to parse API response: {e}")
            parse_seconds = perf_counter() - parse_start
            result = InferenceResult(
                success=False,
                fail_reason=classify_response_parse_error(e, finish_reason=finish_reason),
                fail_detail=build_response_parse_fail_detail(
                    e,
                    finish_reason=finish_reason,
                    raw_message_content=raw_message_content,
                    raw_reasoning_content=raw_reasoning_content,
                ),
                profile={
                    "image_preprocess": image_prep_profile,
                    "request_wait_seconds": request_wait_seconds,
                    "response_parse_seconds": parse_seconds,
                    "total_seconds": image_prep_profile["total_seconds"] + request_wait_seconds + parse_seconds,
                },
            )
        else:
            parse_seconds = perf_counter() - parse_start
            result = InferenceResult(
                success=True,
                finish_reason=finish_reason,
                conversations=[
                    Conversation(
                        agent=model,
                        messages=[*query_messages, message.to_dict()]
                    )
                ],
                last_message_content=last_message_content,
                profile={
                    "image_preprocess": image_prep_profile,
                    "request_wait_seconds": request_wait_seconds,
                    "response_parse_seconds": parse_seconds,
                    "total_seconds": image_prep_profile["total_seconds"] + request_wait_seconds + parse_seconds,
                },
            )
        results.append(result)
    return results, token_usage
