from dataclasses import dataclass, field
from math import ceil, floor
import base64
import io
import os

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
    finish_reason: str = ""
    conversations: list[Conversation] = field(default_factory=list)
    last_message_content: str = ""


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


def prepare_image(image_path: str, min_pixels: int | None, max_pixels: int | None) -> str:
    """ Prepare image in base64 format. Resize the image if necessary. """
    image = Image.open(image_path).convert("RGB")
    image = maybe_resize_image(image, min_pixels, max_pixels)

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


async def query_api_vqa(
    image_path: str,
    user_prompt: str,
    model: str,
    client: AsyncOpenAI,
    image_max_pixels: int | None = None,
    system_prompt: str | None = None,
    n: int = 1,
    **kwargs,
) -> list[InferenceResult]:

    image_url = f"data:image/jpeg;base64,{prepare_image(image_path, None, image_max_pixels)}"
    try:
        query_messages, response = await query_api(
            query=user_prompt,
            model=model,
            client=client,
            image_url=image_url,
            image_detail="high",
            context=[{"role": "system", "content": system_prompt}] if system_prompt else [],
            n=n,
            **kwargs,
        )
    except Exception as e:
        print(f"WARNING: Failed to query API: {e}")
        return [InferenceResult(success=False, fail_reason='api_error') for _ in range(n)]

    assert isinstance(response.choices, list), f"Unexpected response choices type: {type(response.choices)}"
    assert len(response.choices) == n, f"Unexpected number of choices: {len(response.choices)}!={n}"
    
    results = []
    for choice in response.choices:
        try:
            finish_reason = choice.finish_reason
            message = choice.message
            assert message.role == "assistant", f"Unexpected message role: {message.role}"
            last_message_content = message.content
        except Exception as e:
            print(f"WARNING: Failed to parse API response: {e}")
            result = InferenceResult(success=False, fail_reason='failed_to_parse_response')
        else:
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
            )
        results.append(result)
    return results