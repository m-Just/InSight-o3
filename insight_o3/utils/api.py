import asyncio
from pprint import pformat

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage

try:
    from .api_logger import log_chat_completion
except ImportError:
    log_chat_completion = None


def prune_non_text_content(message: dict | ChatCompletionMessage) -> dict:
    if isinstance(message, ChatCompletionMessage):
        message = message.to_dict()
    message_pruned = {}
    for key, value in message.items():
        if key != "content":
            message_pruned[key] = value
            continue
        if value is None:
            message_pruned[key] = None
        elif isinstance(value, str):
            message_pruned[key] = value
        else:
            message_pruned[key] = []
            for v in value:
                if v["type"] == "text":
                    message_pruned[key].append(v)
                else:
                    message_pruned[key].append(
                        {
                            "type": v["type"],
                            v["type"]: "[pruned]",
                        }
                    )
    return message_pruned


def _format_error_message(
    err: Exception,
    model: str,
    client: AsyncOpenAI,
    messages: list[dict],
    show_detailed_error_message: bool = False,
) -> str:
    error_message = f"failed to query \"{model}\" at {client.base_url}: {repr(err)}"
    if show_detailed_error_message:
        error_message += f"\n  Client information:\n    {client.timeout=}\n    {client.max_retries=}"
        formatted_input_messages = pformat([prune_non_text_content(message) for message in messages])
        formatted_input_messages = formatted_input_messages.replace("\n", "\n    ")
        error_message += f"\n  Input messages:\n    {formatted_input_messages}"
    return error_message


async def complete_chat_and_maybe_log(
    messages: list[dict],
    model: str,
    client: AsyncOpenAI,
    show_detailed_error_message: bool = False,
    **chat_completion_kwargs,
) -> ChatCompletion:

    assert isinstance(client, AsyncOpenAI), "client must be an instance of AsyncOpenAI"

    try:
        response = await client.chat.completions.create(
            messages=messages,
            model=model,
            **chat_completion_kwargs,
        )

    except Exception as err:
        raise RuntimeError(_format_error_message(err, model, client, messages, show_detailed_error_message)) from err

    if log_chat_completion:
        messages_to_log = [prune_non_text_content(message) for message in messages]
        log_chat_completion(messages_to_log, response, client.api_key, str(client.base_url))

    return response


async def query_api(
    query: str | list[dict],
    model: str,
    client: AsyncOpenAI,
    image_url: str | None = None,        # image url for image input
    image_detail: str = "auto",          # "low", "auto", or "high" for gpt models
    context: list[dict] | None = None,   # context messages for multi-round conversation
    **kwargs,
) -> tuple[list[dict], ChatCompletion]:

    if isinstance(query, str):
        query = [{"type": "text", "text": query}]

    if image_url is not None:
        query.insert(0, {
            "type": "image_url",
            "image_url": {"url": image_url, "detail": image_detail},
        })

    messages = [*context] if context else []
    messages.append({"role": "user", "content": query})

    return messages, await complete_chat_and_maybe_log(
        messages=messages,
        model=model,
        client=client,
        **kwargs,
    )


if __name__ == "__main__":
    async def _demo():
        client = AsyncOpenAI()
        try:
            _, response = await query_api(
                query="What is the capital of France?",
                model="gpt-5-nano",
                client=client,
                n=1,
            )
            print(response)
        finally:
            await client.close()

    asyncio.run(_demo())