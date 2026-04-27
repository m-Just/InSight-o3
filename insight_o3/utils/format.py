
def middle_truncate(text: str, max_chars: int = 1000) -> str:
    """Return *text* as-is if short enough, otherwise keep the first and last
    ``max_chars // 2`` characters with a truncation marker in between."""
    if len(text) <= max_chars:
        return text
    keep = max_chars // 2
    omitted = len(text) - 2 * keep
    return f"{text[:keep]}\n... [{omitted} chars truncated] ...\n{text[-keep:]}"


def _format_reasoning(message: dict, max_chars: int) -> str:
    """Extract and format the reasoning field of a message, if present."""
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    if not reasoning:
        return ""
    return f"--- reasoning ---\n{middle_truncate(str(reasoning), max_chars)}\n--- end reasoning ---\n"


def format_messages(
    messages: list[dict],
    omit_image_data: bool = True,
    omit_system_prompt: bool = True,
    max_reasoning_chars: int = 1000,
) -> str:
    s = ""

    for message in messages:
        role = message.get('role', 'unknown').upper()
        if role == 'SYSTEM' and omit_system_prompt:
            continue
        s += f'[{role}]\n'

        s += _format_reasoning(message, max_reasoning_chars)

        if 'content' not in message:
            s += f"<content missing>\n"
            continue

        if isinstance(message['content'], str):
            s += message['content']
        elif isinstance(message['content'], list):
            for content in message['content']:
                if not isinstance(content, dict):
                    s += f"<unexpected content part type: {type(content)}>"
                    continue
                if 'type' not in content:
                    s += f"<content part missing 'type'>"
                    continue
                if content['type'] == 'text':
                    s += content.get('text', '<content part text missing>')
                elif 'image_url' in content:
                    if omit_image_data:
                        s += '<image>'
                    elif isinstance(content['image_url'], str):
                        s += f"[image_url]({content['image_url']})"
                    elif isinstance(content['image_url'], dict):
                        image_url = content['image_url'].get('url', '<image url missing>')
                        s += f"[image_url]({image_url})"
                    else:
                        s += f"<unexpected image_url type: {type(content['image_url'])}>"
                else:
                    s += f"<unexpected content part type: {content['type']}>"
        else:
            s += f"<unexpected content type: {type(message['content'])}>"

        s += '\n'

    return s