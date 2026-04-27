import os
import time
import uuid
import json

from openai.types.chat import ChatCompletion


SAVE_DIR = os.path.expanduser(os.getenv("API_LOGGER_SAVE_DIR", "~/.dumps/api_requests"))
PROJECT_NAME = os.getenv("API_LOGGER_PROJECT_NAME", "default_project")


def log_chat_completion(messages: list[dict], chat_completion: ChatCompletion, api_key: str, base_url: str):
    """
    Log the chat completion to the local file system.

    Args:
        messages: list[dict] - the messages sent to the API
        chat_completion: ChatCompletion - the chat completion response from the API
        api_key: str - the API key used to make the request (only the last 8 characters are saved)
        base_url: str - the base URL used to make the request
    """
    date_str = time.strftime("%Y-%m-%d")
    save_path = os.path.join(SAVE_DIR, date_str, PROJECT_NAME, str(uuid.uuid4()))
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "messages.jsonl"), "w") as f:
        json_str = json.dumps(messages, ensure_ascii=False)
        f.write(json_str + "\n")
    with open(os.path.join(save_path, "response.jsonl"), "w") as f:
        json_str = json.dumps(chat_completion.to_dict(), ensure_ascii=False)
        f.write(json_str + "\n")
    with open(os.path.join(save_path, "api_url_and_key.jsonl"), "w") as f:
        api_url_and_key = {"api_key": api_key[-8:], "base_url": base_url}
        json_str = json.dumps(api_url_and_key, ensure_ascii=False)
        f.write(json_str + "\n")