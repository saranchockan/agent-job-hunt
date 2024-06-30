import os
from typing import List, Optional

from anthropic import Anthropic
from anthropic.types import Message

from agent_job_hunt.llm.services.models import CLAUDE_THREE_OPUS_MODEL

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
ANTHROPIC_AI_CLIENT = Anthropic(api_key=ANTHROPIC_API_KEY)


def fetch_anthropic_ai_response(
    prompt: str, images: Optional[List[bytes]] = [], max_tokens: Optional[int] = 1024
) -> str:
    """
    Sends a prompt and optional images to the Anthropic AI API and returns the AI's response.

    Args:
        prompt (str): The prompt to send to the AI.
        images (Optional[List[bytes]], optional): A list of image bytes to include in the request.
        Defaults to None.
        model (str, optional): The name of the AI model to use. Defaults to CLAUDE_THREE_OPUS_MODEL.
        max_tokens (int, optional): The maximum number of tokens to generate in the AI's response.
        Defaults to 1024.

    Returns:
        str: The AI's response as a string.
    """
    return __extract_message_from_anthropic_ai_reponse(
        ANTHROPIC_AI_CLIENT.messages.create(
            model=CLAUDE_THREE_OPUS_MODEL,
            max_tokens=max_tokens,
            messages=[
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image,
                    },
                }
                for image in images
            ]
            + [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
    )


def __extract_message_from_anthropic_ai_reponse(message: Message) -> str:
    """
    Extracts and concatenates all text content from a Message object.

    Args:
        message (Message): A Message object returned by the Anthropic Python SDK.

    Returns:
        str: A string containing all the text content from the Message object.
    """
    text = ""
    for content_item in message.content:
        if content_item.type == "text":
            text += content_item.text
    return text
