import os
from typing import List, Optional, Union

from anthropic import Anthropic
from anthropic.types import ImageBlockParam, Message, TextBlockParam, MessageParam

from agent_job_hunt.llm.services.models import CLAUDE_THREE_OPUS_MODEL

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
ANTHROPIC_AI_CLIENT = Anthropic(api_key=ANTHROPIC_API_KEY)


def fetch_anthropic_ai_response(
    prompt: str, images: List[bytes] = [], max_tokens: int = 1024
) -> Union[str, None]:
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

    image_content: List[ImageBlockParam] = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": image.decode("utf-8"),
            },
        }
        for image in images
    ]
    text_content: List[TextBlockParam] = [
        {
            "text": prompt,
            "type": "text",
        }
    ]
    return __extract_message_from_anthropic_ai_reponse(
        ANTHROPIC_AI_CLIENT.messages.create(
            model=CLAUDE_THREE_OPUS_MODEL,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": image_content + text_content,
                }
            ],
        )
    )


def __extract_message_from_anthropic_ai_reponse(message: Message) -> Union[str, None]:
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
