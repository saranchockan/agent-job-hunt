import os
from typing import Optional

import requests

from agent_job_hunt.llm.services.models import SONAR_SMALL_ONLINE_MODEL

PERPLEXITY_API_TOKEN = os.environ["PERPLEXITY_API_TOKEN"]


def fetch_perplexity_ai_response(
    user_prompt: str, model: Optional[str] = SONAR_SMALL_ONLINE_MODEL
) -> str:
    """
    Fetches a response from the Perplexity AI API based on the given user prompt and model.

    Args:
        user_prompt (str): The user's prompt or message.
        model (Optional[str], optional): The model to be used for generating the response. Defaults to SONAR_SMALL_ONLINE_MODEL.

    Returns:
        str: The generated response from the Perplexity AI API.
    """

    return __extract_message_from_perplexity_ai_response(
        response=requests.post(
            "https://api.perplexity.ai/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": user_prompt},
                ],
            },
            headers={
                "accept": "application/json",
                "content-type": "application/json",
                "authorization": f"Bearer {PERPLEXITY_API_TOKEN}",
            },
        ).json()
    )


def __extract_message_from_perplexity_ai_response(response) -> str:
    """
    Extracts the message content from the Perplexity AI API response.

    Args:
        response (dict): The API response as a dictionary.

    Returns:
        str: The extracted message content.
    """
    return response["choices"][0]["message"]["content"]
