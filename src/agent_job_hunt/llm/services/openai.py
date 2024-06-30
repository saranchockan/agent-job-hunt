from typing import Union, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletion

from agent_job_hunt.llm.services.models import GPT_FOUR_TURBO_MODEL

OPENAI_CLIENT = OpenAI()


def fetch_open_ai_response(
    prompt: str, model: str = GPT_FOUR_TURBO_MODEL
) -> Union[str, None]:
    """
    Generate a response from the OpenAI API based on the given prompt.

    Args:
        prompt (str): The input message or question to be sent to the OpenAI API.
        model (str, optional): The OpenAI model to use for generating the response.
            Defaults to GPT_FOUR_TURBO_MODEL.

    Returns:
        str: The generated response from the OpenAI API.
            If no response is available, returns an empty string.
    """
    try:
        return __extract_message_from_openai_response(
            response=OPENAI_CLIENT.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
        )
    except Exception as e:
        print(f"Error generating OpenAI response: {e}")
        return None


def __extract_message_from_openai_response(
    response: ChatCompletion,
) -> Union[str, None]:
    """
    Extract the generated message from the OpenAI API response.

    Args:
        response (ChatCompletion): The response object returned by the OpenAI API.

    Returns:
        str: The generated message extracted from the response.
            If no message is available, returns an empty string.
    """
    try:
        if response.choices:
            message = response.choices[0].message.content
            return message
        else:
            return ""
    except Exception as e:
        print(f"Error extracting message from OpenAI response: {e}")
        return ""
