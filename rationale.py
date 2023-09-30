from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from typing import Literal
import openai
import sys
import openai
#from decouple import config
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

openai.api_key = "" 
OPENAI_MODEL = "gpt-3.5-turbo"

@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: str

    @staticmethod
    def from_response(response: dict) -> Message:
        response["choices"][0]["message"]
        return Message(**(response["choices"][0]["message"]))

    def __str__(self) -> str:
        return f"{self.role}: {self.content}"


class Transcript(list[Message]):
    def raw(self) -> list[dict]:
        return [asdict(m) for m in self]

    def __str__(self) -> str:
        return "\n".join([str(m) for m in self])


def chat_step(transcript_state: Transcript) -> Message:
    chat_completion = openai.ChatCompletion.create(
        model=OPENAI_MODEL, messages=transcript_state.raw()
    )
    if not isinstance(chat_completion, dict):
        raise Exception(f"Unexpected response: {chat_completion}")
    response_message = Message.from_response(chat_completion)
    transcript_state.append(response_message)
    return response_message

def openai_retinale(code: str) -> str:
    prompt = f"""Explain the below moonscript code within 50 words:

    {code}
    """
    transcript = Transcript(
          [
              Message("user", prompt),
          ]
      )
    m = chat_step(transcript)
    return m.content