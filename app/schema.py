from typing import List

from pydantic import BaseModel, Field


class ChatRequestBody(BaseModel):
    messages: List[dict]
    model: str
    stream: bool = Field(default=False)
    temperature: float = Field(default=0.7)
