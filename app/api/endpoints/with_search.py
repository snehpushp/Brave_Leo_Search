import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.schema import ChatRequestBody
from app.utils.search import orchestrator

_logger = logging.getLogger("uvicorn")

openai_router = APIRouter(prefix="/openai", tags=["OpenAI"])


@openai_router.post("/chat")
async def chat_completion(request_body: ChatRequestBody):
    """
    This api will facilitate chat to search the internet and then respond
    :param request_body:
    :return:
    """
    return StreamingResponse(orchestrator(messages=request_body.messages), media_type="text/event-stream")
