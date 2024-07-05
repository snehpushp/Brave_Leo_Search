from fastapi import APIRouter

from app.api.endpoints.with_search import openai_router

api_router = APIRouter()

api_router.include_router(openai_router)
