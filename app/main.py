import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import api_router
from app import version

load_dotenv()

_logger = logging.getLogger("uvicorn")


def get_application():
    # Core Application Instance
    _logger.info("Creating core application instance")
    _app = FastAPI(title="Brave Leo Search Functionality", version=version)

    # Add routers
    _app.include_router(api_router)

    _app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],
    )

    _logger.info("Core application instance created successfully")
    return _app


app = get_application()
