import os
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from tool_selector.catalog import Tool, load_catalog
from tool_selector.search import DEFAULT_CACHE, DEFAULT_CATALOG, Selector


class SelectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=8000)
    top_k: int = Field(default=15, ge=1, le=100)

    @field_validator("query")
    @classmethod
    def strip_query(cls, value):
        if not value.strip():
            raise ValueError("Query must not be blank")
        if re.search(r"[\u0400-\u04ff]", value):
            raise ValueError("Only English queries are supported; please send the request in English")
        return value.strip()


class SelectionResponse(BaseModel):
    tools: list[Tool]
    catalog_version: str


def create_app(selector: Selector | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.selector = selector or Selector(
            load_catalog(os.getenv("CATALOG_PATH", str(DEFAULT_CATALOG))),
            cache_dir=os.getenv("CACHE_DIR", str(DEFAULT_CACHE)),
        )
        yield

    app = FastAPI(title="Tool Selector", version="1.0.0", lifespan=lifespan)

    @app.post("/select-tools", response_model=SelectionResponse)
    def select_tools(body: SelectionRequest, request: Request):
        engine = request.app.state.selector
        return SelectionResponse(tools=engine.select(body.query, body.top_k), catalog_version=engine.version)

    return app


app = create_app()
