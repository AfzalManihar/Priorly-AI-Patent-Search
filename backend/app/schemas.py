"""Request models (PRD section 10)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class OptionsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decompose: bool = True
    hyde: bool = True
    hybrid: bool = True
    rerank: bool = True


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: str
    options: OptionsModel = Field(default_factory=OptionsModel)
    k: Literal[5, 10] = 10


class ReportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    search_id: str
    title: str = Field(min_length=1, max_length=200)
    shortlist: list[str] = Field(default_factory=list, max_length=50)


class SettingsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    embedding_model: str | None = None
    reranker: str | None = None
    llm_model: str | None = None
    rrf_k: int | None = Field(default=None, ge=1, le=200)
    candidates: int | None = Field(default=None, ge=10, le=200)
    rerank_depth: int | None = Field(default=None, ge=10, le=100)
