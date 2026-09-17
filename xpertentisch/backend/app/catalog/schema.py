"""Pydantic-Schema für den TiSCH-Katalog (Phase 1)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Perspektive(BaseModel):
    id: str
    de: str = ""
    en: str = ""
    color: str = ""
    icon: str = ""
    position: str = ""
    grundhaltung: str = ""
    argumentationsstil: str = ""
    quelle: str = ""  # html-js | api_server.AGENTS_DE | api_server.ArenaPair


class Tisch(BaseModel):
    id: str
    html: str
    title: str
    modi: list[str] = Field(default_factory=list)
    perspectives: list[Perspektive] = Field(default_factory=list)
    methoden_de: list[str] = Field(default_factory=list)
    luecken: list[str] = Field(default_factory=list)


class Antagonist(BaseModel):
    id: str
    name: str
    fachgebiet: str
    position: str
    grundhaltung: str = ""
    gegenposition: str = ""
    rang: int | None = None


class Katalog(BaseModel):
    verfahren: Literal["extract_catalog"] = "extract_catalog"
    version: str = "1"
    tische: list[Tisch]
    antagonisten: list[Antagonist] = Field(default_factory=list)
    agenten_de: dict[str, str] = Field(default_factory=dict)
    agenten_en: dict[str, str] = Field(default_factory=dict)
    luecken: list[str] = Field(default_factory=list)
