"""Katalog-Extrakt: 12 HTML, Schema, byteidentische position."""

from __future__ import annotations

from pathlib import Path

from app.catalog.extract import HTML_FILES, extract, verify_positions, write_katalog
from app.catalog.schema import Katalog, Tisch


def test_zwoelf_html_vorhanden():
    from app.catalog.extract import backend_dir

    root = backend_dir()
    missing = [n for n in HTML_FILES if not (root / n).exists()]
    assert missing == []
    assert len(HTML_FILES) == 12


def test_extract_schema_und_positionen(tmp_path, monkeypatch):
    kat = extract()
    Katalog.model_validate(kat.model_dump())
    assert len(kat.tische) == 12
    assert len(kat.agenten_de) == 22
    assert kat.antagonisten, "ArenaPair.position muss extrahiert sein"
    errors = verify_positions(kat)
    assert errors == [], errors[:5]
    ids = {t.id for t in kat.tische}
    assert "familientisch" in ids
    fam = next(t for t in kat.tische if t.id == "familientisch")
    assert "frage" in fam.modi
    assert fam.perspectives
    for a in kat.antagonisten:
        assert a.position
        Tisch.model_validate  # schema import used
        assert isinstance(a.position, str)


def test_yaml_wird_geschrieben():
    kat = extract()
    write_katalog(kat)
    from app.catalog.extract import tische_dir

    d = tische_dir()
    yamls = list(d.glob("*.yaml"))
    assert len([p for p in yamls if not p.name.startswith("_")]) == 12
