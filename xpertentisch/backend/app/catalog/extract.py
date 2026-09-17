"""Extrahiert den TiSCH-Katalog aus 12 HTML-Dateien und api_server.py.

api_server.py wird nur gelesen, nicht geändert.
Jede extrahierte ``position`` muss als Bytes in der Quelle vorkommen.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

from app.catalog.schema import Antagonist, Katalog, Perspektive, Tisch

HTML_FILES = (
    "tisch-hub.html",
    "index.html",
    "integrationstisch.html",
    "der-tisch.html",
    "coachingtisch.html",
    "expertentisch.html",
    "familientisch.html",
    "juristisch.html",
    "wirtschaftstisch.html",
    "literatentisch.html",
    "medizintisch.html",
    "trainingstisch.html",
)

ID_FOR_HTML = {
    "tisch-hub.html": "hub",
    "index.html": "teamtisch",
    "integrationstisch.html": "integrationstisch",
    "der-tisch.html": "der-tisch",
    "coachingtisch.html": "coachingtisch",
    "expertentisch.html": "expertentisch",
    "familientisch.html": "familientisch",
    "juristisch.html": "juristisch",
    "wirtschaftstisch.html": "wirtschaftstisch",
    "literatentisch.html": "literatentisch",
    "medizintisch.html": "medizintisch",
    "trainingstisch.html": "trainingstisch",
}

PERSPEKTIVEN_RE = re.compile(
    r"const\s+PERSPEKTIVEN\s*=\s*(\[.*?\]);",
    re.S,
)
MODE_RE = re.compile(r"setMode\('([a-z0-9_-]+)'\)")
TITLE_RE = re.compile(r"<title>([^<]*)</title>", re.I)
OBJ_RE = re.compile(
    r"\{\s*id:\s*'([^']+)'\s*,\s*icon:\s*'([^']*)'\s*,\s*"
    r"de:\s*'((?:\\'|[^'])*)'\s*,\s*en:\s*'((?:\\'|[^'])*)'\s*,\s*"
    r"color:\s*'([^']*)'",
    re.S,
)


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "der-tisch-backend" / "api_server.py").exists():
            return p
    raise FileNotFoundError("der-tisch-backend/api_server.py nicht gefunden")


def backend_dir() -> Path:
    return repo_root() / "der-tisch-backend"


def tische_dir() -> Path:
    return Path(__file__).resolve().parent / "tische"


def assert_position_bytes(source: bytes, position: str, herkunft: str) -> None:
    if not position:
        return
    needle = position.encode("utf-8")
    if needle not in source:
        raise AssertionError(
            f"position nicht byteidentisch in {herkunft}: {position[:80]!r}"
        )


def _join_str(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _join_str(node.left) + _join_str(node.right)
    raise TypeError(type(node).__name__)


def _dict_of_strings(node: ast.Dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in zip(node.keys, node.values):
        if k is None:
            continue
        out[_join_str(k)] = _join_str(v)
    return out


def extract_api_server(path: Path) -> tuple[dict[str, str], dict[str, str], list[Antagonist]]:
    src = path.read_text(encoding="utf-8")
    raw = src.encode("utf-8")
    tree = ast.parse(src)
    agents_de: dict[str, str] = {}
    agents_en: dict[str, str] = {}
    antagonisten: list[Antagonist] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id == "AGENTS_DE" and isinstance(node.value, ast.Dict):
            agents_de = _dict_of_strings(node.value)
        elif target.id == "AGENTS_EN" and isinstance(node.value, ast.Dict):
            agents_en = _dict_of_strings(node.value)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "id", None) or getattr(getattr(func, "attr", None), "real", None)
        if getattr(func, "id", "") != "ArenaPair" and getattr(func, "attr", "") != "ArenaPair":
            continue
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        if "position" not in kw or "id" not in kw:
            continue
        position = _join_str(kw["position"])
        assert_position_bytes(raw, position, str(path))
        antagonisten.append(
            Antagonist(
                id=_join_str(kw["id"]),
                name=_join_str(kw["name"]) if "name" in kw else "",
                fachgebiet=_join_str(kw["fachgebiet"]) if "fachgebiet" in kw else "",
                position=position,
                grundhaltung=_join_str(kw["grundhaltung"]) if "grundhaltung" in kw else "",
                gegenposition=_join_str(kw["gegenposition"]) if "gegenposition" in kw else "",
                rang=int(kw["rang"].value) if "rang" in kw and isinstance(kw["rang"], ast.Constant) and isinstance(kw["rang"].value, int) else None,
            )
        )
    return agents_de, agents_en, antagonisten


def _slug(name: str) -> str:
    return (
        name.lower()
        .replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
        .replace(" ", "-").replace("'", "")
    )


def extract_html(path: Path, html_name: str, agents_de: dict[str, str] | None = None) -> Tisch:
    text = path.read_text(encoding="utf-8")
    raw = text.encode("utf-8")
    luecken: list[str] = []
    agents_de = agents_de or {}
    title_m = TITLE_RE.search(text)
    title = title_m.group(1).strip() if title_m else html_name
    modi = sorted(set(MODE_RE.findall(text)))
    perspectives: list[Perspektive] = []
    block = PERSPEKTIVEN_RE.search(text)
    if not block:
        luecken.append("kein const PERSPEKTIVEN")
    else:
        for m in OBJ_RE.finditer(block.group(1)):
            perspectives.append(
                Perspektive(
                    id=m.group(1),
                    icon=m.group(2),
                    de=m.group(3).replace("\\'", "'"),
                    en=m.group(4).replace("\\'", "'"),
                    color=m.group(5),
                    quelle="html-js",
                )
            )
        if not perspectives:
            luecken.append("PERSPEKTIVEN-Block ohne parsebare Einträge")
    # JS-Objektfeld position: '...' — nicht CSS
    for m in re.finditer(r"(?<![-\\w])position:\s*'((?:\\'|[^'])*)'", text):
        pos = m.group(1).replace("\\'", "'")
        if len(pos) < 20:
            continue
        assert_position_bytes(raw, pos, str(path))
        if perspectives and not perspectives[0].position:
            perspectives[0].position = pos
        else:
            luecken.append("zusätzliche JS-position ohne PERSPEKTIVEN-Zuordnung")
    tisch_id = ID_FOR_HTML[html_name]
    if not perspectives:
        for m in re.finditer(r"'([^']+)':\s*\{\s*iconClass:", text):
            name = m.group(1)
            if name in agents_de:
                perspectives.append(
                    Perspektive(
                        id=_slug(name),
                        de=name,
                        position="",
                        quelle="roleConfig+AGENTS_DE-name",
                    )
                )
        if perspectives:
            luecken.append("PERSPEKTIVEN fehlte; Rollen aus roleConfig+AGENTS_DE")
        elif "kein const PERSPEKTIVEN" not in luecken:
            luecken.append("kein const PERSPEKTIVEN")
    if tisch_id == "familientisch" and "frage" not in modi:
        modi = ["frage", *modi]
    return Tisch(
        id=tisch_id,
        html=f"der-tisch-backend/{html_name}",
        title=title,
        modi=modi,
        perspectives=perspectives,
        luecken=luecken,
    )


def yaml_escape(s: str) -> str:
    if s == "":
        return '""'
    if any(c in s for c in ":#{}[]&*!|>'\"%@`\n"):
        return json.dumps(s, ensure_ascii=False)
    return s


def dump_tisch(t: Tisch) -> str:
    lines = [
        f"id: {t.id}",
        f"html: {yaml_escape(t.html)}",
        f"title: {yaml_escape(t.title)}",
        "modi:",
    ]
    if not t.modi:
        lines.append("  []")
    else:
        for m in t.modi:
            lines.append(f"  - {m}")
    lines.append("perspectives:")
    if not t.perspectives:
        lines.append("  []")
    else:
        for p in t.perspectives:
            lines.append(f"  - id: {p.id}")
            lines.append(f"    de: {yaml_escape(p.de)}")
            lines.append(f"    en: {yaml_escape(p.en)}")
            lines.append(f"    color: {yaml_escape(p.color)}")
            lines.append(f"    icon: {yaml_escape(p.icon)}")
            lines.append(f"    position: {yaml_escape(p.position)}")
            lines.append(f"    quelle: {p.quelle}")
    lines.append("luecken:")
    if not t.luecken:
        lines.append("  []")
    else:
        for l in t.luecken:
            lines.append(f"  - {yaml_escape(l)}")
    lines.append("")
    return "\n".join(lines)


def extract() -> Katalog:
    root = backend_dir()
    html_paths = [root / name for name in HTML_FILES]
    missing = [p.name for p in html_paths if not p.exists()]
    api = root / "api_server.py"
    agents_de, agents_en, antagonisten = extract_api_server(api)
    tische = [extract_html(p, p.name, agents_de) for p in html_paths if p.exists()]
    # der-tisch table gets method names from AGENTS_DE
    for t in tische:
        if t.id in ("der-tisch", "teamtisch", "integrationstisch"):
            t.methoden_de = list(agents_de.keys())
    luecken = []
    if missing:
        luecken.append("fehlende HTML: " + ", ".join(missing))
    if len(html_paths) != 12:
        luecken.append(f"erwartet 12 HTML, Liste hat {len(html_paths)}")
    found = len(tische)
    if found != 12:
        luecken.append(f"extrahiert {found} von 12 HTML")
    return Katalog(
        tische=tische,
        antagonisten=antagonisten,
        agenten_de=agents_de,
        agenten_en=agents_en,
        luecken=luecken,
    )


def write_katalog(kat: Katalog) -> None:
    out = tische_dir()
    out.mkdir(parents=True, exist_ok=True)
    for t in kat.tische:
        (out / f"{t.id}.yaml").write_text(dump_tisch(t), encoding="utf-8")
    (out / "_antagonisten.yaml").write_text(
        json.dumps([a.model_dump() for a in kat.antagonisten], ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (out / "_agenten.json").write_text(
        json.dumps({"de": kat.agenten_de, "en": kat.agenten_en}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def verify_positions(kat: Katalog) -> list[str]:
    """Jede nichtleere position muss in der Quelldatei stehen."""
    root = repo_root()
    api_bytes = (backend_dir() / "api_server.py").read_bytes()
    errors: list[str] = []
    for a in kat.antagonisten:
        try:
            assert_position_bytes(api_bytes, a.position, "api_server.py")
        except AssertionError as exc:
            errors.append(str(exc))
    for tisch in kat.tische:
        html_bytes = (root / tisch.html).read_bytes()
        for persp in tisch.perspectives:
            if not persp.position:
                continue
            quelle = persp.quelle or ""
            raw = api_bytes if quelle.startswith("api_server") else html_bytes
            herkunft = "api_server.py" if raw is api_bytes else tisch.html
            try:
                assert_position_bytes(raw, persp.position, herkunft)
            except AssertionError as exc:
                errors.append(str(exc))
    return errors


def main() -> int:
    kat = extract()
    Katalog.model_validate(kat.model_dump())
    write_katalog(kat)
    errors = verify_positions(kat)
    print(f"tische={len(kat.tische)} antagonisten={len(kat.antagonisten)} "
          f"agenten_de={len(kat.agenten_de)} luecken={len(kat.luecken)}")
    for t in kat.tische:
        print(f"  {t.id}: perspektiven={len(t.perspectives)} modi={len(t.modi)} luecken={t.luecken}")
    if kat.luecken:
        print("katalog-luecken:", kat.luecken)
    if errors:
        print("POSITION_FAIL", len(errors))
        for e in errors[:10]:
            print(" ", e)
        return 1
    print("position byteidentisch: ok")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    raise SystemExit(main())
