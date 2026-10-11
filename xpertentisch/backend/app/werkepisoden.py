"""Werkepisoden — Prototyp.

Eine Episode wird abgeschlossen, das Werk bleibt offen. Anliegen, Werkfassungen,
Annahmen, Post-its, Episoden und Weiterwirkungen hängen an keiner Sitzung: eine
Beobachtung überlebt die Sitzung, in der sie gemacht wurde.

Zwei Regeln tragen das Modul:
- Was die Maschine anlegt, ist ein Vorschlag — nie ein bestätigter Befund.
- Fassungen und abgeschlossene Episoden werden nicht überschrieben, nur ergänzt.

Konzept und offene Fragen: KONZEPT_Werkepisoden_2026-10-11.md im Wurzelverzeichnis.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .db import Store, new_id, now

ORIGIN_MENSCH = "mensch"
ORIGIN_MASCHINE = "maschine"

POSTIT_OFFEN = "offen"
POSTIT_VORGESCHLAGEN = "vorgeschlagen"
POSTIT_STATUS = "^(offen|vorgeschlagen|bestaetigt|entkraeftet)$"

HALTUNG_EINWAND = "einwand"

PRUEF_VORGESCHLAGEN = "vorgeschlagen"
PRUEF_ANGENOMMEN = "angenommen"
PRUEF_ABGELEHNT = "abgelehnt"
PRUEF_GEPRUEFT = "geprueft"

EPISODE_OFFEN = "offen"
EPISODE_ABGESCHLOSSEN = "abgeschlossen"

#: Welche Tabelle hinter einem Entscheidungsgegenstand steht.
GEGENSTAENDE = {
    "annahme": "annahmen",
    "postit": "postits",
    "werkfassung": "werkfassungen",
    "entscheidung": "entscheidungen",
}


class AnliegenCreate(BaseModel):
    titel: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=8000)


class AnliegenFassungCreate(BaseModel):
    text: str = Field(min_length=1, max_length=8000)


class WerkCreate(BaseModel):
    titel: str = Field(min_length=1, max_length=200)
    art: str = Field(default="", max_length=60)


class Bilanz(BaseModel):
    """Was gegenüber der Vorfassung blieb, sich änderte, hinzukam und verloren ging."""

    geblieben: list[str] = Field(default_factory=list)
    geaendert: list[str] = Field(default_factory=list)
    hinzugekommen: list[str] = Field(default_factory=list)
    verloren: list[str] = Field(default_factory=list)


class WerkfassungCreate(BaseModel):
    #: Stellen, aus denen die Fassung besteht. Post-its und Annahmen nennen sie beim Namen.
    zusammensetzung: list[str] = Field(default_factory=list, max_length=200)
    anliegen_fassung_id: str | None = None
    #: Was dieses Werk zu diesem Werk machen soll — Stand dieser Fassung, revidierbar.
    invarianten: list[str] = Field(default_factory=list, max_length=50)
    bilanz: Bilanz = Field(default_factory=Bilanz)


class AnnahmeCreate(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class VerwendungCreate(BaseModel):
    werkfassung_id: str
    stelle: str = Field(default="", max_length=200)


class Betroffen(BaseModel):
    annahme_id: str
    haltung: str = Field(default=HALTUNG_EINWAND, pattern="^(einwand|stuetze|frage)$")


class PostitCreate(BaseModel):
    werkfassung_id: str
    stelle: str = Field(default="", max_length=200)
    text: str = Field(min_length=1, max_length=4000)
    zitat: str = Field(default="", max_length=4000)
    origin: str = Field(default=ORIGIN_MENSCH, pattern="^(mensch|maschine)$")
    annahmen: list[Betroffen] = Field(default_factory=list, max_length=20)
    #: Optionaler Herkunftsverweis auf einen Auftrag.
    herkunft_job: str | None = None


class PostitAusMarker(BaseModel):
    werkfassung_id: str
    stelle: str = Field(default="", max_length=200)
    #: Eigene Worte zur Beobachtung. Leer heißt: die Notiz des Markers gilt.
    text: str = Field(default="", max_length=4000)
    annahmen: list[Betroffen] = Field(default_factory=list, max_length=20)


class PostitStatus(BaseModel):
    status: str = Field(pattern=POSTIT_STATUS)


class WeiterwirkungCreate(BaseModel):
    quelle_art: str = Field(pattern="^(postit|annahme)$")
    quelle_id: str
    annahme_id: str | None = None
    ziel_werkfassung_id: str
    origin: str = Field(default=ORIGIN_MENSCH, pattern="^(mensch|maschine)$")


class WeiterwirkungStatus(BaseModel):
    status: str = Field(pattern="^(angenommen|abgelehnt)$")


class WeiterwirkungErgebnis(BaseModel):
    ergebnis: str = Field(pattern="^(betroffen|nicht_betroffen|unklar)$")
    notiz: str = Field(default="", max_length=4000)
    #: Die Episode, in der geprüft wurde.
    episode_id: str | None = None


class Verweis(BaseModel):
    """Ein Verweis statt einer Kopie. `art` sagt, worauf `verweis` zeigt."""

    art: str = Field(min_length=1, max_length=40)
    verweis: str = Field(min_length=1, max_length=120)
    notiz: str = Field(default="", max_length=2000)


class EpisodeCreate(BaseModel):
    werk_id: str
    titel: str = Field(min_length=1, max_length=200)
    ausgangslage: str = Field(default="", max_length=8000)
    ausgang_werkfassung_id: str | None = None
    ausgang_anliegen_fassung_id: str | None = None
    ausgang_auftraege: list[str] = Field(default_factory=list, max_length=50)
    perspektiven: list[Verweis] = Field(default_factory=list, max_length=100)
    bearbeitung: list[Verweis] = Field(default_factory=list, max_length=100)
    #: Nachtrag zu einer abgeschlossenen Episode.
    ergaenzt_episode_id: str | None = None


class EpisodeUpdate(BaseModel):
    """Änderung an einer offenen Episode. `None` heißt unverändert."""

    titel: str | None = Field(default=None, min_length=1, max_length=200)
    ausgangslage: str | None = Field(default=None, max_length=8000)
    perspektiven: list[Verweis] | None = None
    bearbeitung: list[Verweis] | None = None


class EpisodePostit(BaseModel):
    postit_id: str
    rolle: str = Field(default="beobachtung", pattern="^(beobachtung|gegenbeispiel)$")


class EntscheidungCreate(BaseModel):
    art: str = Field(pattern="^(uebernommen|zurueckgestellt|verworfen|zurueckgenommen)$")
    gegenstand_art: str = Field(pattern="^(annahme|postit|werkfassung|entscheidung)$")
    gegenstand_id: str
    begruendung: str = Field(default="", max_length=4000)
    neue_werkfassung_id: str | None = None
    neue_anliegen_fassung_id: str | None = None


JSON_SPALTEN = ("zusammensetzung", "invarianten", "bilanz", "ausgang_auftraege",
                "perspektiven", "bearbeitung")


def _zeile(row: Any) -> dict[str, Any]:
    """JSON-Spalten kommen als Listen und Objekte heraus, nicht als Text."""
    daten = dict(row)
    for spalte in JSON_SPALTEN:
        if isinstance(daten.get(spalte), str):
            daten[spalte] = json.loads(daten[spalte])
    return daten


def register(app: FastAPI, store: Store) -> None:
    """Hängt die Werkepisoden-Endpunkte an die Anwendung."""

    async def _eine(sql: str, werte: tuple = ()) -> dict[str, Any] | None:
        async with store.conn.execute(sql, werte) as cur:
            row = await cur.fetchone()
        return _zeile(row) if row else None

    async def _alle(sql: str, werte: tuple = ()) -> list[dict[str, Any]]:
        async with store.conn.execute(sql, werte) as cur:
            return [_zeile(r) for r in await cur.fetchall()]

    async def _schreibe(sql: str, werte: tuple | dict) -> None:
        await store.conn.execute(sql, werte)
        await store.conn.commit()

    async def _require(tabelle: str, kennung: str | None, name: str) -> dict[str, Any]:
        zeile = await _eine(f"SELECT * FROM {tabelle} WHERE id=?", (kennung,))
        if zeile is None:
            raise HTTPException(status_code=404, detail=f"{name} nicht gefunden.")
        return zeile

    async def _naechste_nummer(tabelle: str, spalte: str, kennung: str) -> int:
        zeile = await _eine(
            f"SELECT COALESCE(MAX(nummer), 0) + 1 AS n FROM {tabelle} WHERE {spalte}=?",
            (kennung,),
        )
        return int(zeile["n"])  # type: ignore[index]

    # ---------------------------------------------------------------- Anliegen

    async def _neue_anliegen_fassung(anliegen_id: str, text: str) -> dict[str, Any]:
        fassung = {
            "id": new_id("anf"),
            "anliegen_id": anliegen_id,
            "nummer": await _naechste_nummer("anliegen_fassungen", "anliegen_id", anliegen_id),
            "text": text,
            "created_at": now(),
        }
        await _schreibe(
            "INSERT INTO anliegen_fassungen (id, anliegen_id, nummer, text, created_at)"
            " VALUES (:id, :anliegen_id, :nummer, :text, :created_at)",
            fassung,
        )
        return fassung

    async def _anliegen_mit_fassungen(anliegen_id: str) -> dict[str, Any]:
        anliegen = await _require("anliegen", anliegen_id, "Anliegen")
        anliegen["fassungen"] = await _alle(
            "SELECT * FROM anliegen_fassungen WHERE anliegen_id=? ORDER BY nummer",
            (anliegen_id,),
        )
        return anliegen

    @app.post("/api/anliegen", status_code=201)
    async def create_anliegen(body: AnliegenCreate) -> dict[str, Any]:
        anliegen = {"id": new_id("anl"), "titel": body.titel.strip(), "created_at": now()}
        await _schreibe(
            "INSERT INTO anliegen (id, titel, created_at) VALUES (:id, :titel, :created_at)",
            anliegen,
        )
        await _neue_anliegen_fassung(anliegen["id"], body.text.strip())
        return await _anliegen_mit_fassungen(anliegen["id"])

    @app.post("/api/anliegen/{anliegen_id}/fassungen", status_code=201)
    async def create_anliegen_fassung(
        anliegen_id: str, body: AnliegenFassungCreate
    ) -> dict[str, Any]:
        """Eine neue Fassung des Anliegens. Die bisherigen bleiben stehen."""
        await _require("anliegen", anliegen_id, "Anliegen")
        return await _neue_anliegen_fassung(anliegen_id, body.text.strip())

    @app.get("/api/anliegen/{anliegen_id}")
    async def get_anliegen(anliegen_id: str) -> dict[str, Any]:
        return await _anliegen_mit_fassungen(anliegen_id)

    # ------------------------------------------------------------------- Werke

    @app.post("/api/werke", status_code=201)
    async def create_werk(body: WerkCreate) -> dict[str, Any]:
        werk = {"id": new_id("wrk"), "titel": body.titel.strip(), "art": body.art.strip(),
                "created_at": now()}
        await _schreibe(
            "INSERT INTO werke (id, titel, art, created_at)"
            " VALUES (:id, :titel, :art, :created_at)",
            werk,
        )
        return {**werk, "fassungen": []}

    @app.get("/api/werke")
    async def list_werke() -> dict[str, Any]:
        return {"werke": await _alle("SELECT * FROM werke ORDER BY created_at")}

    @app.get("/api/werke/{werk_id}")
    async def get_werk(werk_id: str) -> dict[str, Any]:
        werk = await _require("werke", werk_id, "Werk")
        werk["fassungen"] = await _alle(
            "SELECT * FROM werkfassungen WHERE werk_id=? ORDER BY nummer", (werk_id,)
        )
        werk["episoden"] = await _alle(
            "SELECT id, titel, status, ergaenzt_episode_id, created_at, closed_at"
            " FROM episoden WHERE werk_id=? ORDER BY created_at",
            (werk_id,),
        )
        return werk

    @app.post("/api/werke/{werk_id}/fassungen", status_code=201)
    async def create_werkfassung(werk_id: str, body: WerkfassungCreate) -> dict[str, Any]:
        await _require("werke", werk_id, "Werk")
        if body.anliegen_fassung_id:
            await _require("anliegen_fassungen", body.anliegen_fassung_id, "Anliegen-Fassung")
        fassung = {
            "id": new_id("wfs"),
            "werk_id": werk_id,
            "nummer": await _naechste_nummer("werkfassungen", "werk_id", werk_id),
            "zusammensetzung": body.zusammensetzung,
            "anliegen_fassung_id": body.anliegen_fassung_id,
            "invarianten": body.invarianten,
            "bilanz": body.bilanz.model_dump(),
            "created_at": now(),
        }
        await _schreibe(
            "INSERT INTO werkfassungen (id, werk_id, nummer, zusammensetzung,"
            " anliegen_fassung_id, invarianten, bilanz, created_at) VALUES (:id, :werk_id,"
            " :nummer, :zusammensetzung, :anliegen_fassung_id, :invarianten, :bilanz,"
            " :created_at)",
            {**fassung, **{k: json.dumps(fassung[k], ensure_ascii=False)
                           for k in ("zusammensetzung", "invarianten", "bilanz")}},
        )
        return fassung

    # ---------------------------------------------------------------- Annahmen

    async def _verwendungen(annahme_id: str) -> list[dict[str, Any]]:
        return await _alle(
            "SELECT v.id, v.werkfassung_id, v.stelle, f.werk_id, f.nummer,"
            " w.titel AS werk_titel FROM annahme_verwendungen v"
            " JOIN werkfassungen f ON f.id = v.werkfassung_id"
            " JOIN werke w ON w.id = f.werk_id"
            " WHERE v.annahme_id=? ORDER BY w.created_at, f.nummer, v.created_at",
            (annahme_id,),
        )

    @app.post("/api/annahmen", status_code=201)
    async def create_annahme(body: AnnahmeCreate) -> dict[str, Any]:
        annahme = {"id": new_id("ann"), "text": body.text.strip(), "created_at": now()}
        await _schreibe(
            "INSERT INTO annahmen (id, text, created_at) VALUES (:id, :text, :created_at)",
            annahme,
        )
        return {**annahme, "verwendungen": []}

    @app.get("/api/annahmen/{annahme_id}")
    async def get_annahme(annahme_id: str) -> dict[str, Any]:
        annahme = await _require("annahmen", annahme_id, "Annahme")
        annahme["verwendungen"] = await _verwendungen(annahme_id)
        return annahme

    @app.post("/api/annahmen/{annahme_id}/verwendungen", status_code=201)
    async def create_verwendung(annahme_id: str, body: VerwendungCreate) -> dict[str, Any]:
        """Hält fest, dass eine Werkfassung diese Annahme an einer Stelle nutzt."""
        await _require("annahmen", annahme_id, "Annahme")
        await _require("werkfassungen", body.werkfassung_id, "Werkfassung")
        await _schreibe(
            "INSERT OR IGNORE INTO annahme_verwendungen (id, annahme_id, werkfassung_id,"
            " stelle, created_at) VALUES (?,?,?,?,?)",
            (new_id("vrw"), annahme_id, body.werkfassung_id, body.stelle.strip(), now()),
        )
        return {"verwendungen": await _verwendungen(annahme_id)}

    # ---------------------------------------------------------------- Post-its

    async def _pruefauftraege_anlegen(postit: dict[str, Any], annahme_id: str) -> None:
        """Ein Einwand wandert zu jeder anderen Werkfassung mit derselben Annahme.

        Das ist ein maschineller Schluss aus gemeinsamer Annahme — also ein
        Vorschlag. Ob die andere Fassung wirklich betroffen ist, prüft jemand.
        """
        ziele = await _alle(
            "SELECT DISTINCT werkfassung_id FROM annahme_verwendungen"
            " WHERE annahme_id=? AND werkfassung_id<>?",
            (annahme_id, postit["werkfassung_id"]),
        )
        for ziel in ziele:
            await store.conn.execute(
                "INSERT OR IGNORE INTO weiterwirkungen (id, quelle_art, quelle_id,"
                " annahme_id, ziel_werkfassung_id, origin, status, created_at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (new_id("wwk"), "postit", postit["id"], annahme_id,
                 ziel["werkfassung_id"], ORIGIN_MASCHINE, PRUEF_VORGESCHLAGEN, now()),
            )
        await store.conn.commit()

    async def _annahme_anhaengen(postit: dict[str, Any], betroffen: Betroffen) -> None:
        await _schreibe(
            "INSERT INTO postit_annahmen (postit_id, annahme_id, haltung, created_at)"
            " VALUES (?,?,?,?) ON CONFLICT(postit_id, annahme_id)"
            " DO UPDATE SET haltung=excluded.haltung",
            (postit["id"], betroffen.annahme_id, betroffen.haltung, now()),
        )
        if betroffen.haltung == HALTUNG_EINWAND:
            await _pruefauftraege_anlegen(postit, betroffen.annahme_id)

    async def _postit_anlegen(
        *, werkfassung_id: str, stelle: str, text: str, zitat: str, origin: str,
        annahmen: list[Betroffen], herkunft: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        await _require("werkfassungen", werkfassung_id, "Werkfassung")
        for betroffen in annahmen:
            await _require("annahmen", betroffen.annahme_id, "Annahme")
        postit = {
            "id": new_id("pst"),
            "werkfassung_id": werkfassung_id,
            "stelle": stelle.strip(),
            "text": text.strip(),
            "zitat": zitat,
            "origin": origin,
            # Was die Maschine anlegt, beginnt als Vorschlag.
            "status": POSTIT_VORGESCHLAGEN if origin == ORIGIN_MASCHINE else POSTIT_OFFEN,
            "herkunft_art": None, "herkunft_id": None,
            "herkunft_sitzung": None, "herkunft_job": None,
            **(herkunft or {}),
            "created_at": now(),
        }
        await _schreibe(
            "INSERT INTO postits (id, werkfassung_id, stelle, text, zitat, origin, status,"
            " herkunft_art, herkunft_id, herkunft_sitzung, herkunft_job, created_at)"
            " VALUES (:id, :werkfassung_id, :stelle, :text, :zitat, :origin, :status,"
            " :herkunft_art, :herkunft_id, :herkunft_sitzung, :herkunft_job, :created_at)",
            postit,
        )
        for betroffen in annahmen:
            await _annahme_anhaengen(postit, betroffen)
        return await _postit_mit_bezuegen(postit["id"])

    async def _postit_mit_bezuegen(postit_id: str) -> dict[str, Any]:
        postit = await _require("postits", postit_id, "Post-it")
        postit["annahmen"] = await _alle(
            "SELECT p.annahme_id, p.haltung, a.text FROM postit_annahmen p"
            " JOIN annahmen a ON a.id = p.annahme_id WHERE p.postit_id=?"
            " ORDER BY p.created_at",
            (postit_id,),
        )
        postit["weiterwirkungen"] = await _alle(
            "SELECT * FROM weiterwirkungen WHERE quelle_art='postit' AND quelle_id=?"
            " ORDER BY created_at",
            (postit_id,),
        )
        return postit

    @app.post("/api/postits", status_code=201)
    async def create_postit(body: PostitCreate) -> dict[str, Any]:
        herkunft = None
        if body.herkunft_job:
            job = await store.get_job(body.herkunft_job)
            if job is None:
                raise HTTPException(status_code=404, detail="Auftrag nicht gefunden.")
            herkunft = {"herkunft_art": "job", "herkunft_id": job["id"],
                        "herkunft_sitzung": job["session_id"], "herkunft_job": job["id"]}
        return await _postit_anlegen(
            werkfassung_id=body.werkfassung_id, stelle=body.stelle, text=body.text,
            zitat=body.zitat, origin=body.origin, annahmen=body.annahmen, herkunft=herkunft,
        )

    @app.post("/api/markers/{marker_id}/postit", status_code=201)
    async def postit_aus_marker(marker_id: str, body: PostitAusMarker) -> dict[str, Any]:
        """Macht aus einem Marker ein Post-it. Der Marker bleibt, wie er ist.

        Zitat und Notiz werden mitgenommen: Marker werden bei jeder neuen
        Einschätzung ersetzt und verschwinden mit ihrer Sitzung.
        """
        marker = await _eine("SELECT * FROM markers WHERE id=?", (marker_id,))
        if marker is None:
            raise HTTPException(status_code=404, detail="Marker nicht gefunden.")
        text = body.text.strip() or marker["note"].strip() or marker["quote"].strip()
        return await _postit_anlegen(
            werkfassung_id=body.werkfassung_id, stelle=body.stelle, text=text,
            zitat=marker["quote"],
            # Den Marker hat die Maschine gesetzt, das Post-it daraus macht ein Mensch.
            origin=ORIGIN_MENSCH, annahmen=body.annahmen,
            herkunft={"herkunft_art": "marker", "herkunft_id": marker["id"],
                      "herkunft_sitzung": marker["session_id"],
                      "herkunft_job": marker["job_id"]},
        )

    @app.get("/api/postits/{postit_id}")
    async def get_postit(postit_id: str) -> dict[str, Any]:
        return await _postit_mit_bezuegen(postit_id)

    @app.get("/api/werkfassungen/{werkfassung_id}/postits")
    async def list_postits(werkfassung_id: str) -> dict[str, Any]:
        await _require("werkfassungen", werkfassung_id, "Werkfassung")
        return {"postits": await _alle(
            "SELECT * FROM postits WHERE werkfassung_id=? ORDER BY created_at",
            (werkfassung_id,),
        )}

    @app.post("/api/postits/{postit_id}/status")
    async def set_postit_status(postit_id: str, body: PostitStatus) -> dict[str, Any]:
        await _require("postits", postit_id, "Post-it")
        await _schreibe("UPDATE postits SET status=? WHERE id=?", (body.status, postit_id))
        return await _postit_mit_bezuegen(postit_id)

    @app.post("/api/postits/{postit_id}/annahmen", status_code=201)
    async def postit_betrifft_annahme(postit_id: str, body: Betroffen) -> dict[str, Any]:
        """Bezieht ein Post-it auf eine Annahme. Ein Einwand löst Prüfaufträge aus."""
        postit = await _require("postits", postit_id, "Post-it")
        await _require("annahmen", body.annahme_id, "Annahme")
        await _annahme_anhaengen(postit, body)
        return await _postit_mit_bezuegen(postit_id)

    # --------------------------------------------------------- Weiterwirkungen

    @app.post("/api/weiterwirkungen", status_code=201)
    async def create_weiterwirkung(body: WeiterwirkungCreate) -> dict[str, Any]:
        await _require(GEGENSTAENDE[body.quelle_art], body.quelle_id, "Quelle")
        await _require("werkfassungen", body.ziel_werkfassung_id, "Werkfassung")
        annahme_id = body.quelle_id if body.quelle_art == "annahme" else body.annahme_id
        if annahme_id:
            await _require("annahmen", annahme_id, "Annahme")
        vorhanden = await _eine(
            "SELECT * FROM weiterwirkungen WHERE quelle_art=? AND quelle_id=?"
            " AND annahme_id IS ? AND ziel_werkfassung_id=?",
            (body.quelle_art, body.quelle_id, annahme_id, body.ziel_werkfassung_id),
        )
        if vorhanden:
            raise HTTPException(status_code=409, detail="Diesen Prüfauftrag gibt es schon.")
        auftrag = {
            "id": new_id("wwk"),
            "quelle_art": body.quelle_art,
            "quelle_id": body.quelle_id,
            "annahme_id": annahme_id,
            "ziel_werkfassung_id": body.ziel_werkfassung_id,
            "origin": body.origin,
            # Die Maschine kann nichts annehmen, nur vorschlagen.
            "status": (PRUEF_VORGESCHLAGEN if body.origin == ORIGIN_MASCHINE
                       else PRUEF_ANGENOMMEN),
            "created_at": now(),
        }
        await _schreibe(
            "INSERT INTO weiterwirkungen (id, quelle_art, quelle_id, annahme_id,"
            " ziel_werkfassung_id, origin, status, created_at) VALUES (:id, :quelle_art,"
            " :quelle_id, :annahme_id, :ziel_werkfassung_id, :origin, :status, :created_at)",
            auftrag,
        )
        return await _require("weiterwirkungen", auftrag["id"], "Prüfauftrag")

    @app.post("/api/weiterwirkungen/{auftrag_id}/status")
    async def set_weiterwirkung_status(
        auftrag_id: str, body: WeiterwirkungStatus
    ) -> dict[str, Any]:
        """Nimmt einen vorgeschlagenen Prüfauftrag an oder lehnt ihn ab."""
        auftrag = await _require("weiterwirkungen", auftrag_id, "Prüfauftrag")
        if auftrag["status"] == PRUEF_GEPRUEFT:
            raise HTTPException(status_code=409, detail="Dieser Prüfauftrag ist geprüft.")
        await _schreibe(
            "UPDATE weiterwirkungen SET status=? WHERE id=?", (body.status, auftrag_id)
        )
        return await _require("weiterwirkungen", auftrag_id, "Prüfauftrag")

    @app.post("/api/weiterwirkungen/{auftrag_id}/ergebnis")
    async def set_weiterwirkung_ergebnis(
        auftrag_id: str, body: WeiterwirkungErgebnis
    ) -> dict[str, Any]:
        """Hält das Ergebnis genau dieses Prüfauftrags fest."""
        auftrag = await _require("weiterwirkungen", auftrag_id, "Prüfauftrag")
        if auftrag["status"] == PRUEF_ABGELEHNT:
            raise HTTPException(status_code=409, detail="Dieser Prüfauftrag wurde abgelehnt.")
        if auftrag["status"] == PRUEF_GEPRUEFT:
            raise HTTPException(
                status_code=409,
                detail="Dieser Prüfauftrag hat schon ein Ergebnis. Neue Erfahrung: neues Post-it.",
            )
        if body.episode_id:
            await _require("episoden", body.episode_id, "Episode")
        await _schreibe(
            "UPDATE weiterwirkungen SET status=?, ergebnis=?, ergebnis_notiz=?,"
            " ergebnis_episode_id=?, geprueft_at=? WHERE id=?",
            (PRUEF_GEPRUEFT, body.ergebnis, body.notiz.strip(), body.episode_id, now(),
             auftrag_id),
        )
        return await _require("weiterwirkungen", auftrag_id, "Prüfauftrag")

    # ---------------------------------------------------------------- Episoden

    async def _require_offene_episode(episode_id: str) -> dict[str, Any]:
        episode = await _require("episoden", episode_id, "Episode")
        if episode["status"] == EPISODE_ABGESCHLOSSEN:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Diese Episode ist abgeschlossen und bleibt, wie sie ist. "
                    "Ein Nachtrag ist eine neue Episode, die auf sie verweist."
                ),
            )
        return episode

    async def _episode_ganz(episode_id: str) -> dict[str, Any]:
        """Die sechs Bestandteile. Erprobung und Weiterwirkung sind Verweise."""
        episode = await _require("episoden", episode_id, "Episode")
        erprobung = await _alle(
            "SELECT e.postit_id, e.rolle, p.text, p.status, p.werkfassung_id, p.stelle"
            " FROM episoden_postits e JOIN postits p ON p.id = e.postit_id"
            " WHERE e.episode_id=? ORDER BY e.created_at",
            (episode_id,),
        )
        entscheidungen = await _alle(
            "SELECT * FROM entscheidungen WHERE episode_id=? ORDER BY created_at",
            (episode_id,),
        )
        # Weiterwirkung wächst nach dem Abschluss weiter: das Werk bleibt offen.
        postit_ids = [e["postit_id"] for e in erprobung]
        weiterwirkung: list[dict[str, Any]] = []
        if postit_ids:
            platzhalter = ",".join("?" for _ in postit_ids)
            weiterwirkung = await _alle(
                f"SELECT * FROM weiterwirkungen WHERE quelle_art='postit'"
                f" AND quelle_id IN ({platzhalter}) ORDER BY created_at",
                tuple(postit_ids),
            )
        return {
            "id": episode["id"],
            "werk_id": episode["werk_id"],
            "titel": episode["titel"],
            "status": episode["status"],
            "ergaenzt_episode_id": episode["ergaenzt_episode_id"],
            "created_at": episode["created_at"],
            "closed_at": episode["closed_at"],
            "ausgangslage": {
                "text": episode["ausgangslage"],
                "werkfassung_id": episode["ausgang_werkfassung_id"],
                "anliegen_fassung_id": episode["ausgang_anliegen_fassung_id"],
                "auftraege": episode["ausgang_auftraege"],
            },
            "perspektiven": episode["perspektiven"],
            "bearbeitung": episode["bearbeitung"],
            "erprobung": erprobung,
            "entscheidung": entscheidungen,
            "weiterwirkung": weiterwirkung,
            "nachtraege": await _alle(
                "SELECT id, titel, status FROM episoden WHERE ergaenzt_episode_id=?"
                " ORDER BY created_at",
                (episode_id,),
            ),
        }

    @app.post("/api/episoden", status_code=201)
    async def create_episode(body: EpisodeCreate) -> dict[str, Any]:
        await _require("werke", body.werk_id, "Werk")
        if body.ausgang_werkfassung_id:
            await _require("werkfassungen", body.ausgang_werkfassung_id, "Werkfassung")
        if body.ausgang_anliegen_fassung_id:
            await _require("anliegen_fassungen", body.ausgang_anliegen_fassung_id,
                           "Anliegen-Fassung")
        if body.ergaenzt_episode_id:
            await _require("episoden", body.ergaenzt_episode_id, "Episode")
        episode_id = new_id("eps")
        await _schreibe(
            "INSERT INTO episoden (id, werk_id, titel, status, ausgangslage,"
            " ausgang_werkfassung_id, ausgang_anliegen_fassung_id, ausgang_auftraege,"
            " perspektiven, bearbeitung, ergaenzt_episode_id, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (episode_id, body.werk_id, body.titel.strip(), EPISODE_OFFEN,
             body.ausgangslage.strip(), body.ausgang_werkfassung_id,
             body.ausgang_anliegen_fassung_id,
             json.dumps(body.ausgang_auftraege, ensure_ascii=False),
             json.dumps([v.model_dump() for v in body.perspektiven], ensure_ascii=False),
             json.dumps([v.model_dump() for v in body.bearbeitung], ensure_ascii=False),
             body.ergaenzt_episode_id, now()),
        )
        return await _episode_ganz(episode_id)

    @app.get("/api/episoden/{episode_id}")
    async def get_episode(episode_id: str) -> dict[str, Any]:
        return await _episode_ganz(episode_id)

    @app.post("/api/episoden/{episode_id}")
    async def update_episode(episode_id: str, body: EpisodeUpdate) -> dict[str, Any]:
        await _require_offene_episode(episode_id)
        felder: dict[str, Any] = {}
        if body.titel is not None:
            felder["titel"] = body.titel.strip()
        if body.ausgangslage is not None:
            felder["ausgangslage"] = body.ausgangslage.strip()
        for name in ("perspektiven", "bearbeitung"):
            wert = getattr(body, name)
            if wert is not None:
                felder[name] = json.dumps([v.model_dump() for v in wert], ensure_ascii=False)
        if not felder:
            raise HTTPException(status_code=422, detail="Nichts zu ändern.")
        zuweisung = ", ".join(f"{k}=:{k}" for k in felder)
        await _schreibe(
            f"UPDATE episoden SET {zuweisung} WHERE id=:id", {**felder, "id": episode_id}
        )
        return await _episode_ganz(episode_id)

    @app.post("/api/episoden/{episode_id}/postits", status_code=201)
    async def episode_erprobung(episode_id: str, body: EpisodePostit) -> dict[str, Any]:
        """Nimmt ein Post-it als Beobachtung oder Gegenbeispiel in die Erprobung auf."""
        await _require_offene_episode(episode_id)
        await _require("postits", body.postit_id, "Post-it")
        await _schreibe(
            "INSERT OR REPLACE INTO episoden_postits (episode_id, postit_id, rolle,"
            " created_at) VALUES (?,?,?,?)",
            (episode_id, body.postit_id, body.rolle, now()),
        )
        return await _episode_ganz(episode_id)

    @app.post("/api/episoden/{episode_id}/entscheidungen", status_code=201)
    async def episode_entscheidung(
        episode_id: str, body: EntscheidungCreate
    ) -> dict[str, Any]:
        await _require_offene_episode(episode_id)
        await _require(GEGENSTAENDE[body.gegenstand_art], body.gegenstand_id, "Gegenstand")
        if body.neue_werkfassung_id:
            await _require("werkfassungen", body.neue_werkfassung_id, "Werkfassung")
        if body.neue_anliegen_fassung_id:
            await _require("anliegen_fassungen", body.neue_anliegen_fassung_id,
                           "Anliegen-Fassung")
        await _schreibe(
            "INSERT INTO entscheidungen (id, episode_id, art, gegenstand_art,"
            " gegenstand_id, begruendung, neue_werkfassung_id, neue_anliegen_fassung_id,"
            " created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (new_id("ent"), episode_id, body.art, body.gegenstand_art, body.gegenstand_id,
             body.begruendung.strip(), body.neue_werkfassung_id,
             body.neue_anliegen_fassung_id, now()),
        )
        return await _episode_ganz(episode_id)

    @app.post("/api/episoden/{episode_id}/abschluss")
    async def close_episode(episode_id: str) -> dict[str, Any]:
        """Schließt die Episode ab. Danach ist sie unveränderlich."""
        episode = await _require("episoden", episode_id, "Episode")
        if episode["status"] != EPISODE_ABGESCHLOSSEN:
            await _schreibe(
                "UPDATE episoden SET status=?, closed_at=? WHERE id=?",
                (EPISODE_ABGESCHLOSSEN, now(), episode_id),
            )
        return await _episode_ganz(episode_id)

    # ----------------------------------- Was hat diese Erfahrung verändert?

    @app.get("/api/postits/{postit_id}/veraenderung")
    async def veraenderung(postit_id: str) -> dict[str, Any]:
        """Der Weg von einer Beobachtung bis zur neuen Fassung des Anliegens.

        Post-it → betroffene Annahmen → alle Werkfassungen mit derselben
        Annahme (samt Prüfauftrag) → Episoden und Entscheidungen → spätere
        Versuche → Veränderung des Anliegens. Wo der Weg endet, steht es da.
        """
        postit = await _postit_mit_bezuegen(postit_id)
        fassung = await _require("werkfassungen", postit["werkfassung_id"], "Werkfassung")
        werk = await _require("werke", fassung["werk_id"], "Werk")
        herkunft = None
        if postit["herkunft_art"]:
            herkunft = {
                "art": postit["herkunft_art"],
                "id": postit["herkunft_id"],
                "sitzung": postit["herkunft_sitzung"],
                "job": postit["herkunft_job"],
                # Die Sitzung darf fehlen; das Post-it gilt trotzdem.
                "sitzung_vorhanden": (
                    await store.get_session(postit["herkunft_sitzung"] or "")
                ) is not None,
            }

        auftraege = {
            (w["annahme_id"], w["ziel_werkfassung_id"]): w for w in postit["weiterwirkungen"]
        }
        annahmen = []
        for bezug in postit["annahmen"]:
            fassungen: dict[str, dict[str, Any]] = {}
            for v in await _verwendungen(bezug["annahme_id"]):
                eintrag = fassungen.setdefault(v["werkfassung_id"], {
                    "werkfassung_id": v["werkfassung_id"],
                    "werk_id": v["werk_id"],
                    "werk_titel": v["werk_titel"],
                    "nummer": v["nummer"],
                    "stellen": [],
                    "ausgangspunkt": v["werkfassung_id"] == fassung["id"],
                    "pruefauftrag": auftraege.get((bezug["annahme_id"], v["werkfassung_id"])),
                })
                eintrag["stellen"].append(v["stelle"])
            annahmen.append({**bezug, "werkfassungen": list(fassungen.values())})

        annahme_ids = [a["annahme_id"] for a in annahmen]
        platzhalter = ",".join("?" for _ in annahme_ids) or "NULL"
        treffer = await _alle(
            "SELECT episode_id FROM episoden_postits WHERE postit_id=?"
            " UNION SELECT episode_id FROM entscheidungen"
            "   WHERE gegenstand_art='postit' AND gegenstand_id=?"
            f" UNION SELECT episode_id FROM entscheidungen"
            f"   WHERE gegenstand_art='annahme' AND gegenstand_id IN ({platzhalter})"
            " UNION SELECT ergebnis_episode_id FROM weiterwirkungen"
            "   WHERE quelle_art='postit' AND quelle_id=?"
            "   AND ergebnis_episode_id IS NOT NULL",
            (postit_id, postit_id, *annahme_ids, postit_id),
        )
        episoden = []
        for t in treffer:
            ganz = await _episode_ganz(t["episode_id"])
            episoden.append({
                "id": ganz["id"], "titel": ganz["titel"], "status": ganz["status"],
                "werk_id": ganz["werk_id"], "created_at": ganz["created_at"],
                "closed_at": ganz["closed_at"],
                "ergaenzt_episode_id": ganz["ergaenzt_episode_id"],
                "entscheidungen": ganz["entscheidung"],
            })
        episoden.sort(key=lambda e: e["created_at"])

        versuche = []
        anliegen = []
        for episode in episoden:
            for ent in episode["entscheidungen"]:
                if ent["neue_werkfassung_id"]:
                    neue = await _require("werkfassungen", ent["neue_werkfassung_id"],
                                          "Werkfassung")
                    versuche.append({"werkfassung": neue, "entscheidung_id": ent["id"],
                                     "episode_id": episode["id"]})
                if ent["neue_anliegen_fassung_id"]:
                    nachher = await _require("anliegen_fassungen",
                                             ent["neue_anliegen_fassung_id"],
                                             "Anliegen-Fassung")
                    vorher = await _eine(
                        "SELECT * FROM anliegen_fassungen WHERE anliegen_id=? AND nummer=?",
                        (nachher["anliegen_id"], nachher["nummer"] - 1),
                    )
                    anliegen.append({
                        "anliegen": await _require("anliegen", nachher["anliegen_id"],
                                                   "Anliegen"),
                        "vorher": vorher, "nachher": nachher,
                        "entscheidung_id": ent["id"], "episode_id": episode["id"],
                    })

        ziele = [f for a in annahmen for f in a["werkfassungen"] if not f["ausgangspunkt"]]
        ungeprueft = [f for f in ziele
                      if f["pruefauftrag"] and f["pruefauftrag"]["status"] != PRUEF_GEPRUEFT]
        luecken = []
        if not annahmen:
            luecken.append("Das Post-it ist noch auf keine Annahme bezogen.")
        elif not ziele:
            luecken.append("Keine andere Werkfassung nutzt eine der betroffenen Annahmen.")
        if ungeprueft:
            luecken.append(f"{len(ungeprueft)} Prüfaufträge sind noch nicht geprüft.")
        if not episoden:
            luecken.append("Keine Episode hat diese Beobachtung aufgenommen.")
        elif not any(e["entscheidungen"] for e in episoden):
            luecken.append("Zu dieser Beobachtung ist noch nichts entschieden.")
        if not anliegen:
            luecken.append("Das Anliegen hat sich durch diese Beobachtung bisher nicht verändert.")

        return {
            "frage": "Was hat diese Erfahrung verändert?",
            "postit": {k: postit[k] for k in
                       ("id", "text", "zitat", "stelle", "origin", "status", "created_at")},
            "ausgangspunkt": {
                "werk": werk,
                "werkfassung": fassung,
                "anliegen_fassung": await _eine(
                    "SELECT * FROM anliegen_fassungen WHERE id=?",
                    (fassung["anliegen_fassung_id"],),
                ),
                "herkunft": herkunft,
            },
            "annahmen": annahmen,
            "episoden": episoden,
            "spaetere_versuche": versuche,
            "anliegen": anliegen,
            "luecken": luecken,
            "hinweis": (
                "Prüfaufträge mit Herkunft 'maschine' sind Vorschläge, bis jemand sie "
                "geprüft hat. 'betroffen' steht nur dort, wo ein Ergebnis eingetragen ist."
            ),
        }
