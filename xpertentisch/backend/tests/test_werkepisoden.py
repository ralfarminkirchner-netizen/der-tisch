"""Werkepisoden: Post-its ohne Sitzungsbindung, Prüfaufträge, Episoden, der Weg.

Zeigt, was der Prototyp zusagt:
- Ein Post-it überlebt das Löschen seiner Herkunftssitzung.
- Ein Einwand an einer Annahme erzeugt je anderer Werkfassung genau einen
  Prüfauftrag im Stand „vorgeschlagen“; keiner gilt ungeprüft als bestätigt.
- „Was hat diese Erfahrung verändert?“ führt bis zur neuen Anliegen-Fassung.
- Eine abgeschlossene Episode weist Änderungen ab.
"""

from __future__ import annotations

import aiosqlite
import pytest

from conftest import scenario, wait_for_jobs


async def lege_an(client, pfad: str, **daten) -> dict:
    resp = await client.post(pfad, json=daten)
    assert resp.status_code == 201, resp.text
    return resp.json()


async def werk_mit_fassung(client, titel: str, anliegen_fassung_id=None, **fassung) -> dict:
    werk = await lege_an(client, "/api/werke", titel=titel, art="app")
    return await lege_an(
        client, f"/api/werke/{werk['id']}/fassungen",
        anliegen_fassung_id=anliegen_fassung_id, **fassung,
    )


async def drei_werke_eine_annahme(client) -> dict:
    """Drei Werke, deren erste Fassung dieselbe Annahme nutzt; ein viertes nicht."""
    anliegen = await lege_an(
        client, "/api/anliegen", titel="Einstieg",
        text="Menschen sollen ohne Anleitung beginnen können.",
    )
    erste = anliegen["fassungen"][0]
    app_ = await werk_mit_fassung(client, "Die App", erste["id"],
                                  zusammensetzung=["Startbild", "Erste Aufgabe"])
    text = await werk_mit_fassung(client, "Der Begleittext", zusammensetzung=["Kapitel 1"])
    spiel = await werk_mit_fassung(client, "Die Spielmechanik", zusammensetzung=["Zug 1"])
    fremd = await werk_mit_fassung(client, "Unbeteiligtes Werk")
    annahme = await lege_an(client, "/api/annahmen",
                            text="Wer das Startbild sieht, weiß, was zu tun ist.")
    for fassung, stelle in ((app_, "Startbild"), (text, "Kapitel 1"), (spiel, "Zug 1")):
        await lege_an(client, f"/api/annahmen/{annahme['id']}/verwendungen",
                      werkfassung_id=fassung["id"], stelle=stelle)
    # Zweite Stelle in derselben Fassung: darf keinen zweiten Prüfauftrag erzeugen.
    await lege_an(client, f"/api/annahmen/{annahme['id']}/verwendungen",
                  werkfassung_id=text["id"], stelle="Kapitel 3")
    return {"anliegen": anliegen, "app": app_, "text": text, "spiel": spiel,
            "fremd": fremd, "annahme": annahme}


async def sitzung_mit_marker(client, session_id: str) -> dict:
    scenario("fake-a-model",
             text="Eine tägliche Sicherung der Datenbank ist notwendig und sinnvoll.")
    scenario("fake-b-model",
             text="Eine tägliche Sicherung der Datenbank ist nicht notwendig und nicht sinnvoll.")
    await client.post(
        f"/api/sessions/{session_id}/sparks",
        json={"prompt": "Täglich sichern?", "client_request_id": "req-werk"},
    )
    bundle = await wait_for_jobs(client, session_id)
    return bundle["sparks"][0]["markers"][0]


# ----------------------------------------------------- Post-it und Sitzung

@pytest.mark.asyncio
async def test_postit_ueberlebt_das_loeschen_seiner_herkunftssitzung(client, session_id):
    welt = await drei_werke_eine_annahme(client)
    marker = await sitzung_mit_marker(client, session_id)

    postit = await lege_an(
        client, f"/api/markers/{marker['id']}/postit",
        werkfassung_id=welt["app"]["id"], stelle="Startbild",
        text="Zwei Stimmen widersprechen sich genau an dieser Stelle.",
    )
    assert postit["zitat"] == marker["quote"]
    assert postit["herkunft_art"] == "marker"
    assert postit["herkunft_sitzung"] == session_id
    assert postit["status"] == "offen"

    store = client.app.state.store
    await store.conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
    await store.conn.commit()

    # Sitzung und Marker sind weg — das Löschen hat also wirklich durchgegriffen.
    assert (await client.get(f"/api/sessions/{session_id}")).status_code == 404
    assert await store.list_markers(session_id) == []

    danach = (await client.get(f"/api/postits/{postit['id']}")).json()
    assert danach["text"] == postit["text"]
    assert danach["zitat"] == marker["quote"]
    assert danach["werkfassung_id"] == welt["app"]["id"]

    weg = (await client.get(f"/api/postits/{postit['id']}/veraenderung")).json()
    assert weg["ausgangspunkt"]["herkunft"]["sitzung_vorhanden"] is False


@pytest.mark.asyncio
async def test_marker_und_beziehungen_bleiben_unveraendert(client, session_id):
    welt = await drei_werke_eine_annahme(client)
    marker = await sitzung_mit_marker(client, session_id)
    vorher = (await client.get(f"/api/sessions/{session_id}")).json()

    await lege_an(client, f"/api/markers/{marker['id']}/postit",
                  werkfassung_id=welt["app"]["id"])

    nachher = (await client.get(f"/api/sessions/{session_id}")).json()
    assert nachher["sparks"][0]["markers"] == vorher["sparks"][0]["markers"]
    assert nachher["relations"] == vorher["relations"]


@pytest.mark.asyncio
async def test_unbekannter_marker_ergibt_kein_postit(client):
    welt = await drei_werke_eine_annahme(client)
    resp = await client.post("/api/markers/mrk_gibtsnicht/postit",
                             json={"werkfassung_id": welt["app"]["id"]})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_maschinelles_postit_beginnt_als_vorschlag(client):
    welt = await drei_werke_eine_annahme(client)
    postit = await lege_an(client, "/api/postits", werkfassung_id=welt["app"]["id"],
                           text="Vermutung aus einer Auswertung.", origin="maschine")
    assert postit["status"] == "vorgeschlagen"


# ------------------------------------------------------------ Prüfaufträge

@pytest.mark.asyncio
async def test_einwand_erzeugt_je_anderer_werkfassung_genau_einen_pruefauftrag(client):
    welt = await drei_werke_eine_annahme(client)
    postit = await lege_an(
        client, "/api/postits", werkfassung_id=welt["app"]["id"], stelle="Startbild",
        text="Drei von vier Testpersonen wussten nicht, wo sie anfangen sollen.",
        annahmen=[{"annahme_id": welt["annahme"]["id"], "haltung": "einwand"}],
    )
    auftraege = postit["weiterwirkungen"]

    ziele = sorted(a["ziel_werkfassung_id"] for a in auftraege)
    assert ziele == sorted([welt["text"]["id"], welt["spiel"]["id"]])
    # Nicht die eigene Fassung, nicht das unbeteiligte Werk.
    assert welt["app"]["id"] not in ziele
    assert welt["fremd"]["id"] not in ziele
    for a in auftraege:
        assert a["origin"] == "maschine"
        assert a["status"] == "vorgeschlagen"
        assert a["ergebnis"] is None
        assert a["geprueft_at"] is None

    # Derselbe Einwand noch einmal: es bleibt bei einem Auftrag je Fassung.
    nochmal = await lege_an(client, f"/api/postits/{postit['id']}/annahmen",
                            annahme_id=welt["annahme"]["id"], haltung="einwand")
    assert len(nochmal["weiterwirkungen"]) == 2


@pytest.mark.asyncio
async def test_stuetze_und_frage_erzeugen_keine_pruefauftraege(client):
    welt = await drei_werke_eine_annahme(client)
    postit = await lege_an(
        client, "/api/postits", werkfassung_id=welt["app"]["id"],
        text="Hier hat es auf Anhieb geklappt.",
        annahmen=[{"annahme_id": welt["annahme"]["id"], "haltung": "stuetze"}],
    )
    assert postit["weiterwirkungen"] == []


@pytest.mark.asyncio
async def test_jeder_pruefauftrag_bekommt_sein_eigenes_ergebnis(client):
    welt = await drei_werke_eine_annahme(client)
    postit = await lege_an(
        client, "/api/postits", werkfassung_id=welt["app"]["id"], text="Einwand.",
        annahmen=[{"annahme_id": welt["annahme"]["id"]}],
    )
    nach_ziel = {a["ziel_werkfassung_id"]: a for a in postit["weiterwirkungen"]}
    fuer_text = nach_ziel[welt["text"]["id"]]
    fuer_spiel = nach_ziel[welt["spiel"]["id"]]

    resp = await client.post(f"/api/weiterwirkungen/{fuer_text['id']}/ergebnis",
                             json={"ergebnis": "betroffen", "notiz": "Kapitel 1 setzt es voraus."})
    assert resp.status_code == 200
    assert resp.json()["status"] == "geprueft"

    stand = (await client.get(f"/api/postits/{postit['id']}")).json()
    nach_ziel = {a["ziel_werkfassung_id"]: a for a in stand["weiterwirkungen"]}
    assert nach_ziel[welt["text"]["id"]]["ergebnis"] == "betroffen"
    # Der andere Auftrag bleibt, was er war.
    assert nach_ziel[welt["spiel"]["id"]]["status"] == "vorgeschlagen"
    assert nach_ziel[welt["spiel"]["id"]]["ergebnis"] is None

    # Ein Ergebnis wird nicht überschrieben.
    resp = await client.post(f"/api/weiterwirkungen/{fuer_text['id']}/ergebnis",
                             json={"ergebnis": "nicht_betroffen"})
    assert resp.status_code == 409

    # Abgelehnt heißt: kein Ergebnis.
    await client.post(f"/api/weiterwirkungen/{fuer_spiel['id']}/status",
                      json={"status": "abgelehnt"})
    resp = await client.post(f"/api/weiterwirkungen/{fuer_spiel['id']}/ergebnis",
                             json={"ergebnis": "betroffen"})
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_maschine_kann_einen_pruefauftrag_nur_vorschlagen(client):
    welt = await drei_werke_eine_annahme(client)
    von_maschine = await lege_an(
        client, "/api/weiterwirkungen", quelle_art="annahme",
        quelle_id=welt["annahme"]["id"], ziel_werkfassung_id=welt["fremd"]["id"],
        origin="maschine",
    )
    assert von_maschine["status"] == "vorgeschlagen"

    von_mensch = await lege_an(
        client, "/api/weiterwirkungen", quelle_art="annahme",
        quelle_id=welt["annahme"]["id"], ziel_werkfassung_id=welt["spiel"]["id"],
    )
    assert von_mensch["status"] == "angenommen"
    assert von_mensch["ergebnis"] is None


# ---------------------------------------------------------------- Episoden

async def abgeschlossene_episode(client, welt: dict, postit: dict) -> dict:
    """Eine Episode, die den Einwand aufnimmt, entscheidet und das Anliegen neu fasst."""
    werk_id = welt["app"]["werk_id"]
    episode = await lege_an(
        client, "/api/episoden", werk_id=werk_id, titel="Der erste Bildschirm",
        ausgangslage="Reicht das Startbild als Einstieg?",
        ausgang_werkfassung_id=welt["app"]["id"],
        ausgang_anliegen_fassung_id=welt["anliegen"]["fassungen"][0]["id"],
        perspektiven=[{"art": "annahme", "verweis": welt["annahme"]["id"]},
                      {"art": "linse", "verweis": "Erstkontakt", "notiz": "ohne Vorwissen"}],
        bearbeitung=[{"art": "werkfassung", "verweis": welt["app"]["id"]}],
    )
    await lege_an(client, f"/api/episoden/{episode['id']}/postits",
                  postit_id=postit["id"], rolle="gegenbeispiel")

    neue_fassung = await lege_an(
        client, f"/api/werke/{werk_id}/fassungen",
        zusammensetzung=["Startbild", "Geführter erster Schritt", "Erste Aufgabe"],
        invarianten=["Kein Konto vor dem ersten Erfolg"],
        bilanz={"geblieben": ["Startbild"], "hinzugekommen": ["Geführter erster Schritt"]},
    )
    neues_anliegen = await lege_an(
        client, f"/api/anliegen/{welt['anliegen']['id']}/fassungen",
        text="Menschen sollen mit einem geführten ersten Schritt beginnen können.",
    )
    await lege_an(
        client, f"/api/episoden/{episode['id']}/entscheidungen",
        art="verworfen", gegenstand_art="annahme", gegenstand_id=welt["annahme"]["id"],
        begruendung="Die Erprobung widerspricht.",
        neue_werkfassung_id=neue_fassung["id"],
        neue_anliegen_fassung_id=neues_anliegen["id"],
    )
    resp = await client.post(f"/api/episoden/{episode['id']}/abschluss")
    assert resp.status_code == 200
    return {"episode": resp.json(), "neue_fassung": neue_fassung,
            "neues_anliegen": neues_anliegen}


async def einwand(client, welt: dict) -> dict:
    return await lege_an(
        client, "/api/postits", werkfassung_id=welt["app"]["id"], stelle="Startbild",
        text="Drei von vier Testpersonen wussten nicht, wo sie anfangen sollen.",
        annahmen=[{"annahme_id": welt["annahme"]["id"], "haltung": "einwand"}],
    )


@pytest.mark.asyncio
async def test_episode_hat_sechs_bestandteile_und_verweist_statt_zu_kopieren(client):
    welt = await drei_werke_eine_annahme(client)
    postit = await einwand(client, welt)
    episode = (await abgeschlossene_episode(client, welt, postit))["episode"]

    for teil in ("ausgangslage", "perspektiven", "bearbeitung", "erprobung",
                 "entscheidung", "weiterwirkung"):
        assert teil in episode
    assert episode["ausgangslage"]["werkfassung_id"] == welt["app"]["id"]
    assert episode["erprobung"][0]["postit_id"] == postit["id"]
    assert episode["entscheidung"][0]["art"] == "verworfen"
    assert len(episode["weiterwirkung"]) == 2

    # Das Post-it wird später entkräftet: die Episode zeigt den heutigen Stand,
    # weil sie verweist und nichts kopiert hat.
    await client.post(f"/api/postits/{postit['id']}/status", json={"status": "entkraeftet"})
    spaeter = (await client.get(f"/api/episoden/{episode['id']}")).json()
    assert spaeter["erprobung"][0]["status"] == "entkraeftet"


@pytest.mark.asyncio
async def test_abgeschlossene_episode_weist_aenderungen_ab(client):
    welt = await drei_werke_eine_annahme(client)
    postit = await einwand(client, welt)
    episode = (await abgeschlossene_episode(client, welt, postit))["episode"]
    eid = episode["id"]
    assert episode["status"] == "abgeschlossen"

    weiteres = await lege_an(client, "/api/postits", werkfassung_id=welt["app"]["id"],
                             text="Nachträgliche Beobachtung.")
    versuche = [
        (f"/api/episoden/{eid}", {"titel": "Umbenannt"}),
        (f"/api/episoden/{eid}", {"ausgangslage": "Andere Frage."}),
        (f"/api/episoden/{eid}/postits", {"postit_id": weiteres["id"]}),
        (f"/api/episoden/{eid}/entscheidungen",
         {"art": "zurueckgenommen", "gegenstand_art": "annahme",
          "gegenstand_id": welt["annahme"]["id"]}),
    ]
    for pfad, daten in versuche:
        resp = await client.post(pfad, json=daten)
        assert resp.status_code == 409, (pfad, resp.text)
        assert "abgeschlossen" in resp.json()["detail"]

    unveraendert = (await client.get(f"/api/episoden/{eid}")).json()
    assert unveraendert["titel"] == episode["titel"]
    assert unveraendert["erprobung"] == episode["erprobung"]
    assert unveraendert["entscheidung"] == episode["entscheidung"]

    # Ergänzen geht — als neue Episode, die auf die alte verweist.
    nachtrag = await lege_an(
        client, "/api/episoden", werk_id=welt["app"]["werk_id"],
        titel="Nachtrag zum ersten Bildschirm", ergaenzt_episode_id=eid,
    )
    assert nachtrag["status"] == "offen"
    alt = (await client.get(f"/api/episoden/{eid}")).json()
    assert [n["id"] for n in alt["nachtraege"]] == [nachtrag["id"]]


@pytest.mark.asyncio
async def test_unveraenderlichkeit_gilt_auch_an_der_schnittstelle_vorbei(client):
    """Die Datenbank selbst weist ab, nicht nur der Endpunkt."""
    welt = await drei_werke_eine_annahme(client)
    postit = await einwand(client, welt)
    fertig = await abgeschlossene_episode(client, welt, postit)
    conn = client.app.state.store.conn

    for sql, werte in (
        ("UPDATE episoden SET titel='x' WHERE id=?", (fertig["episode"]["id"],)),
        ("DELETE FROM entscheidungen WHERE episode_id=?", (fertig["episode"]["id"],)),
        ("DELETE FROM episoden_postits WHERE episode_id=?", (fertig["episode"]["id"],)),
        ("UPDATE werkfassungen SET nummer=99 WHERE id=?", (fertig["neue_fassung"]["id"],)),
        ("UPDATE anliegen_fassungen SET text='x' WHERE id=?",
         (fertig["neues_anliegen"]["id"],)),
    ):
        with pytest.raises(aiosqlite.IntegrityError):
            await conn.execute(sql, werte)
        await conn.rollback()


# ----------------------------------- Was hat diese Erfahrung verändert?

@pytest.mark.asyncio
async def test_ansicht_liefert_den_weg_bis_zur_anliegen_fassung(client):
    welt = await drei_werke_eine_annahme(client)
    postit = await einwand(client, welt)
    fertig = await abgeschlossene_episode(client, welt, postit)

    resp = await client.get(f"/api/postits/{postit['id']}/veraenderung")
    assert resp.status_code == 200
    weg = resp.json()

    # 1. Die Beobachtung und ihr Ausgangspunkt.
    assert weg["postit"]["id"] == postit["id"]
    assert weg["ausgangspunkt"]["werkfassung"]["id"] == welt["app"]["id"]
    assert weg["ausgangspunkt"]["anliegen_fassung"]["nummer"] == 1

    # 2. Die betroffene Annahme.
    assert [a["annahme_id"] for a in weg["annahmen"]] == [welt["annahme"]["id"]]
    assert weg["annahmen"][0]["haltung"] == "einwand"

    # 3. Alle Werkfassungen mit derselben Annahme, jede mit ihrem Prüfauftrag.
    fassungen = {f["werkfassung_id"]: f for f in weg["annahmen"][0]["werkfassungen"]}
    assert set(fassungen) == {welt["app"]["id"], welt["text"]["id"], welt["spiel"]["id"]}
    assert fassungen[welt["app"]["id"]]["ausgangspunkt"] is True
    assert fassungen[welt["app"]["id"]]["pruefauftrag"] is None
    assert fassungen[welt["text"]["id"]]["stellen"] == ["Kapitel 1", "Kapitel 3"]
    for kennung in (welt["text"]["id"], welt["spiel"]["id"]):
        assert fassungen[kennung]["pruefauftrag"]["status"] == "vorgeschlagen"

    # 4. Episode und Entscheidung.
    assert [e["id"] for e in weg["episoden"]] == [fertig["episode"]["id"]]
    entscheidung = weg["episoden"][0]["entscheidungen"][0]
    assert entscheidung["art"] == "verworfen"
    assert entscheidung["gegenstand_id"] == welt["annahme"]["id"]

    # 5. Der spätere Versuch.
    assert [v["werkfassung"]["id"] for v in weg["spaetere_versuche"]] == [
        fertig["neue_fassung"]["id"]
    ]
    assert weg["spaetere_versuche"][0]["werkfassung"]["nummer"] == 2

    # 6. Die Veränderung des Anliegens: vorher und nachher, mit Anlass.
    assert len(weg["anliegen"]) == 1
    veraendert = weg["anliegen"][0]
    assert veraendert["vorher"]["id"] == welt["anliegen"]["fassungen"][0]["id"]
    assert veraendert["nachher"]["id"] == fertig["neues_anliegen"]["id"]
    assert veraendert["nachher"]["nummer"] == 2
    assert veraendert["entscheidung_id"] == entscheidung["id"]

    # Was noch offen ist, steht dabei — der Weg tut nicht fertiger, als er ist.
    assert weg["luecken"] == ["2 Prüfaufträge sind noch nicht geprüft."]


@pytest.mark.asyncio
async def test_ansicht_benennt_wo_der_weg_endet(client):
    welt = await drei_werke_eine_annahme(client)
    postit = await lege_an(client, "/api/postits", werkfassung_id=welt["fremd"]["id"],
                           text="Eine Beobachtung ohne Folgen.")
    weg = (await client.get(f"/api/postits/{postit['id']}/veraenderung")).json()

    assert weg["annahmen"] == []
    assert weg["episoden"] == []
    assert weg["anliegen"] == []
    assert any("keine Annahme" in l for l in weg["luecken"])
    assert any("Anliegen" in l for l in weg["luecken"])


@pytest.mark.asyncio
async def test_anliegen_fassungen_bleiben_alle_erhalten(client):
    welt = await drei_werke_eine_annahme(client)
    await lege_an(client, f"/api/anliegen/{welt['anliegen']['id']}/fassungen",
                  text="Zweite Fassung.")
    anliegen = (await client.get(f"/api/anliegen/{welt['anliegen']['id']}")).json()
    assert [f["nummer"] for f in anliegen["fassungen"]] == [1, 2]
    assert anliegen["fassungen"][0]["text"].startswith("Menschen sollen ohne Anleitung")
