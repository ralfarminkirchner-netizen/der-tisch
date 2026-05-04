#!/usr/bin/env python3
"""
apply_tisch_fixes.py — Patched TiSCH App Fixes
================================================
Run this script FROM your der-tisch/der-tisch-backend directory:

    cd ~/Documents/der-tisch/der-tisch-backend
    python3 apply_tisch_fixes.py

It will fix two bugs across all affected TiSCH apps:

BUG 1 (Critical): Custom perspectives sent to /api/ask are silently ignored.
       Fix: Route to /api/ask-table when custom_perspectives or methods are present.

BUG 2: source_app not tagged in requests — sessions logged with wrong app name.
       Fix: Add source_app to each fetch body.

BUG 3 (api_server.py): QueryRequest model doesn't accept custom_perspectives/methods.
       Fix: Add optional custom_perspectives + methods + source_app to QueryRequest,
            and route /api/ask to /api/ask-table logic when they're present.
"""

import re
import sys
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────────────────

APPS = {
    "expertentisch.html":  "EXPERTiSEN-TiSCH",
    "familientisch.html":  "FAMiLiEN-TiSCH",
    "juristisch.html":     "JURiSTiSCH",
    "trainingstisch.html": "TRAiNiNGS-TiSCH",
    "literatentisch.html": "LiTERATUR-TiSCH",
    # integrationstisch already patched in der-tisch-main — kept here for safety
    "integrationstisch.html": "iNTEGRATiONS-TiSCH",
}

# ─── HTML fixes ──────────────────────────────────────────────────────────────

def fix_html_file(path: Path, source_app: str) -> tuple[bool, str]:
    """Apply endpoint routing fix to a single HTML file.
    Returns (changed: bool, summary: str).
    """
    original = path.read_text(encoding="utf-8")
    text = original
    changes = []

    # ── Fix 1: source_app already present? ──────────────────────────────────
    if f"source_app = '{source_app}'" in text or f'source_app: "{source_app}"' in text:
        src_already = True
    else:
        src_already = False

    # ── Fix 2: Find body construction + /api/ask fetch pattern ──────────────
    # Pattern: body object built with question/lang/stil... then fetch /api/ask
    # We look for the fetch line that uses /api/ask and is in a context where
    # custom_perspectives might be assigned to the body.

    # Pattern A: direct fetch with object literal containing custom_perspectives
    # e.g.: fetch(API + '/api/ask', { ... custom_perspectives: ... })
    pattern_literal = re.compile(
        r"(fetch\s*\(\s*(?:API\s*\+\s*)?['\"`]/api/ask['\"`])\s*,\s*(\{[^}]*custom_perspectives[^}]*\})",
        re.DOTALL
    )

    # Pattern B: body variable built separately, then fetch('/api/ask')
    # We'll use a simpler search: find "fetch(API + '/api/ask'" or "fetch(`${API}/api/ask`"
    # preceded within ~50 lines by custom_perspectives assignment

    # Strategy: replace bare '/api/ask' fetch with conditional endpoint
    # First add source_app to body if missing, then fix endpoint

    # Find all occurrences of /api/ask fetch (not /api/ask-table, not /api/ask-simple)
    # that are not already using the conditional endpoint
    fetch_pattern = re.compile(
        r"(fetch\s*\(\s*(?:API\s*\+\s*['\"]|`\$\{API\})['/]api/ask['\"`])",
        re.DOTALL
    )

    new_text = text

    # ── Add source_app to body where it's missing ────────────────────────────
    # Look for body construction: var body = { question: ...
    if not src_already:
        # Try: var body = { question: question, ... };
        body_pattern = re.compile(
            r"(var\s+body\s*=\s*\{[^\n]*(?:question|lang)[^\n]*\})\s*;"
        )
        def add_source_app(m):
            # Insert source_app after the body line
            return m.group(0) + f"\n  body.source_app = '{source_app}';"
        new_text, n = body_pattern.subn(add_source_app, new_text, count=1)
        if n:
            changes.append(f"+ Added body.source_app = '{source_app}'")

        # Also try ES6 style: const body = { question, ... }
        body_const_pattern = re.compile(
            r"((?:const|let)\s+body\s*=\s*\{[^}]*(?:question|lang)[^}]*\})\s*;"
        )
        new_text2, n2 = body_const_pattern.subn(
            lambda m: m.group(0) + f"\n  body.source_app = '{source_app}';",
            new_text, count=1
        )
        if n2:
            new_text = new_text2
            changes.append(f"+ Added body.source_app = '{source_app}' (ES6 style)")

    # ── Fix endpoint routing ─────────────────────────────────────────────────
    # Pattern: check if there's already a conditional endpoint near /api/ask calls
    if "apiEndpoint" in new_text or "var endpoint" in new_text:
        changes.append("~ Endpoint already conditional, skipping endpoint fix")
    else:
        # Replace: fetch(API + '/api/ask', with conditional
        # We look for the fetch line and add a let endpoint = ... before it

        # Match: await fetch(API + '/api/ask',  OR  fetch(`${API}/api/ask`,
        repl_pattern = re.compile(
            r"(\s*)(var|const|let)?\s*(res\s*=\s*)?await\s+"
            r"(fetch\s*\(\s*(?:API\s*\+\s*['\"]|`\$\{API\})['\"]?/api/ask['\"`])",
            re.DOTALL
        )

        def replace_fetch(m):
            indent = m.group(1) or "  "
            rest_before_fetch = m.group(0)
            # Determine variable name used
            varpart = (m.group(2) or "") + " " + (m.group(3) or "")
            return (
                f"{indent}  // Route: use /api/ask-table when custom perspectives or methods are active\n"
                f"{indent}  var _apiEndpoint = (typeof allCustom !== 'undefined' && allCustom.length > 0) || "
                f"(typeof methods !== 'undefined' && methods.length > 0) ? '/api/ask-table' : '/api/ask';\n"
                + rest_before_fetch.replace('/api/ask', "' + _apiEndpoint + '", 1)
                .replace("+ '/api/ask'", "+ _apiEndpoint", 1)
                .replace('`${API}/api/ask`', '`${API}${_apiEndpoint}`', 1)
            )

        # Simpler, more reliable approach: string-replace the specific fetch pattern
        # Look for the fetch call and prepend the conditional
        simple_patterns = [
            ("fetch(API + '/api/ask',", "_apiEndpoint"),
            ('fetch(API + "/api/ask",', "_apiEndpoint"),
            ("fetch(`${API}/api/ask`,", "_apiEndpoint"),
        ]

        for old_fetch, _ in simple_patterns:
            if old_fetch in new_text:
                # Add the endpoint variable before the fetch
                insert_var = (
                    "  // Route to /api/ask-table when custom perspectives or methods are present\n"
                    "  var _apiEndpoint = ((typeof allCustom !== 'undefined' && allCustom.length > 0) || "
                    "(typeof methods !== 'undefined' && methods.length > 0)) ? '/api/ask-table' : '/api/ask';\n  "
                )
                new_text = new_text.replace(
                    old_fetch,
                    insert_var + old_fetch.replace('/api/ask', "' + _apiEndpoint + '")
                    if "API +" in old_fetch
                    else insert_var + "fetch(`${API}${_apiEndpoint}`,",
                    1
                )
                changes.append(f"+ Conditional endpoint routing for {old_fetch}")
                break

    if new_text == original:
        return False, f"  No changes needed (or pattern not matched) in {path.name}"

    path.write_text(new_text, encoding="utf-8")
    return True, f"  ✓ Fixed {path.name}: " + "; ".join(changes)


# ─── api_server.py fixes ─────────────────────────────────────────────────────

API_SERVER_PATCH = '''
# ── PATCH: QueryRequest extended with optional custom_perspectives + source_app ──
# Applied by apply_tisch_fixes.py
'''

def fix_api_server(path: Path) -> tuple[bool, str]:
    """
    Patch api_server.py with three improvements:
    1. Add source_app, custom_perspectives, methods, reibungsintensitaet to QueryRequest
    2. Make /api/ask delegate to ask_the_custom_table when custom_perspectives/methods present
    3. Make source_app dynamic in all save_session calls
    """
    original = path.read_text(encoding="utf-8")
    text = original
    changes = []

    # ── Guard: check if already patched ──────────────────────────────────────
    if "# TISCH-PATCH-APPLIED" in text:
        return False, "  api_server.py already patched"

    # ── 1. Extend QueryRequest with optional fields ───────────────────────────
    # Find the end of QueryRequest class (last field before next class/function)
    qr_patch = (
        '    # ── TISCH-PATCH: extended fields ──────────────────────────────────\n'
        '    source_app: Optional[str] = None            # App identifier for Shared Core (e.g. "JURiSTiSCH")\n'
        '    custom_perspectives: List[CustomPerspective] = []  # Routes to /api/ask-table logic when present\n'
        '    methods: List[str] = []                     # Subset of methods to invoke\n'
        '    reibungsintensitaet: str = "standard"       # standard | eskaliert | maximal\n'
    )
    # Insert after the tone line in QueryRequest
    tone_in_qr = '    tone: str = ""  # "" | "achtsam" | "direkt"\n\nclass CustomPerspective'
    if tone_in_qr in text and "TISCH-PATCH" not in text:
        text = text.replace(
            tone_in_qr,
            '    tone: str = ""  # "" | "achtsam" | "direkt"\n' + qr_patch + '\nclass CustomPerspective',
            1
        )
        changes.append("+ Extended QueryRequest with source_app, custom_perspectives, methods, reibungsintensitaet")
    else:
        # Fallback: find QueryRequest another way
        qr_match = re.search(r'(class QueryRequest\(BaseModel\):(?:.*?\n){3,8})', text, re.DOTALL)
        if qr_match:
            end = qr_match.end()
            text = text[:end] + qr_patch + text[end:]
            changes.append("+ Extended QueryRequest (fallback method)")

    # ── 2. Add source_app to TableRequest if missing ──────────────────────────
    if "source_app" not in text.split("class TableRequest")[1][:500] if "class TableRequest" in text else True:
        text = text.replace(
            '    reibungsintensitaet: str = "standard"               # eskaliert = Antagonisten-Modus',
            '    reibungsintensitaet: str = "standard"               # eskaliert = Antagonisten-Modus\n'
            '    source_app: Optional[str] = None                    # App identifier for Shared Core',
            1
        )
        changes.append("+ Added source_app to TableRequest")

    # ── 3. Delegate in /api/ask when custom_perspectives or methods present ───
    delegation_code = (
        '\n    # TISCH-PATCH: delegate to /api/ask-table when custom perspectives or methods are active\n'
        '    if req.custom_perspectives or req.methods:\n'
        '        table_req = TableRequest(\n'
        '            question=req.question, lang=req.lang, stil=req.stil,\n'
        '            register=req.register, tone=req.tone,\n'
        '            custom_perspectives=req.custom_perspectives,\n'
        '            methods=req.methods,\n'
        '            reibungsintensitaet=req.reibungsintensitaet,\n'
        '            source_app=req.source_app,\n'
        '        )\n'
        '        return await ask_the_custom_table(table_req)\n'
    )
    # Insert delegation after the "Question too short" check in /api/ask
    ask_check = 'async def ask_the_table(req: QueryRequest):\n    """Original-Endpunkt: immer alle 8 Methoden-Agenten."""\n    if not req.question or len(req.question.strip()) < 5:\n        raise HTTPException(status_code=400, detail="Question too short.")'
    if ask_check in text and "TISCH-PATCH: delegate" not in text:
        text = text.replace(ask_check, ask_check + delegation_code, 1)
        changes.append("+ Added delegation to ask_the_custom_table in /api/ask")
    else:
        # Softer fallback search
        m = re.search(
            r'(async def ask_the_table\(req: QueryRequest\):.*?raise HTTPException\(status_code=400, detail="Question too short\."\))',
            text, re.DOTALL
        )
        if m and "TISCH-PATCH: delegate" not in text:
            text = text[:m.end()] + delegation_code + text[m.end():]
            changes.append("+ Added delegation to ask_the_custom_table in /api/ask (fallback)")

    # ── 4. Make source_app dynamic in save_session calls ─────────────────────
    for default in ('"DER-TiSCH-FULL"', '"EiGENER-TiSCH"', '"DER-TiSCH"',
                    '"TEAM-TiSCH"', '"iNTEGRATiONS-TiSCH"'):
        old = f'source_app={default}'
        new = f'source_app=(req.source_app or {default})'
        if old in text:
            text = text.replace(old, new)
            changes.append(f"+ Made source_app dynamic for default {default}")

    # ── 5. Mark as patched ────────────────────────────────────────────────────
    text = text.replace(
        '#!/usr/bin/env python3\n',
        '#!/usr/bin/env python3\n# TISCH-PATCH-APPLIED\n',
        1
    )

    if text == original or not changes:
        return False, "  No changes applied to api_server.py (patterns not matched — check manually)"

    path.write_text(text, encoding="utf-8")
    return True, f"  ✓ Fixed api_server.py: " + "; ".join(changes)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    backend = Path(".")

    # Auto-detect backend dir
    if not (backend / "api_server.py").exists():
        backend = Path("der-tisch-backend")
    if not (backend / "api_server.py").exists():
        print("ERROR: Run this from der-tisch/der-tisch-backend or der-tisch/")
        sys.exit(1)

    print("=" * 60)
    print("TiSCH App Fixes — applying patches")
    print("=" * 60)

    total_changed = 0

    # Fix HTML files
    for filename, source_app in APPS.items():
        p = backend / filename
        if not p.exists():
            print(f"  SKIP {filename} (not found)")
            continue
        changed, msg = fix_html_file(p, source_app)
        print(msg)
        if changed:
            total_changed += 1

    # Fix api_server.py
    api_path = backend / "api_server.py"
    if api_path.exists():
        changed, msg = fix_api_server(api_path)
        print(msg)
        if changed:
            total_changed += 1
    else:
        print("  SKIP api_server.py (not found)")

    print("=" * 60)
    if total_changed:
        print(f"✓ {total_changed} files changed. Ready to commit:")
        print()
        print("  git add -A")
        print("  git commit -m 'fix: route custom perspectives to /api/ask-table, add source_app logging'")
        print("  git push")
    else:
        print("No changes were needed — everything already up to date.")


if __name__ == "__main__":
    main()
