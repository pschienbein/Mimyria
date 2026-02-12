import re


def _strip_comments(line: str) -> str:
    # CP2K uses '!' for comments; also strip '#' just in case
    line = line.split('!', 1)[0]
    line = line.split('#', 1)[0]
    return line.strip()


def _build_tree(text: str):
    """Very small CP2K section tree. Nodes hold children + raw lines."""
    root = {"name": "__ROOT__", "children": [], "lines": [], "parent": None}
    stack = [root]

    for raw in text.splitlines():
        line = _strip_comments(raw)
        if not line:
            continue

        if line.startswith("&"):
            body = line[1:].strip()
            upper = body.upper()
            if upper.startswith("END"):
                # &END or &END NAME
                parts = body.split(None, 1)
                end_name = parts[1].upper() if len(parts) > 1 else None
                if end_name:
                    while len(stack) > 1 and stack[-1]["name"] != end_name:
                        stack.pop()
                if len(stack) > 1:
                    stack.pop()
                continue

            # New section: first token is the section name (ignore trailing args)
            name = body.split()[0].upper()
            node = {"name": name, "children": [], "lines": [], "parent": stack[-1]}
            stack[-1]["children"].append(node)
            stack.append(node)
        else:
            stack[-1]["lines"].append(line)
    return root


def _find_nodes(root, path: str):
    parts = [p.strip().upper() for p in path.split("/") if p.strip()]
    curr = [root]
    for part in parts:
        nxt = []
        for node in curr:
            nxt.extend([c for c in node["children"] if c["name"] == part])
        curr = nxt
        if not curr:
            break
    return curr


def is_section_present(text: str, path: str) -> bool:
    """Return True if at least one section matching the path exists."""
    root = _build_tree(text)
    return bool(_find_nodes(root, path))


def check_variables_as_placeholders(text: str, path: str, expected: dict):
    """
    expected: mapping like {"INTENSITY":"FIELD_INTENSITY", "POLARISATION":"FIELD_POLARISATION"}.
    Returns (ok_overall, per_section_results).
    ok_overall is True if ANY matching section has all expected keys set exactly to the placeholders.
    """
    root = _build_tree(text)
    nodes = _find_nodes(root, path)
    exp = {k.upper(): v for k, v in expected.items()}
    results = []

    for n in nodes:
        found = {}
        for line in n["lines"]:
            for key in exp:
                m = re.match(rf"^\s*{re.escape(key)}\b\s*(.*)$", line, re.IGNORECASE)
                if m:
                    found[key] = m.group(1).strip()
        ok = True
        problems = {}
        for key, var in exp.items():
            val = found.get(key)
            want = f"${{{var}}}"
            if val is None:
                ok = False
                problems[key] = "missing"
            elif val != want:
                ok = False
                problems[key] = f"expected {want}, got {val}"
        results.append({"section": path, "ok": ok, "values": found, "problems": problems})

    return (any(r["ok"] for r in results) if nodes else False, results)


# (Optional) If you only care that they're *some* ${VAR}, not specific names:
def check_variables_are_any_placeholder(text: str, path: str, keys: list[str]):
    """
    Verifies each key exists and its value looks like ${SOMETHING}.
    Returns (ok_overall, per_section_results).
    """
    root = _build_tree(text)
    nodes = _find_nodes(root, path)
    pat = re.compile(r"^\$\{\w[\w\d_]*\}$")
    keysU = [k.upper() for k in keys]
    results = []
    for n in nodes:
        found, probs, ok = {}, {}, True
        for line in n["lines"]:
            for key in keysU:
                m = re.match(rf"^\s*{re.escape(key)}\b\s*(.*)$", line, re.IGNORECASE)
                if m:
                    found[key] = m.group(1).strip()
        for key in keysU:
            val = found.get(key)
            if val is None:
                ok, probs[key] = False, "missing"
            elif not pat.match(val):
                ok, probs[key] = False, f"not a ${'{VAR}'} placeholder (got {val})"
        results.append({"section": path, "ok": ok, "values": found, "problems": probs})
    return (any(r["ok"] for r in results) if nodes else False, results)


def check_keywords_equal(text: str, path: str, expected: dict, *,
                         case_sensitive: bool = True,
                         strip_quotes: bool = True):
    """
    Verify that, in ANY matching section at `path`, each keyword in `expected`
    is present and equals the given literal value.

    expected example: {"FILENAME": "dft-frc"}

    Returns (ok_overall, per_section_results) where ok_overall is True if at least
    one matching section satisfies all expectations.
    """
    root = _build_tree(text)
    nodes = _find_nodes(root, path)
    results = []

    for n in nodes:
        # Collect last value per keyword in this section
        kv = {}
        for line in n["lines"]:
            m = re.match(r"^\s*([A-Za-z0-9_]+)\b\s*(.*)$", line)
            if not m:
                continue
            key = m.group(1).upper()
            val = m.group(2).strip()
            kv[key] = val  # last-one-wins

        ok = True
        problems = {}
        for key, expected_val in expected.items():
            kU = key.upper()
            val = kv.get(kU)
            if val is None:
                ok = False
                problems[key] = "missing"
                continue

            # Optionally strip single/double quotes
            if strip_quotes and ((val.startswith('"') and val.endswith('"')) or
                                 (val.startswith("'") and val.endswith("'"))):
                val_cmp = val[1:-1]
            else:
                val_cmp = val

            if case_sensitive:
                good = (val_cmp == expected_val)
            else:
                good = (val_cmp.lower() == expected_val.lower())

            if not good:
                ok = False
                problems[key] = f"expected {expected_val!r}, got {val!r}"

        results.append({"section": path, "ok": ok, "found": kv, "problems": problems})

    return (any(r["ok"] for r in results) if nodes else False, results)


# Handy single-key helper:
def get_keyword_value(text: str, path: str, key: str):
    """
    Returns the last value for KEY in the first matching section at path,
    or None if not present.
    """
    root = _build_tree(text)
    nodes = _find_nodes(root, path)
    if not nodes:
        return None
    n = nodes[0]
    keyU = key.upper()
    val = None
    for line in n["lines"]:
        m = re.match(r"^\s*([A-Za-z0-9_]+)\b\s*(.*)$", line)
        if m and m.group(1).upper() == keyU:
            val = m.group(2).strip()
    return val


def check_section_guarded_by_if(text: str, path: str, if_condition: str, *,
                                endif_directive: str = "@ENDIF"):
    # Normalize expected directives
    if_line_expected = if_condition.strip()
    if not if_line_expected.upper().startswith("@IF"):
        if_line_expected = "@IF " + if_line_expected

    def _norm(s: str) -> str:
        # collapse whitespace + case-insensitive compare
        return re.sub(r"\s+", " ", s.strip()).upper()

    want_if = _norm(if_line_expected)
    want_endif = _norm(endif_directive)

    # Preprocess lines but keep original indices
    raw_lines = text.splitlines()
    stripped = [_strip_comments(l) for l in raw_lines]  # uses your existing helper

    def prev_nonempty(i: int):
        j = i - 1
        while j >= 0:
            if stripped[j]:
                return j
            j -= 1
        return None

    def next_nonempty(i: int):
        j = i + 1
        while j < len(stripped):
            if stripped[j]:
                return j
            j += 1
        return None

    # Path match state via section stack
    path_parts = [p.strip().upper() for p in path.split("/") if p.strip()]
    stack = []

    # Track occurrences: start/end indices for each matching section
    occurrences = []
    open_occ = None  # dict when inside a matching section

    for i, line in enumerate(stripped):
        if not line:
            continue

        if line.startswith("&"):
            body = line[1:].strip()
            upper = body.upper()

            if upper.startswith("END"):
                # &END or &END NAME
                parts = body.split(None, 1)
                end_name = parts[1].upper() if len(parts) > 1 else None

                # Pop stack appropriately (mirror your _build_tree behavior)
                if end_name:
                    while len(stack) > 0 and stack[-1] != end_name:
                        stack.pop()
                if stack:
                    popped = stack.pop()
                else:
                    popped = None

                # If we were in the matching section, close it when it ends
                if open_occ is not None:
                    # If &END had a name, require it matches the target leaf when possible
                    if (end_name is None) or (end_name == path_parts[-1]) or (popped == path_parts[-1]):
                        open_occ["end_idx"] = i
                        occurrences.append(open_occ)
                        open_occ = None
                continue

            # New section start
            name = body.split()[0].upper()
            stack.append(name)

            if stack == path_parts:
                open_occ = {"section": path, "start_idx": i, "end_idx": None}

    # If file ended while still open, record as unterminated
    if open_occ is not None:
        occurrences.append(open_occ)

    results = []
    for occ in occurrences:
        start_i = occ["start_idx"]
        end_i = occ["end_idx"]

        ok = True
        problems = {}

        if end_i is None:
            ok = False
            problems["section"] = "unterminated section (missing &END)"
            # Can't reliably check following @ENDIF
            endif_ok = False
            endif_found = None
        else:
            # Check @IF immediately before the section start (ignoring blanks/comments)
            pi = prev_nonempty(start_i)
            if_found = stripped[pi] if pi is not None else None
            if if_found is None or _norm(if_found) != want_if:
                ok = False
                problems["@IF"] = f"expected '{if_line_expected}' immediately before section"
            # Check @ENDIF immediately after the section end (ignoring blanks/comments)
            ni = next_nonempty(end_i)
            endif_found = stripped[ni] if ni is not None else None
            endif_ok = (endif_found is not None and _norm(endif_found) == want_endif)
            if not endif_ok:
                ok = False
                problems["@ENDIF"] = f"expected '{endif_directive}' immediately after section"

        results.append({
            "section": path,
            "ok": ok,
            "start_line": start_i + 1,  # 1-based for humans
            "end_line": (end_i + 1) if end_i is not None else None,
            "found_if": stripped[prev_nonempty(start_i)] if prev_nonempty(start_i) is not None else None,
            "found_endif": endif_found,
            "problems": problems
        })

    ok_overall = any(r["ok"] for r in results) if results else False
    if not results:
        results = [{
            "section": path,
            "ok": False,
            "start_line": None,
            "end_line": None,
            "found_if": None,
            "found_endif": None,
            "problems": {"section": "missing"}
        }]

    return ok_overall, results


