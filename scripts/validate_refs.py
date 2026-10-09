#!/usr/bin/env python3
"""Validate references.bib against Crossref + arXiv (internet required; run locally/CI).
Match = title similarity >= 0.90 AND year within +-1 (books/old papers: year +-3) AND first-author surname found.
Writes audit/references_report.csv; with --fix rewrites note={status:...} and adds doi; --strict exits 1 if any non-verified."""
import re, sys, csv, json, argparse, difflib, urllib.parse, urllib.request, xml.etree.ElementTree as ET
ENTRY = re.compile(r"@(\w+)\{([^,]+),(.*?)\n\}", re.S)
def field(body, name):
    m = re.search(rf"{name}\s*=\s*\{{+(.*?)\}}+\s*,?\s*\n", body + "\n", re.S | re.I)
    return re.sub(r"[{}\\]", "", m.group(1)).strip() if m else ""
def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "hypatiax-ref-validator/1.0 (mailto:you@example.org)"})
    return urllib.request.urlopen(req, timeout=30).read()
def sim(a, b): return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
def crossref(title):
    d = json.loads(get("https://api.crossref.org/works?rows=3&query.bibliographic=" + urllib.parse.quote(title)))
    for it in d["message"]["items"]:
        t = (it.get("title") or [""])[0]
        y = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
        au = " ".join(a.get("family", "") for a in it.get("author", []))
        yield t, y, au, it.get("DOI", "")
def arxiv(title):
    x = ET.fromstring(get("http://export.arxiv.org/api/query?max_results=3&search_query=ti:" + urllib.parse.quote('"' + title + '"')))
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for e in x.findall("a:entry", ns):
        t = " ".join(e.find("a:title", ns).text.split()); y = int(e.find("a:published", ns).text[:4])
        au = " ".join(a.find("a:name", ns).text for a in e.findall("a:author", ns))
        yield t, y, au, "arXiv:" + e.find("a:id", ns).text.rsplit("/", 1)[-1]
def check(title, year, author):
    sur = author.split(" and ")[0].split(",")[0].strip().lower()
    tol = 3 if year < 1990 else 1
    best = (0, "", "")
    for src in (crossref, arxiv):
        try:
            for t, y, au, ident in src(title):
                s = sim(t, title)
                if s >= 0.90 and y and abs(y - year) <= tol and sur in au.lower() and s > best[0]:
                    best = (s, t, ident)
        except Exception as e:
            print("  lookup error:", e, file=sys.stderr)
        if best[0]: break
    return best
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("bib"); ap.add_argument("--strict", action="store_true")
    ap.add_argument("--fix", action="store_true"); ap.add_argument("--out", default="audit/references_report.csv")
    a = ap.parse_args(); txt = open(a.bib, encoding="utf8").read(); rows = []; new = txt
    for typ, key, body in ENTRY.findall(txt):
        t, y, au = field(body, "title"), int(field(body, "year") or 0), field(body, "author")
        s, mt, ident = check(t, y, au)
        st = "verified" if s else "unverified"
        rows.append([key, st, ident, mt, round(s, 3)]); print(f"{st:10} {key}")
        if a.fix:
            nb = re.sub(r"status:\w+", f"status:{st}", body)
            if ident and ident.startswith("10.") and "doi" not in nb.lower():
                nb = nb.replace("  note =", f"  doi = {{{ident}}},\n  note =")
            new = new.replace(body, nb)
    csv.writer(open(a.out, "w", newline="")).writerows([["bibkey", "status", "id", "matched_title", "score"]] + rows)
    if a.fix: open(a.bib, "w", encoding="utf8").write(new)
    bad = [r for r in rows if r[1] != "verified"]
    print(f"{len(rows)-len(bad)}/{len(rows)} verified"); sys.exit(1 if (a.strict and bad) else 0)
if __name__ == "__main__": main()
