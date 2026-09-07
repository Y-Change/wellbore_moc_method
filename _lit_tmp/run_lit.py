# -*- coding: utf-8 -*-
"""OpenAlex + Semantic Scholar lookup for Paper A cepstrum metrics."""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request

OUT = r"E:\water_hammer_research\wellbore_moc_method\_lit_tmp"
os.makedirs(OUT, exist_ok=True)
MAILTO = "nature-skills@users.noreply.github.com"
UA = f"nature-academic-search/2.0 (mailto:{MAILTO})"


def get(url: str, timeout: int = 40) -> dict | list:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def openalex_search(q: str, n: int = 8) -> list[dict]:
    params = {
        "search": q,
        "per_page": str(n),
        "sort": "cited_by_count:desc",
        "mailto": MAILTO,
        "select": "id,doi,display_name,publication_year,cited_by_count,authorships,primary_location,abstract_inverted_index",
    }
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    data = get(url)
    rows = []
    for w in data.get("results", []):
        authors = []
        for a in (w.get("authorships") or [])[:6]:
            name = (a.get("author") or {}).get("display_name")
            if name:
                authors.append(name)
        loc = w.get("primary_location") or {}
        src = (loc.get("source") or {}) if loc else {}
        rows.append({
            "title": w.get("display_name"),
            "year": w.get("publication_year"),
            "cited_by_count": w.get("cited_by_count"),
            "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
            "authors": authors,
            "journal": src.get("display_name"),
            "openalex": w.get("id"),
        })
    return rows


def openalex_doi(doi: str) -> dict | None:
    url = "https://api.openalex.org/works/https://doi.org/" + urllib.parse.quote(doi) + f"?mailto={MAILTO}"
    try:
        w = get(url)
    except Exception as e:
        return {"doi": doi, "error": str(e)}
    authors = []
    for a in (w.get("authorships") or [])[:8]:
        name = (a.get("author") or {}).get("display_name")
        if name:
            authors.append(name)
    loc = w.get("primary_location") or {}
    src = (loc.get("source") or {}) if loc else {}
    return {
        "title": w.get("display_name"),
        "year": w.get("publication_year"),
        "cited_by_count": w.get("cited_by_count"),
        "doi": doi,
        "authors": authors,
        "journal": src.get("display_name"),
        "type": w.get("type"),
    }


def s2_doi(doi: str) -> dict | None:
    url = (
        "https://api.semanticscholar.org/graph/v1/paper/DOI:"
        + urllib.parse.quote(doi)
        + "?fields=title,year,citationCount,influentialCitationCount,venue,authors,externalIds"
    )
    try:
        w = get(url)
        return {
            "title": w.get("title"),
            "year": w.get("year"),
            "citationCount": w.get("citationCount"),
            "influentialCitationCount": w.get("influentialCitationCount"),
            "venue": w.get("venue"),
            "doi": doi,
        }
    except Exception as e:
        return {"doi": doi, "error": str(e)}


queries = [
    "cepstrum leak detection pipeline pressure transient Taghvaei",
    "inverse transient analysis Liggett Chen leak",
    "A review of transient based leak detection Colombo",
    "cepstrum analysis mechanical problems Randall",
    "leak size detectability Ferrante Brunone",
    "frequency response diagram leak location Lee Lambert",
    "The cepstrum a guide to processing Childers",
    "matched field processing pipeline leak Gong",
]

all_q = {}
for q in queries:
    try:
        all_q[q] = openalex_search(q, 8)
        print(f"OK search: {q} -> {len(all_q[q])}")
    except Exception as e:
        all_q[q] = [{"error": str(e)}]
        print(f"FAIL search: {q}: {e}")
    time.sleep(0.2)

dois = [
    "10.1088/0957-0233/17/2/018",  # Taghvaei 2006
    "10.1016/j.ymssp.2011.10.011",  # Ghazali 2012
    "10.1061/(ASCE)0733-9429(1994)120:8(934)",  # Liggett Chen
    "10.1080/00221686.2009.9521927",  # Colombo review? check
    "10.1016/j.jhydrol.2009.02.008",  # maybe wrong
    "10.1016/j.ymssp.2016.12.026",  # Randall history?
    "10.1109/PROC.1977.10747",  # Childers 1977
    "10.1016/S0022-1694(03)00223-5",  # Ferrante Brunone 2003? AWR
    "10.1016/S0309-1708(02)00089-6",  # Ferrante Brunone AWR harmonic
    "10.1007/s11269-014-0722-z",  # Ferrante 2014 detectability
    "10.1016/j.jsv.2004.08.021",  # Lee 2005 JSV FRD
    "10.1080/00221686.2010.507014",  # Duan 2010 essential info
    "10.1061/(ASCE)0733-9429(2005)131:8(715)",  # Vitkovsky?
    "10.1177/0954406217722805",  # Ghazali/Beck cepstrum of xcorr
    "10.5942/jawwa.2012.104.0108",  # Taghvaei onsite
    "10.1016/j.ymssp.2004.01.006",  # Randall 2004 real cepstrum gear
    "10.1109/TASSP.1977.1162974",  # maybe Oppenheim
    "10.1121/1.1909342",  # Noll 1967 cepstrum pitch JAES/JASA
    "10.1016/j.ymssp.2010.07.017",
    "10.1080/00221686.2006.9521718",  # Lee FR method
    "10.1016/j.jher.2010.12.003",
    "10.1016/j.advwatres.2002.11.001",
]

doi_rows = []
for d in dois:
    oa = openalex_doi(d)
    time.sleep(0.15)
    s2 = s2_doi(d)
    time.sleep(0.15)
    doi_rows.append({"doi": d, "openalex": oa, "s2": s2})
    cite = None
    if oa and not oa.get("error"):
        cite = oa.get("cited_by_count")
        title = oa.get("title")
    else:
        title = (s2 or {}).get("title")
        cite = (s2 or {}).get("citationCount")
    print(f"{cite!s:>6}  {d}  {str(title)[:80]}")

payload = {"queries": all_q, "dois": doi_rows}
out_path = os.path.join(OUT, "paperA_metric_lit.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
print("WROTE", out_path)
