# -*- coding: utf-8 -*-
import json, os, urllib.request, urllib.parse, time

path = r"E:\water_hammer_research\wellbore_moc_method\_lit_tmp\paperA_metric_lit.json"
with open(path, encoding="utf-8") as f:
    data = json.load(f)

print("===== QUERY HITS =====")
for q, rows in data["queries"].items():
    print("\n##", q)
    for x in rows:
        if "error" in x:
            print(" ERR", x["error"])
            continue
        print(f"  {x.get('cited_by_count',0):5d} | {x.get('year')} | {', '.join(x.get('authors') or [])[:55]} | {str(x.get('title'))[:88]}")
        print(f"        {x.get('doi')}  [{x.get('journal')}]")

print("\n===== DOI LOOKUPS (OA / S2) =====")
for row in data["dois"]:
    oa = row.get("openalex") or {}
    s2 = row.get("s2") or {}
    title = oa.get("title") or s2.get("title")
    c_oa = oa.get("cited_by_count")
    c_s2 = s2.get("citationCount")
    print(f"  OA={c_oa!s:>5} S2={c_s2!s:>5} | {row['doi']}")
    print(f"       {str(title)[:100]}")

# extra DOI lookups
mailto = "nature-skills@users.noreply.github.com"
extra = [
    "10.1016/j.jher.2009.05.003",
    "10.1016/j.jher.2009.02.003",
    "10.1016/j.jsv.2005.03.023",
    "10.1016/S0309-1708(02)00090-2",
    "10.1016/S0309-1708(02)00089-6",
    "10.1016/j.ymssp.2021.107874",
    "10.1121/1.1910900",
    "10.1109/TAU.1967.1161897",
    "10.1061/(ASCE)0733-9496(2007)133:6(596)",
    "10.1016/j.jhydrol.2012.12.044",
    "10.1080/00221686.2011.553486",
    "10.1016/j.ymssp.2012.09.015",
    "10.1016/j.ymssp.2018.06.058",
    "10.1016/j.ymssp.2013.05.018",
    "10.1109/78.80769",
]

print("\n===== EXTRA DOI =====")
for d in extra:
    url = "https://api.openalex.org/works/https://doi.org/" + urllib.parse.quote(d) + "?mailto=" + mailto + "&select=display_name,publication_year,cited_by_count,authorships,primary_location"
    req = urllib.request.Request(url, headers={"User-Agent": "mailto:" + mailto})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            w = json.loads(r.read().decode())
        authors = ", ".join([(a.get("author") or {}).get("display_name", "") for a in (w.get("authorships") or [])[:4]])
        j = ((w.get("primary_location") or {}).get("source") or {}).get("display_name", "")
        print(f"  {w.get('cited_by_count',0):5d} | {w.get('publication_year')} | {authors[:60]}")
        print(f"        {w.get('display_name','')[:100]}")
        print(f"        {d} [{j}]")
    except Exception as e:
        print("  FAIL", d, type(e).__name__, e)
    time.sleep(0.12)

# targeted searches
print("\n===== TARGETED SEARCH =====")
qs = [
    "A selective literature review of transient-based leak detection methods",
    "Pipe system diagnosis and leak detection by unsteady-state tests Ferrante",
    "Cepstrum pitch determination Noll",
    "matched field processing water pipeline leak Gong Lambert",
    "leak detectability size Ferrante Meniconi",
]
for q in qs:
    params = {
        "search": q,
        "per_page": "5",
        "sort": "relevance_score:desc",
        "mailto": mailto,
        "select": "display_name,publication_year,cited_by_count,doi,authorships,primary_location",
    }
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "mailto:" + mailto})
    print("\nQ:", q)
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            payload = json.loads(r.read().decode())
        for w in payload.get("results", []):
            authors = ", ".join([(a.get("author") or {}).get("display_name", "") for a in (w.get("authorships") or [])[:3]])
            doi = (w.get("doi") or "").replace("https://doi.org/", "")
            print(f"  {w.get('cited_by_count',0):5d} | {w.get('publication_year')} | {authors[:55]} | {w.get('display_name','')[:80]}")
            print(f"        {doi}")
    except Exception as e:
        print(" FAIL", e)
    time.sleep(0.15)
