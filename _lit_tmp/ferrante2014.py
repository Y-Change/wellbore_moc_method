# -*- coding: utf-8 -*-
import json, urllib.parse, urllib.request
mailto = "nature-skills@users.noreply.github.com"
params = {
    "search": "Leak size, detectability and test conditions in pressurized pipe systems Ferrante",
    "per_page": "5",
    "mailto": mailto,
    "select": "display_name,publication_year,cited_by_count,doi,authorships",
}
url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
req = urllib.request.Request(url, headers={"User-Agent": "mailto:" + mailto})
with urllib.request.urlopen(req, timeout=25) as r:
    payload = json.loads(r.read().decode())
for w in payload.get("results", []):
    doi = (w.get("doi") or "").replace("https://doi.org/", "")
    authors = ", ".join([(a.get("author") or {}).get("display_name", "") for a in (w.get("authorships") or [])[:4]])
    print(w.get("cited_by_count"), w.get("publication_year"), authors)
    print(" ", w.get("display_name"))
    print(" ", doi)
