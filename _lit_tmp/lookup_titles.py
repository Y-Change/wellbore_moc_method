# -*- coding: utf-8 -*-
import json, urllib.parse, urllib.request, time

mailto = "nature-skills@users.noreply.github.com"
titles = [
    "Cepstrum pitch determination",
    "Pipe system diagnosis and leak detection by unsteady-state tests. 1. Harmonic analysis",
    "Leak size, detectability and test conditions in pressurized pipe systems",
    "Leak Detection in Pipelines using the Damping of Fluid Transients",
    "Frequency Domain Analysis for Detecting Pipeline Leaks",
    "Leak location using the pattern of the frequency response diagram in pipelines",
    "Pipeline Network Features and Leak Detection by Cross-Correlation Analysis of Reflected Waves",
    "Least squares deconvolution for leak detection with a pseudo random binary sequence",
    "Leak location in pipelines using the impulse response function",
    "The cepstrum: A guide to processing",
]
for t in titles:
    params = {
        "filter": "title.search:" + t,
        "per_page": "3",
        "mailto": mailto,
        "select": "display_name,publication_year,cited_by_count,doi,authorships,primary_location",
    }
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "mailto:" + mailto})
    print("\nT:", t[:70])
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            payload = json.loads(r.read().decode())
        for w in payload.get("results", []):
            authors = ", ".join([(a.get("author") or {}).get("display_name", "") for a in (w.get("authorships") or [])[:3]])
            doi = (w.get("doi") or "").replace("https://doi.org/", "")
            print(f"  {w.get('cited_by_count',0):5d} | {w.get('publication_year')} | {authors[:60]}")
            print(f"        {w.get('display_name')}")
            print(f"        {doi}")
    except Exception as e:
        print(" FAIL", e)
    time.sleep(0.12)
