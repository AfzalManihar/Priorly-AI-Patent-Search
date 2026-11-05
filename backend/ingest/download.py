"""Download PatentsView bulk tables (PRD section 7).

  python -m ingest.download --years 2015 2016 2017 --out data/raw
  python -m ingest.download --list          # print the URLs without downloading

The base URL and file names are PatentsView's published bulk-download layout at the time of writing.
They change occasionally: check https://patentsview.org/download/data-download-tables and adjust
BASE / TABLES below if a download returns 404. This script has NOT been run against the live site
from the environment that generated this project, so treat it as a starting point.
"""
from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://s3.amazonaws.com/data.patentsview.org/download"
TABLES = ["g_patent", "g_cpc_current", "g_us_patent_citation", "g_assignee_disambiguated"]
PER_YEAR = ["g_claims_{year}", "g_detail_desc_text_{year}"]


def urls(years: list[int]) -> list[str]:
    names = list(TABLES) + [t.format(year=y) for y in years for t in PER_YEAR]
    return [f"{BASE}/{name}.tsv.zip" for name in names]


def fetch(url: str, out_dir: Path) -> None:
    target = out_dir / Path(url).name
    tsv = out_dir / Path(url).name.replace(".zip", "")
    if tsv.exists():
        print(f"skip {tsv.name} (already extracted)")
        return
    print(f"download {url}")
    with urllib.request.urlopen(url) as response, open(target, "wb") as handle:
        shutil.copyfileobj(response, handle)
    with zipfile.ZipFile(target) as archive:
        archive.extractall(out_dir)
    target.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, nargs="+", default=[2015, 2016, 2017])
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        print("\n".join(urls(args.years)))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    for url in urls(args.years):
        fetch(url, args.out)


if __name__ == "__main__":
    main()
