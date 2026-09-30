from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any


OWNER = "DanWBR"
REPO = "dwsim"
REF = "windows"  # DWSIM branch used earlier

OUTPUT_CSV = (
    Path(__file__).resolve().parents[2]
    / "data/source/activity_fitting/dwsim_uniquac_combinatorial_parameters.csv"
)

XML_FILES = [
    "DWSIM.Thermodynamics/Assets/Databases/dwsim.xml",
    "DWSIM.Thermodynamics/Assets/Databases/FoodProp.xml",
]

ADDCOMPS_DIR = "PlatformFiles/Common/addcomps"


@dataclass
class Row:
    source: str
    name: str
    cas: str
    formula: str
    dwsim_id: str
    UNIQUAC_r: str
    UNIQUAC_q: str


rows: list[Row] = []


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def github_request(url: str) -> bytes:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "dwsim-uniquac-fetcher",
    }

    # Optional, but helpful if you hit GitHub's unauthenticated rate limit.
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(request) as response:
        return response.read()


def raw_url(path: str) -> str:
    return f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{REF}/{path}"


def api_contents_url(path: str) -> str:
    return f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{path}?ref={REF}"


def fetch_raw_file(path: str) -> bytes:
    return github_request(raw_url(path))


def add_xml_file(path: str) -> None:
    try:
        data = fetch_raw_file(path)
    except Exception as exc:
        print(f"Skipping {path}: {exc}")
        return

    root = ET.fromstring(data)

    for comp in root.findall(".//component") + root.findall(".//compound"):
        name = clean(comp.findtext("Name"))
        r = clean(comp.findtext("UNIQUAC_r"))
        q = clean(comp.findtext("UNIQUAC_q"))

        if r or q:
            rows.append(
                Row(
                    source=path,
                    name=name,
                    cas=clean(comp.findtext("CAS_Number")),
                    formula=clean(comp.findtext("Formula")),
                    dwsim_id=clean(comp.findtext("ID")),
                    UNIQUAC_r=r,
                    UNIQUAC_q=q,
                )
            )


def find_case_insensitive(data: dict[str, Any], key: str) -> str:
    for k, v in data.items():
        if k.lower() == key.lower():
            return clean(v)
    return ""


def add_json_blob(source: str, blob: bytes) -> None:
    try:
        data = json.loads(blob.decode("utf-8-sig"))
    except Exception:
        return

    if not isinstance(data, dict):
        return

    r = (
        find_case_insensitive(data, "UNIQUAC_R")
        or find_case_insensitive(data, "UNIQUAC_r")
    )
    q = (
        find_case_insensitive(data, "UNIQUAC_Q")
        or find_case_insensitive(data, "UNIQUAC_q")
    )

    if r or q:
        rows.append(
            Row(
                source=source,
                name=clean(data.get("Name")) or source.rsplit("/", 1)[-1].removesuffix(".json"),
                cas=clean(data.get("CAS_Number")) or clean(data.get("CAS")),
                formula=clean(data.get("Formula")),
                dwsim_id=clean(data.get("ID")),
                UNIQUAC_r=r,
                UNIQUAC_q=q,
            )
        )


def add_addcomps_json_files() -> None:
    try:
        listing_raw = github_request(api_contents_url(ADDCOMPS_DIR))
    except Exception as exc:
        print(f"Skipping {ADDCOMPS_DIR}: {exc}")
        return

    listing = json.loads(listing_raw.decode("utf-8"))

    if not isinstance(listing, list):
        print(f"Skipping {ADDCOMPS_DIR}: GitHub API did not return a directory listing")
        return

    for item in listing:
        if not isinstance(item, dict):
            continue

        path = item.get("path", "")
        download_url = item.get("download_url", "")

        if not path.endswith(".json") or not download_url:
            continue

        try:
            blob = github_request(download_url)
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
            continue

        add_json_blob(path, blob)


def main() -> None:
    for path in XML_FILES:
        add_xml_file(path)

    add_addcomps_json_files()

    rows.sort(key=lambda row: (row.source, row.name))

    output_dir = os.path.dirname(OUTPUT_CSV)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source",
                "name",
                "cas",
                "formula",
                "dwsim_id",
                "UNIQUAC_r",
                "UNIQUAC_q",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)

    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
