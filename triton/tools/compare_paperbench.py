#!/usr/bin/env python3
"""Per-language comparison table of Sidon paper-bench metrics for several systems.

Each system is NAME=ASR_DIR,QUALITY_DIR[,REFERENCE_DIR] (directories holding
asr_details.jsonl / NISQA_results.csv + dnsmos.csv / reference_details.jsonl).
p-CER (CallFriend) / WER (OpenDialog) is pooled errors / reference units per
language, NISQA and DNSMOS (P.808) are track means, speaker similarity and VAD
accuracy are sample means. Prints a markdown table and writes JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

LANG_RE = re.compile(r"callfriend-([a-z]+)")
LANG_MAP = {"deu": "de", "fra": "fr", "jpn": "ja", "spa": "es", "zho": "zh"}


def language_of(sample_id: str, fallback: str = "en") -> str:
    m = LANG_RE.match(sample_id)
    return LANG_MAP.get(m.group(1), m.group(1)) if m else fallback


def jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def sample_id_from_path(path: str) -> str:
    name = Path(path).name
    return re.sub(r"\.spk[12]\.wav$|\.wav$", "", name)


def collect(asr: Path | None, quality: Path | None, reference: Path | None) -> dict:
    stats = defaultdict(lambda: defaultdict(list))
    if asr is not None:
        for r in jsonl(asr / "asr_details.jsonl"):
            lang = r.get("language") or language_of(r["id"])
            for key in (lang, "all"):
                stats[key]["errors"].append(r["errors"])
                stats[key]["units"].append(r["reference_units"])
    if quality is not None:
        for fname, col, key in (("NISQA_results.csv", "mos_pred", "nisqa"), ("dnsmos.csv", "P808_MOS", "dnsmos")):
            path = quality / fname
            if not path.exists():
                continue
            with path.open() as handle:
                for row in csv.DictReader(handle):
                    path_col = row.get("filepath_deg") or row.get("filename")
                    lang = language_of(sample_id_from_path(path_col))
                    for k in (lang, "all"):
                        stats[k][key].append(float(row[col]))
    if reference is not None and (reference / "reference_details.jsonl").exists():
        for r in jsonl(reference / "reference_details.jsonl"):
            lang = language_of(r["id"])
            for k in (lang, "all"):
                stats[k]["spk_sim"].append(r["speaker_similarity"])
                stats[k]["vad"].append(r["vad_accuracy"])
    out = {}
    for lang, s in stats.items():
        row = {}
        if s["units"]:
            row["error_rate_percent"] = 100.0 * sum(s["errors"]) / sum(s["units"])
            row["num_samples"] = len(s["units"])
        for k in ("nisqa", "dnsmos", "spk_sim", "vad"):
            if s[k]:
                row[k] = sum(s[k]) / len(s[k])
        out[lang] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--system", action="append", required=True, help="NAME=ASR_DIR,QUALITY_DIR[,REFERENCE_DIR]")
    ap.add_argument("--title", default="")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    systems = {}
    for spec in args.system:
        name, dirs = spec.split("=", 1)
        parts = [Path(p) if p else None for p in dirs.split(",")] + [None, None]
        systems[name] = collect(parts[0], parts[1], parts[2])
    langs = sorted({lang for s in systems.values() for lang in s if lang != "all"}) + ["all"]
    cols = [("error_rate_percent", "err%", "{:.2f}"), ("nisqa", "NISQA", "{:.3f}"),
            ("dnsmos", "DNSMOS", "{:.3f}"), ("spk_sim", "SpkSim", "{:.3f}"), ("vad", "VAD", "{:.3f}")]
    print(f"### {args.title}" if args.title else "")
    for key, label, fmt in cols:
        if not any(key in s.get(lang, {}) for s in systems.values() for lang in langs):
            continue
        print(f"\n{label}\n")
        print("| system | " + " | ".join(langs) + " |")
        print("|---" * (len(langs) + 1) + "|")
        for name, s in systems.items():
            vals = [fmt.format(s[lang][key]) if key in s.get(lang, {}) else "-" for lang in langs]
            print(f"| {name} | " + " | ".join(vals) + " |")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(systems, indent=1))


if __name__ == "__main__":
    main()
