#!/usr/bin/env python
"""Download the RETECO v1.1 train/dev release (Track 1 by default) from Hugging Face.

    python scripts/download_data.py                    # all 13 Track 1 domains (~2 GB)
    python scripts/download_data.py --domains iota hsm # a subset, for smoke tests
    python scripts/download_data.py --track2           # also fetch Track 2
"""
import argparse

from huggingface_hub import snapshot_download

REPO = "DataScience-UIBK/RETECO-SemEval2027"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/reteco")
    ap.add_argument("--domains", nargs="+", default=None)
    ap.add_argument("--track2", action="store_true")
    ap.add_argument("--revision", default="main")
    a = ap.parse_args()
    patterns = ["split_manifest.json", "README.md"]
    patterns += [f"track1_tempo/{d}/*" for d in a.domains] if a.domains else ["track1_tempo/*"]
    if a.track2:
        patterns.append("track2_recor/*")
    snapshot_download(REPO, repo_type="dataset", local_dir=a.out, allow_patterns=patterns,
                      revision=a.revision)
    print("done ->", a.out)


if __name__ == "__main__":
    main()
