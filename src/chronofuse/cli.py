"""Command-line interface.

    chronofuse build    --config configs/base.yaml --domains iota
    chronofuse run      --config configs/experiments/bm25.yaml --split dev --subtrack 1a --domains all
    chronofuse evaluate --run results/runs/<exp>/1a_dev --split dev --subtrack 1a
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from .config import load_config, resolve_path
from .data.reteco import TRACK1_DOMAINS, load_qrels
from .eval.metrics import iter_domains, per_query_ndcg, read_trec, recall_at, summarize, write_trec


def _domains(arg: list[str]) -> list[str]:
    return iter_domains(arg if arg != ["all"] else "all", TRACK1_DOMAINS)


def cmd_build(args) -> None:
    from .pipeline import build_domain

    cfg = load_config(args.config)
    for d in _domains(args.domains):
        t = time.time()
        build_domain(cfg, d, force=args.force)
        logging.info("[%s] built in %.1fs", d, time.time() - t)


def _evaluate(run_dir: Path, cfg, split: str, subtrack: str, domains: list[str]) -> dict:
    root = resolve_path(cfg.get("paths.data", "data/raw/reteco"))
    per_dom, recalls = {}, {}
    for d in domains:
        qrels = load_qrels(root, d, split, steps=(subtrack == "1b"))
        f = run_dir / f"{d}.trec"
        run = read_trec(f) if f.exists() else {}
        per_dom[d] = per_query_ndcg(run, qrels)
        recalls[d] = round(recall_at(run, qrels, 100), 4)
    res = summarize(per_dom)
    res["recall@100"] = recalls
    res["macro_recall@100"] = round(sum(recalls.values()) / len(recalls), 4) if recalls else 0.0
    return res


def cmd_run(args) -> None:
    from .pipeline import run_domain

    cfg = load_config(args.config)
    name = cfg.get("name", Path(args.config).stem)
    llm = None
    if cfg.get("query.analyzer") == "llm" or cfg.get("rerank.llm.enabled", False):
        from .llm.client import LLMClient

        llm = LLMClient(model=cfg.get("llm.model"), temperature=cfg.get("llm.temperature", 0.0))
    domains = _domains(args.domains)
    run_dir = resolve_path("results/runs") / name / f"{args.subtrack}_{args.split}"
    for d in domains:
        run = run_domain(cfg, d, args.split, args.subtrack, llm)
        write_trec(run, run_dir / f"{d}.trec", tag=name)
    res = _evaluate(run_dir, cfg, args.split, args.subtrack, domains)
    res.update({"experiment": name, "split": args.split, "subtrack": args.subtrack})
    out = resolve_path("results/metrics") / f"{name}__{args.subtrack}_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def cmd_evaluate(args) -> None:
    cfg = load_config(args.config)
    res = _evaluate(Path(args.run), cfg, args.split, args.subtrack, _domains(args.domains))
    print(json.dumps(res, indent=2))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="chronofuse")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build indexes / metadata / KG per domain")
    b.add_argument("--config", default="configs/base.yaml")
    b.add_argument("--domains", nargs="+", default=["all"])
    b.add_argument("--force", action="store_true")
    b.set_defaults(func=cmd_build)

    r = sub.add_parser("run", help="retrieve + write TREC runs + score")
    r.add_argument("--config", required=True)
    r.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    r.add_argument("--subtrack", default="1a", choices=["1a", "1b"])
    r.add_argument("--domains", nargs="+", default=["all"])
    r.set_defaults(func=cmd_run)

    e = sub.add_parser("evaluate", help="score an existing run directory")
    e.add_argument("--config", default="configs/base.yaml")
    e.add_argument("--run", required=True)
    e.add_argument("--split", default="dev")
    e.add_argument("--subtrack", default="1a", choices=["1a", "1b"])
    e.add_argument("--domains", nargs="+", default=["all"])
    e.set_defaults(func=cmd_evaluate)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
