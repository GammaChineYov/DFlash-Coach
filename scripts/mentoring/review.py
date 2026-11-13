"""Coordinate a mentor review from cached evidence to a provider decision."""

from scripts.mentoring.client import request_review
from scripts.mentoring.credentials import read_key as read_key
from scripts.mentoring.decisions import apply_decision as apply_decision
from scripts.mentoring.decisions import validate_decision as validate_decision
from scripts.mentoring.protocol import review_body
from scripts.mentoring.storage import ReviewStore


def review(cfg, run, key, evidence):
    store = ReviewStore(run, key)
    cached = store.cached_decision(evidence)
    if cached is not None:
        return cached

    ledger = store.load_usage()
    body = review_body(cfg, evidence)
    credential = read_key(cfg)

    return request_review(
        cfg=cfg,
        body=body,
        credential=credential,
        evidence=evidence,
        store=store,
        ledger=ledger,
    )
