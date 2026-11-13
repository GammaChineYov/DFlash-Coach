"""Request a mentor review, preserving its cache and unbounded retry policy."""

import json
import time
import urllib.error
import urllib.request

from scripts.coach_core.constants.mentor import TRANSIENT_HTTP_STATUSES
from scripts.mentoring.protocol import response_decision, review_request


def request_review(*, cfg, body, credential, evidence, store, ledger):
    attempt = 0

    while True:
        attempt += 1
        store.record_attempt(ledger)
        request = review_request(cfg, body, credential)

        try:
            with urllib.request.urlopen(
                request,
                timeout=cfg["timeout_seconds"],
            ) as response:
                result = json.load(response)

            usage = result.get("usage") or {}
            store.record_usage(ledger, usage)
            decision = response_decision(result)
            store.save_result(cfg, result, decision, evidence, usage)
            return decision

        except urllib.error.HTTPError as error:
            if error.code not in TRANSIENT_HTTP_STATUSES:
                raise RuntimeError(
                    f"DeepSeek HTTP {error.code}; "
                    "credentials/provider configuration need attention"
                ) from None
            retry_reason = "HTTP " + str(error.code)

        except (
            urllib.error.URLError,
            TimeoutError,
            KeyError,
            IndexError,
            ValueError,
        ):
            retry_reason = "Network timeout or incomplete/invalid mentor response"

        store.record_waiting(attempt, retry_reason)
        retry_delay = min(2 ** min(attempt, 6), 60)
        time.sleep(retry_delay)
