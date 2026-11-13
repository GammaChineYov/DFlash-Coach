"""Record both deterministic passes over the reserved benchmark inputs."""

import hashlib
import json
import time


def measure_variant(server, label, rows, evidence_path):
    measurements = []
    server.api(
        "/completion",
        {"prompt": rows[0]["prompt"], "n_predict": 32, "temperature": 0, "cache_prompt": False},
    )
    for repeat in range(2):
        ordered = rows if repeat == 0 else list(reversed(rows))
        for row in ordered:
            begin = time.monotonic()
            result = server.api(
                "/completion",
                {
                    "prompt": row["prompt"],
                    "n_predict": 512,
                    "temperature": 0,
                    "top_k": 1,
                    "top_p": 1,
                    "min_p": 0,
                    "repeat_penalty": 1,
                    "seed": 731,
                    "cache_prompt": False,
                },
            )
            text = result.get("content", "")
            timings = result["timings"]
            assert timings.get("cache_n", 0) == 0
            measurement = {
                "variant": label,
                "id": row["id"],
                "repeat": repeat,
                "language": row["language"],
                "length_bucket": row["length_bucket"],
                "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "content": text,
                "timings": timings,
                "wall_seconds": time.monotonic() - begin,
            }
            measurements.append(measurement)
            with evidence_path.open("a") as stream:
                stream.write(json.dumps(measurement, ensure_ascii=False) + "\n")
    return measurements
