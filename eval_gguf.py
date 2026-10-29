"""Exact-match accuracy of the GGUF via llama-server. Usage: python eval_gguf.py <server-bin> <gguf> [limit]"""
import json
import signal
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

server_bin, gguf = sys.argv[1], sys.argv[2]
limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
PORT, SLOTS = 8089, 8
URL = f"http://127.0.0.1:{PORT}"
http = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never route localhost via a proxy
signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))  # so `finally` stops the server

rows = [json.loads(line)["messages"] for line in open("data/eval.jsonl")][:limit]

server = subprocess.Popen([server_bin, "-m", gguf, "--port", str(PORT), "-np", str(SLOTS), "-c", str(SLOTS * 256),
                           "--reasoning", "off"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def post(messages):
    body = json.dumps({"messages": messages, "temperature": 0, "max_tokens": 16}).encode()
    req = urllib.request.Request(f"{URL}/v1/chat/completions", body, {"Content-Type": "application/json"})
    return json.load(http.open(req, timeout=120))["choices"][0]["message"]["content"].strip()


try:
    for _ in range(120):
        try:
            if http.open(f"{URL}/health").status == 200:
                break
        except OSError:
            pass
        time.sleep(1)
    else:
        sys.exit("llama-server did not become healthy")

    start = time.time()
    preds = []
    with ThreadPoolExecutor(SLOTS) as pool:
        for pred in pool.map(lambda m: post(m[:-1]), rows):
            preds.append(pred)
            if len(preds) % 200 == 0:
                print(f"{len(preds)}/{len(rows)}", flush=True)
    elapsed = time.time() - start
finally:
    server.terminate()

errors = Counter((m[-1]["content"], p) for m, p in zip(rows, preds) if p != m[-1]["content"])
correct = len(rows) - errors.total()
print(f"accuracy {correct / len(rows):.4f} ({correct}/{len(rows)}), {elapsed / len(rows) * 1000:.0f} ms/example")
for (gold, pred), n in errors.most_common(10):
    print(f"{n:4d}  {gold!r} -> {pred!r}")
