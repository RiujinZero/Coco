"""Turn chat JSONL ({"text": "USER: ...\n\nASSISTANT: ..."} per line) into the organism's byte stream.

Each dialogue becomes its text followed by a delimiter line so the stream carries turn structure the
organism can address ("USER:" / "ASSISTANT:" are ordinary bytes to it). A held-out tail is written
separately for the stay-on-task test.

    python python/chat_stream.py C:/path/to/manifoldflow_chat_sources/smoltalk_everyday.jsonl data/smoltalk.txt --holdout 200
    python python/chat_stream.py C:/path/to/manifoldflow_chat_sources/lmsys_chat_1m.jsonl data/lmsys_128m.txt --max_bytes 134217728 --holdout 400
"""
import argparse
import json

DELIM = "\n<|endofdialogue|>\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--max_bytes", type=int, default=0)
    ap.add_argument("--holdout", type=int, default=200, help="dialogues written to <dst>.holdout.jsonl instead")
    ap.add_argument("--max_dialogue", type=int, default=8192, help="skip dialogues longer than this many bytes")
    a = ap.parse_args()
    n = 0
    written = 0
    held = []
    with open(a.src, encoding="utf-8") as f, open(a.dst, "wb") as out:
        for line in f:
            try:
                t = json.loads(line)["text"]
            except Exception:
                continue
            if not t.startswith("USER:") or "ASSISTANT:" not in t:
                continue
            b = (t.strip() + DELIM).encode("utf-8", "ignore")
            if len(b) > a.max_dialogue or not b.isascii():
                continue
            n += 1
            if a.holdout and n % 50 == 0 and len(held) < a.holdout:
                held.append(t.strip())
                continue
            out.write(b)
            written += len(b)
            if a.max_bytes and written >= a.max_bytes:
                break
    with open(a.dst + ".holdout.jsonl", "w", encoding="utf-8") as h:
        for t in held:
            h.write(json.dumps({"text": t}) + "\n")
    print(f"{a.dst}: {written} bytes, {n} dialogues seen, {len(held)} held out")


if __name__ == "__main__":
    main()
