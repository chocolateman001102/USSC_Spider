import json
import os
from collections import defaultdict


REPO_ROOT = "/Users/yangchen/Desktop/VAULTs/Working Repo/Tools & Guidelines/USSC_Spider"
CORPUS_DIR = os.path.join(REPO_ROOT, "supreme-corpus")
UTTERANCES_PATH = os.path.join(CORPUS_DIR, "utterances.jsonl")
CONVERSATIONS_PATH = os.path.join(CORPUS_DIR, "conversations.json")
SPEAKERS_PATH = os.path.join(CORPUS_DIR, "speakers.json")
BRIEF_DATA_DIR = os.path.join(REPO_ROOT, "data")  # Changed from "brief_data" to "data"



def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_speakers() -> dict:
    return load_json(SPEAKERS_PATH)


def build_docket_meta(conversations: dict) -> dict:
    # conversations.json is usually a dict of conversation_id -> {meta...}
    docket_meta = defaultdict(lambda: {
        "win_side": None,
        "votes_side": {},
        "advocates": {},
        "conversation_ids": [],
    })
    for conv_id, conv in conversations.items():
        meta = conv.get("meta") or conv
        case_id = meta.get("case_id") or conv.get("case_id")
        if not case_id or "_" not in case_id:
            continue
        docket = case_id.split("_", 1)[1]
        entry = docket_meta[docket]
        entry["conversation_ids"].append(conv_id)
        if meta.get("win_side") is not None:
            entry["win_side"] = meta.get("win_side")
        votes = meta.get("votes_side") or {}
        entry["votes_side"].update(votes)
        adv = meta.get("advocates") or {}
        # Merge advocates per conv
        for k, v in adv.items():
            entry["advocates"][k] = v
    return docket_meta


def build_docket_utterence(speakers: dict) -> dict:
    name_map = {sid: (info.get("name") or sid) for sid, info in speakers.items()}
    docket_lines = defaultdict(list)
    with open(UTTERANCES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                u = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta = u.get("meta") or {}
            case_id = meta.get("case_id") or u.get("case_id")
            if not case_id or "_" not in case_id:
                continue
            docket = case_id.split("_", 1)[1]
            speaker_id = u.get("speaker")
            speaker_name = name_map.get(speaker_id, speaker_id or "Unknown")
            text = (u.get("text") or "").strip()
            line_text = f"{speaker_name}: {text}" if text else f"{speaker_name}:"
            docket_lines[docket].append(line_text)
    return {d: "\n".join(lines) for d, lines in docket_lines.items()}


def target_json_path_for_docket(docket: str) -> str:
    return os.path.join(BRIEF_DATA_DIR, docket, "json", f"{docket}__corpus.json")


def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def main() -> None:
    conversations = load_json(CONVERSATIONS_PATH)
    speakers = load_speakers()
    docket_meta = build_docket_meta(conversations)
    docket_utter = build_docket_utterence(speakers)

    written = 0
    for docket_dir in os.listdir(BRIEF_DATA_DIR):
        docket = docket_dir
        json_dir = os.path.join(BRIEF_DATA_DIR, docket, "json")
        if not os.path.isdir(json_dir):
            continue

        out_path = target_json_path_for_docket(docket)
        meta = docket_meta.get(docket)
        utter = docket_utter.get(docket)
        # Only write if we have at least utterence or some meta
        if not meta and not utter:
            continue

        payload = {
            "docket": docket,
            "utterence": utter,  # keep requested key spelling
            "meta": meta,
            "source": {
                "corpus": "convokit supreme-corpus",
                "docs": "https://convokit.cornell.edu/documentation/supreme.html",
            },
        }
        ensure_parent_dir(out_path)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        written += 1

    print(f"Wrote {written} docket corpus JSON files")


if __name__ == "__main__":
    main()

