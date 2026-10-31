from __future__ import annotations

QUESTION_ID = "decision"
READOUT_PREFILL = "Best answer: ["


def state_text(rec: dict) -> str:
    return rec["state"]


def systemone_question(rec: dict) -> dict:
    q = {"type": rec["qtype"], "instructions": rec["instructions"]}
    if rec["qtype"] == "choice":
        q["criteria"] = dict(rec["criteria"])
    elif rec["qtype"] == "score":
        q["criteria"] = [rec["criteria"][o] for o in rec["options"]]
    return q


def jev_payload(rec: dict, model: str) -> dict:
    return {"model": model, "state": state_text(rec), "questions": {QUESTION_ID: systemone_question(rec)}}


def clef_record(rec: dict) -> dict:
    return {"state": state_text(rec), "questions": {QUESTION_ID: systemone_question(rec)}}


def pplx_row(rec: dict) -> dict:
    return {"state": state_text(rec), "question": systemone_question(rec)}


TRUE_IDS = {"true", "yes", "1"}
FALSE_IDS = {"false", "no", "0"}


def map_option_probs(rec: dict, keys: list, values: list[float]) -> dict[str, float]:
    opts = rec["options"]
    keys = [str(k) for k in keys]
    if set(keys) == set(opts):
        return dict(zip(keys, values))
    if rec["qtype"] == "score":
        if all(k.isdigit() for k in keys) and sorted(int(k) for k in keys) == list(range(len(opts))):
            return {opts[int(k)]: v for k, v in zip(keys, values)}
        texts = {rec["criteria"][o]: o for o in opts}
        if set(keys) == set(texts):
            return {texts[k]: v for k, v in zip(keys, values)}
    if rec["qtype"] == "noul":
        low = [k.lower() for k in keys]
        if len(low) == 2 and {low[0], low[1]} <= TRUE_IDS | FALSE_IDS and (set(low) & TRUE_IDS) and (set(low) & FALSE_IDS):
            p_true = next(v for k, v in zip(low, values) if k in TRUE_IDS)
            return {"yes": p_true, "no": 1.0 - p_true}
    raise ValueError(f"cannot map option keys {keys} to {opts} for qtype {rec['qtype']}")


def readout_user_prompt(rec: dict) -> str:
    lines = [f"[{i}] {rec['criteria'][o]}" for i, o in enumerate(rec["options"], start=1)]
    return (f"{state_text(rec)}\n\nQuestion: {rec['instructions']}\n\nOptions:\n" + "\n".join(lines) +
            "\n\nReply with the number of the single best option.")


def generative_user_prompt(rec: dict) -> str:
    lines = [f"- {o}: {rec['criteria'][o]}" for o in rec["options"]]
    return (f"{state_text(rec)}\n\nQuestion: {rec['instructions']}\n\n"
            "Options (use the option id before the colon):\n" + "\n".join(lines) + "\n\n"
            'Respond with JSON only: {"answer": "<option id>", "confidence": <integer 0-100, '
            "your probability that the answer is correct>}")


def generative_schema(rec: dict) -> dict:
    return {"type": "object",
            "properties": {"answer": {"type": "string", "enum": list(rec["options"])},
                           "confidence": {"type": "integer", "minimum": 0, "maximum": 100}},
            "required": ["answer", "confidence"], "additionalProperties": False}


def verbalized_probs(answer, confidence, options: list[str]) -> dict[str, float]:
    k = len(options)
    if answer not in options or confidence is None:
        return {o: 1.0 / k for o in options}
    c = min(max(float(confidence) / 100.0, 0.0), 1.0)
    rest = (1.0 - c) / (k - 1) if k > 1 else 0.0
    return {o: (c if o == answer else rest) for o in options}
