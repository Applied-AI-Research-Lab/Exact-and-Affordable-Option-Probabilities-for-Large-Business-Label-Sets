from __future__ import annotations

import math
import string

import numpy as np

from .config import READOUT_PRUNE
from .prompts import READOUT_PREFILL, state_text
from .readout import score_options
from .student import batched_backend, cached_option_probs

LETTERS = string.ascii_uppercase


def letters_prompt(rec: dict) -> str:
    lines = [f"[{LETTERS[i]}] {rec['criteria'][o]}" for i, o in enumerate(rec["options"])]
    return (f"{state_text(rec)}\n\nQuestion: {rec['instructions']}\n\nOptions:\n" + "\n".join(lines) +
            "\n\nReply with the letter of the single best option.")


def first_token_probs(first_lp: dict[int, float], k: int, cids: dict) -> list[float]:
    s = np.array([math.exp(first_lp[cids[str(i)[0]]]) for i in range(1, k + 1)])
    return (s / s.sum()).tolist()


def letter_ids(tok) -> dict[str, int]:
    out = {}
    for c in LETTERS:
        ids = tok.encode(c, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"letter {c!r} is not a single token")
        out[c] = ids[0]
    return out


class Scorer:
    def __init__(self, model, tok, cids, device):
        self.model, self.tok, self.cids, self.device = model, tok, cids, device
        self.backend = batched_backend(model, list(cids.values()), device, tok.pad_token_id or 0, 16)
        self.lids = None

    def prompt(self, rec, letters=False) -> list[int]:
        from .student import truncate_text
        user = letters_prompt(rec) if letters else None
        if user is None:
            from .prompts import readout_user_prompt
            user = readout_user_prompt(truncate_text(rec, self.tok))
        text = self.tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
        return self.tok.encode(text + READOUT_PREFILL, add_special_tokens=False)

    def next_lp(self, ids):
        return self.backend([ids])[0]

    def score(self, rec, scorer: str):
        k = len(rec["options"])
        if scorer == "letters":
            if k > len(LETTERS):
                return None, {"skipped": "more than 26 options"}
            if self.lids is None:
                self.lids = letter_ids(self.tok)
            ids = self.prompt(rec, letters=True)
            import torch
            with torch.inference_mode():
                logits = self.model(input_ids=torch.tensor([ids], device=self.device)).logits[0, -1].float()
            lp = torch.log_softmax(logits, -1)
            s = np.exp([lp[self.lids[LETTERS[i]]].item() for i in range(k)])
            return (s / s.sum()).tolist(), {"forward_tokens": len(ids), "calls": 1}
        ids = self.prompt(rec)
        if scorer == "first":
            return first_token_probs(self.next_lp(ids), k, self.cids), {"forward_tokens": len(ids), "calls": 1}
        if scorer == "cached":
            p, d = cached_option_probs(self.model, ids, k, self.cids, self.device)
            n_cont = sum(len(str(i)) for i in range(1, k + 1))
            return p, {**d, "forward_tokens": len(ids) + n_cont}
        if scorer == "trie":
            seen = []

            def counting(batch):
                seen.extend(len(b) for b in batch)
                return self.backend(batch)
            p, d = score_options([ids], [k], counting, self.cids, prune=READOUT_PRUNE)
            return p[0], {**d[0], "forward_tokens": int(sum(seen))}
        raise ValueError(scorer)
