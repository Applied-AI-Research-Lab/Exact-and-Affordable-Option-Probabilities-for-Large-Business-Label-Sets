import math

import numpy as np
import pytest

from bp18 import metrics as M
from bp18 import stats as S
from bp18.prompts import verbalized_probs
from bp18.readout import score_options
from bp18.runners.api import parse_response


CIDS = {c: i for i, c in enumerate("0123456789]")}


def make_backend(target: dict[str, float]):
    inv = {v: k for k, v in CIDS.items()}

    def fn(batch):
        out = []
        for ids in batch:
            s = "".join(inv[i] for i in ids[1:])
            mass = {}
            for lab, p in target.items():
                full = lab + "]"
                if full.startswith(s) and len(full) > len(s):
                    ch = full[len(s)]
                    mass[ch] = mass.get(ch, 0.0) + p
            tot = sum(mass.values())
            out.append({CIDS[ch]: math.log(m / tot) for ch, m in mass.items()})
        return out

    return fn


@pytest.mark.parametrize("k", [3, 12, 77, 151])
def test_trie_recovers_distribution(k):
    rng = np.random.default_rng(k)
    w = rng.dirichlet(np.ones(k))
    target = {str(i + 1): float(w[i]) for i in range(k)}
    probs, diags = score_options([[999]], [k], make_backend(target), CIDS, prune=0.0)
    np.testing.assert_allclose(probs[0], w, atol=1e-9)
    assert abs(diags[0]["mass"] - 1.0) < 1e-9


def test_trie_pruning_keeps_mass():
    target = {"1": 0.9, "2": 0.0999999, "12": 1e-7}
    probs, _ = score_options([[999]], [12], make_backend(target), CIDS, prune=1e-5)
    assert probs[0][0] == pytest.approx(0.9 / 0.9999999, rel=1e-6)


def test_ece_perfect_and_overconfident():
    y = np.array([0, 1] * 50)
    P = np.eye(2)[y]
    assert M.ece_equal_mass(P, y) == pytest.approx(0.0)
    Pw = np.tile([0.99, 0.01], (100, 1))
    assert M.ece_equal_mass(Pw, y) == pytest.approx(0.49, abs=1e-9)


def test_brier_nll_accuracy():
    y = np.array([0, 1])
    P = np.array([[0.8, 0.2], [0.4, 0.6]])
    assert M.accuracy(P, y) == 1.0
    assert M.brier(P, y) == pytest.approx(((0.2**2 + 0.2**2) + (0.4**2 + 0.4**2)) / 2)
    assert M.nll(P, y) == pytest.approx(-(math.log(0.8) + math.log(0.6)) / 2)


def test_aurc_perfect_ranking():
    y = np.array([0, 0, 0, 1])
    P = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]])
    cov, risk = M.risk_coverage(P, y)
    assert risk[-1] == 0.25 and risk[0] == 0.0
    assert M.coverage_at_risk(P, y, 0.0) == 0.75


def test_holm_and_mcnemar():
    assert S.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    a = np.array([1] * 20 + [0] * 0, bool); b = np.array([0] * 20, bool)
    assert S.mcnemar_exact(a, b) < 1e-4
    assert S.mcnemar_exact(a, a) == 1.0


def test_tost():
    assert S.tost_equivalent(-0.01, 0.015, 0.02)
    assert not S.tost_equivalent(-0.03, 0.01, 0.02)


from bp18 import analyze as AN
from bp18.data import sweep
from bp18.scoring import first_token_probs, letters_prompt


def _clinc_rec(i, gold="g", k=151):
    opts = [f"l{j:03d}" for j in range(k - 1)] + ["oos"]
    if gold not in opts:
        opts[i % (k - 1)] = gold
    return {"id": f"clinc150-{i:05d}", "task": "clinc150", "gold": gold, "options": opts,
            "criteria": {o: o for o in opts}, "state": "x", "instructions": "Q", "meta": {"k": k}}


def test_sweep_subsets_are_nested_and_keep_gold_and_order():
    recs = [_clinc_rec(i) for i in range(30)]
    sets = {k: {r["id"].split("-")[1]: r for r in sweep(recs, k)} for k in (10, 25, 50, 100)}
    for idx in sets[10]:
        prev = None
        for k in (10, 25, 50, 100):
            r = sets[k][idx]
            assert len(r["options"]) == k and r["gold"] in r["options"]
            full_order = [o for o in recs[int(idx)]["options"] if o in r["options"]]
            assert r["options"] == full_order
            if prev is not None:
                assert set(prev) <= set(r["options"])
            prev = r["options"]


def test_first_token_shortcut_ties_and_bias():
    cids = {c: i for i, c in enumerate("0123456789]")}
    lp = {cids[c]: np.log(0.1) for c in "0123456789"}
    p = first_token_probs(lp, 15, cids)
    assert p[0] == pytest.approx(p[9]) == pytest.approx(p[14])
    exact_uniform = 1 / 15
    assert abs(p[0] - exact_uniform) < 1e-12
    lp[cids["1"]] = np.log(0.5)
    p = first_token_probs(lp, 15, cids)
    assert p[9] == pytest.approx(p[0])


def test_letters_prompt():
    r = {"state": "Customer message: hi", "instructions": "Which?", "options": ["a", "b", "c"],
         "criteria": {"a": "Alpha", "b": "Beta", "c": "Gamma"}}
    t = letters_prompt(r)
    assert "[A] Alpha" in t and "[C] Gamma" in t and "letter" in t


def test_parse_keys():
    assert AN.parse_key("qwen2b-cached") == ("qwen2b", "cached")
    assert AN.parse_key("qwen_readout") == ("qwen", "vtrie") and AN.parse_key("gemmaE2b_gen") == ("gemmaE2b", "gen")
    assert AN.parse_key("jev") == ("jev", "reference")


def test_exactness_and_comparisons_tables():
    rng = np.random.default_rng(0)
    P = rng.dirichlet(np.ones(12), 200)
    y = rng.integers(0, 12, 200)
    ids = [f"t-{i}" for i in range(200)]
    base = {"ids": ids, "P": P, "y": y, "k": 12, "sec": np.ones(200), "tok": np.ones(200)}
    data = {("m", "trie", "t"): base, ("m", "cached", "t"): {**base, "P": P + 1e-7},
            ("m", "first", "t"): {**base, "P": rng.dirichlet(np.ones(12), 200)}}
    ex = AN.exactness(data).iloc[0]
    assert ex.max_abs_diff < 1e-6 and ex.argmax_agree == 1.0
    cmp = AN.comparisons(data, 200)
    assert set(cmp.b) == {"first"} and len(cmp) == 1


def test_cached_equals_trie_on_a_real_model():
    torch = pytest.importorskip("torch")
    tr = pytest.importorskip("transformers")
    from bp18.readout import char_token_ids, score_options
    from bp18.student import batched_backend, cached_option_probs
    try:
        tok = tr.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        m = tr.AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32).eval()
    except Exception:
        pytest.skip("Qwen3-0.6B not available")
    cids = char_token_ids(tok)
    opts = "\n".join(f"[{i}] option {i}" for i in range(1, 24))
    text = tok.apply_chat_template([{"role": "user", "content": f"Pick one.\n{opts}"}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    prompt = tok.encode(text + "Best answer: [", add_special_tokens=False)
    pc, _ = cached_option_probs(m, prompt, 23, cids, "cpu", chunk=8)
    pt, _ = score_options([prompt], [23], batched_backend(m, list(cids.values()), "cpu", tok.pad_token_id or 0, 16),
                          cids, prune=0.0)
    np.testing.assert_allclose(pc, pt[0], atol=1e-4)
