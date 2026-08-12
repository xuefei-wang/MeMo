import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.ledger import Ledger
from common.llm import LLM, Endpoint
from data import rulearena
from eval.rulearena_eval import build_prompt, extract_total

led = Ledger("ra_probe")
llm = LLM(Endpoint("http://localhost:8002/v1", "student"), led)
N = 100
probs = rulearena.load_problems(0)[:N]
corpus = rulearena.load_corpus(include_tables=True)

def graded(context, tag):
    def one(p):
        out = llm.chat(build_prompt(p, context), purpose=tag,
                       temperature=0.0, max_tokens=4000, think=True)
        pred = extract_total(out)
        return pred, p.gold
    with ThreadPoolExecutor(max_workers=16) as ex:
        res = list(ex.map(one, probs))
    rel = [abs(pr-g)/max(g,1.0) for pr,g in res if pr is not None]
    exact = sum(pr is not None and abs(pr-g)<=1.0 for pr,g in res)/len(res)
    w20 = sum(e<=0.20 for e in rel)/len(res)
    w10 = sum(e<=0.10 for e in rel)/len(res)
    medrel = sorted(rel)[len(rel)//2] if rel else 1.0
    print(f"[{tag}] exact={exact:.2f} within10={w10:.2f} within20={w20:.2f} "
          f"median_rel_err={medrel:.3f} parsed={sum(pr is not None for pr,_ in res)}/{len(res)}", flush=True)

print(f"RuleArena L0 n={N}, Qwen3-8B, THINKING ON, max_tokens=4000", flush=True)
graded(None, "closed-book")
graded(corpus, "ICL-gold")
