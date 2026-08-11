"""Local dashboard to review generated engine datasets (stdlib only, no deps).

Scans runs/ for run dirs containing extraction/ and/or feynman/ with a
dataset.jsonl, and serves an interactive side-by-side viewer:
  python scripts/dashboard.py --port 7842 [--root runs]
Then open http://localhost:7842  (SSH: ssh -L 7842:localhost:7842 <host>).
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"


def classify(ex: dict) -> str:
    s = ex["messages"][0]["content"]
    if "translator" in s:
        return "gold-pair"
    if s == "Kalamang-English vocabulary.":
        return "vocab"
    if "grammar" in s:
        return "note"
    return "other"


def load_mode(run: str, mode: str) -> dict:
    d = RUNS / run / mode
    ds, meta = d / "dataset.jsonl", d / "meta.json"
    if not ds.exists():
        return {"exists": False, "examples": [], "meta": {}}
    exs = []
    for ln in ds.read_text().splitlines():
        if not ln.strip():
            continue
        ex = json.loads(ln)
        m = ex["messages"]
        exs.append({"type": classify(ex),
                    "prompt": m[1]["content"] if len(m) > 1 else "",
                    "answer": m[-1]["content"]})
    return {"exists": True, "examples": exs,
            "meta": json.loads(meta.read_text()) if meta.exists() else {}}


def list_runs() -> list[str]:
    out = []
    if RUNS.exists():
        for d in sorted(RUNS.iterdir()):
            if d.is_dir() and any((d / m / "dataset.jsonl").exists()
                                  for m in ("extraction", "feynman",
                                            "extraction_only", "feynman_core")):
                out.append(d.name)
    return out


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Engine Data Review</title>
<style>
:root{--bg:#0f1115;--card:#1a1d24;--fg:#e6e6e6;--mut:#9aa4b2;--line:#2a2f3a;
--pair:#3b82f6;--note:#ef4444;--vocab:#10b981}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg)}
header{padding:12px 18px;border-bottom:1px solid var(--line);display:flex;gap:14px;align-items:center;flex-wrap:wrap;position:sticky;top:0;background:var(--bg);z-index:5}
h1{font-size:16px;margin:0}select,input{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px 8px}
.stats{display:flex;gap:18px;padding:10px 18px;color:var(--mut);flex-wrap:wrap;border-bottom:1px solid var(--line)}
.stats b{color:var(--fg)}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:14px;padding:14px}
.col h2{font-size:13px;color:var(--mut);text-transform:uppercase;letter-spacing:.05em;margin:0 0 8px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin-bottom:8px}
.badge{display:inline-block;font-size:11px;padding:1px 7px;border-radius:20px;margin-bottom:6px;font-weight:600}
.gold-pair{background:rgba(59,130,246,.15);color:var(--pair)}
.note{background:rgba(239,68,68,.15);color:var(--note)}
.vocab{background:rgba(16,185,129,.15);color:var(--vocab)}
.q{color:var(--mut);font-size:12px;white-space:pre-wrap;margin-bottom:4px}
.a{white-space:pre-wrap}
.filters{display:flex;gap:8px;align-items:center}
label{color:var(--mut);font-size:12px}
.count{color:var(--mut);font-size:12px}
</style></head><body>
<header>
<h1>🔬 Engine Data Review</h1>
<label>run <select id=run></select></label>
<div class=filters>
<label><input type=checkbox class=tf value=gold-pair checked> gold-pair</label>
<label><input type=checkbox class=tf value=note checked> note</label>
<label><input type=checkbox class=tf value=vocab checked> vocab</label>
</div>
<input id=q placeholder="search text…" style="flex:1;min-width:160px">
</header>
<div class=stats id=stats></div>
<div class=cols>
<div class=col><h2 id=h_ext></h2><div id=ext></div></div>
<div class=col><h2 id=h_fey></h2><div id=fey></div></div>
</div>
<script>
let DATA={};
async function loadRuns(){let r=await fetch('/api/runs');let runs=await r.json();
 let s=document.getElementById('run');s.innerHTML=runs.map(x=>`<option>${x}</option>`).join('');
 s.onchange=loadRun;if(runs.length)loadRun();}
async function loadRun(){let run=document.getElementById('run').value;
 let r=await fetch('/api/data?run='+encodeURIComponent(run));DATA=await r.json();render();}
function types(){return [...document.querySelectorAll('.tf:checked')].map(c=>c.value);}
function fmtMeta(m){if(!m||!m.mode)return '';
 return `synth <b>${m.emitted_synthetic_tokens??m.emitted_tokens??'?'}</b> · gold-base <b>${m.gold_base_tokens??0}</b> · gen <b>${m.total_gen_completion_tokens??'?'}</b> · <b>${m.n_examples}</b> ex`;}
function col(side,d){let t=types();let q=document.getElementById('q').value.toLowerCase();
 let exs=(d.examples||[]).filter(e=>t.includes(e.type)&&(!q||(e.prompt+e.answer).toLowerCase().includes(q)));
 let counts={};(d.examples||[]).forEach(e=>counts[e.type]=(counts[e.type]||0)+1);
 document.getElementById('h_'+side).innerHTML=(side=='ext'?'EXTRACTION':'FEYNMAN')+
   ` <span class=count>(${exs.length} shown · `+Object.entries(counts).map(([k,v])=>`${v} ${k}`).join(' · ')+`)</span>`;
 document.getElementById(side=='ext'?'ext':'fey').innerHTML=exs.slice(0,400).map(e=>
   `<div class=card><span class="badge ${e.type}">${e.type}</span>
    <div class=q>${esc(e.prompt)}</div><div class=a>${esc(e.answer)}</div></div>`).join('');}
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function render(){let e=DATA.extraction||{},f=DATA.feynman||{};
 document.getElementById('stats').innerHTML=
   `<div>EXTRACTION: ${fmtMeta(e.meta)}</div><div>FEYNMAN: ${fmtMeta(f.meta)}</div>`;
 col('ext',e);col('fey',f);}
document.getElementById('q').oninput=render;
document.querySelectorAll('.tf').forEach(c=>c.onchange=render);
loadRuns();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            self._send(PAGE.encode(), "text/html; charset=utf-8")
        elif u.path == "/api/runs":
            self._send(json.dumps(list_runs()).encode(), "application/json")
        elif u.path == "/api/data":
            run = parse_qs(u.query).get("run", ["eyeball_mtob"])[0]
            def pick(*names):
                for n in names:
                    r = load_mode(run, n)
                    if r["exists"]:
                        return r
                return {"exists": False, "examples": [], "meta": {}}
            data = {"extraction": pick("extraction", "extraction_only"),
                    "feynman": pick("feynman", "feynman_core")}
            self._send(json.dumps(data).encode(), "application/json")
        else:
            self.send_error(404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7842)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), H)
    print(f"[dashboard] serving on http://localhost:{args.port}  (runs: {list_runs()})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
