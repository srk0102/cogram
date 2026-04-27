# Cogram benchmark — proves cogram > mem0 + plain graphiti

Same corpus, same questions, three memory systems. LLM-as-judge scoring.

## Install benchmark deps

```powershell
.venv\Scripts\python.exe -m pip install mem0ai
```

(Optional — only needed if you include `mem0` in the systems list.)

## Run

Default — uses `episodes/sample_*.md` as corpus and built-in questions:

```powershell
# from repo root, with stack already up (docker compose up -d)
.venv\Scripts\python.exe -m bench.bench
```

Custom corpus + questions:

```powershell
.venv\Scripts\python.exe -m bench.bench `
  --corpus episodes/sample_*.md episodes/d--graphiti*.md `
  --questions bench/questions.json `
  --systems graphiti cogram mem0
```

Skip mem0 if you don't have it installed:

```powershell
.venv\Scripts\python.exe -m bench.bench --systems graphiti cogram
```

## Cost

Per run with default 8 questions × 3 systems on the 4 sample episodes:

- Ingest into 3 systems: ~$0.30-0.60
- Retrieve + answer 8 × 3 = 24 calls: ~$0.05-0.10
- LLM-as-judge for 24 answers: ~$0.05-0.10
- **Total: ~$0.40-0.80 per run**

Bigger corpus and more questions scale linearly.

## Output

Two files in `bench/`:

- `results_<ts>.json` — full data (every answer, every score, every latency)
- `report_<ts>.md` — human-readable summary table + per-question breakdown

The summary table looks like:

```
| System    | Avg score | Avg latency | Total cost |
|-----------|-----------|-------------|------------|
| mem0_oss  | 4.2       | 320ms       | $0.06      |
| graphiti  | 5.8       | 280ms       | $0.18      |
| cogram    | 7.6       | 250ms       | $0.22      |
```

That's the proof, with numbers.

## What it actually measures

- **Quality (score 0-10)** — LLM-as-judge compares each system's answer against
  the user's "ideal" answer in `questions.json`. Higher = more specific to the
  user's stance, fewer generic hedges, no hallucination.
- **Latency (ms)** — wall-clock time from question → final answer.
- **Cost (USD)** — total OpenAI spend for that system across ingest + retrieve
  + answer + judge.

## Adding your own questions

Edit `bench/questions.json` — list of `{"q": "...", "ideal": "..."}`. The ideal
answer is what a sharp friend who knows you would say. The judge compares each
system's output against this.

## Caveats

- **mem0 OSS is vector-only.** Their Pro tier adds a graph layer; we test the
  free tier because that's what's apples-to-apples for self-hosters.
- **Token cost estimates** are approximate. mem0 doesn't expose per-call usage
  cleanly; we estimate based on input length × pricing. Real cost may vary ±20%.
- **First run is expensive** because all three systems do ingest from cold.
  Subsequent runs (with `--skip-ingest`, future feature) would be cache-warm.
- **Single-machine bench**, not load testing. Cost-per-query at 1M users is a
  separate calculation — see Mem0's pricing page for what they charge at scale.

## What you should see

If cogram's intent-annotation + active-subgraph design is doing real work:

- **cogram score > graphiti score** by 1.5-3 points on average
- **cogram score > mem0 score** by 3-5 points (mem0 has no graph reasoning)
- **cogram cost > graphiti cost > mem0 cost** (more layers = more API calls)
- **cogram latency similar to graphiti**, both noticeably faster than mem0 once
  Redis active-subgraph caching kicks in

If cogram doesn't beat the others on score, the intent layer isn't earning its
keep — that's a falsification result and worth knowing.
