# kubeastra-bench

Reproducible benchmark for **deterministic, minimal-diff GitOps remediation** —
the evaluation artifact for the paper *Don't Let the Model Write the YAML:
Deterministic, Minimal-Diff GitOps Remediation from LLM-Proposed Field Changes*
(preprint: [`paper-C1.pdf`](paper-C1.pdf)). The system under test (the span-edit
pipeline) lives in [astraverse-io/KubeAstra](https://github.com/astraverse-io/KubeAstra)
(`ui/backend/gitops`, Apache-2.0).

The benchmark asks a narrow question: given a *known* field change (which
resource, which field, which value), how faithfully does each strategy turn it
into bytes on disk? It compares a deterministic span-edit **SUT** against three
text-generation baselines — **B1** full-file rewrite, **B2** unified diff, **B3**
diff + validate-and-retry — across two models and five seeds, scored by an
automated parse-and-compare **oracle**.

## Layout

```
oracle.py            correctness judge (parse-and-compare, named-list aware)
appliers.py          diff appliers: strict / offset-tolerant / ws-insensitive / GNU patch
applier_study.py     re-score captured B2 diffs under every applier (Table 5)
refusal.py           G5 fail-closed adversarial stratum (Sec 6.8)
oracle_audit.py      independent manual audit of the oracle (Sec 7)
run.py               orchestrator: system x task x seed -> oracle -> metrics
batch.py             Anthropic Message Batches executor (50% off, async)
corpus_builder.py    task generator (round-trip value mutations)
systems/
  base.py            SystemOutput + backend locator
  sut.py             SUT adapter over KubeAstra's real span-edit pipeline
  baselines.py       B1/B2/B3
  prompts.py         engineered baseline prompts (+ one-shot example)
  llm.py             provider adapter (Anthropic / Gemini), retry, prompt caching
corpus/              synthetic seed manifests (unit tests)
corpus-real/         real Apache-2.0 manifests + provenance.json  (see NOTICE)
tasks-real.jsonl     83 field-change tasks over corpus-real
results/             the data behind the paper's tables (see below)
test_*.py            65 tests
```

## Reproduce the no-API results (free, deterministic)

No API keys needed. From the repo root:

```bash
pip install -r requirements.txt

# 1. test suite (65 tests)  — SUT/integration tests need a KubeAstra checkout
#    beside this repo (see below); they skip cleanly otherwise.
python -m pytest -q

# 2. applier study — re-scores the cached 415 B2 diffs (Table 5)
python applier_study.py --tasks tasks-real.jsonl --corpus corpus-real --out results/appliers

# 3. G5 fail-closed refusal stratum (Sec 6.8)
python refusal.py

# 4. oracle audit — stratified sample for manual adjudication (Sec 7)
python oracle_audit.py --tasks tasks-real.jsonl --corpus corpus-real \
    --diffs results/appliers/raw-diffs.jsonl
```

Expected (matches the paper):

- **Applier study (Table 5):** strict 0.027 applied/correct · offset-tolerant
  0.675 (0 misapplied) · GNU `patch --fuzz` 0.964 applied / 0.824 correct /
  **0.140 misapplied** · with `-l` (ignore whitespace) 0.202 misapplied.
- **Refusal (Sec 6.8):** precision 1.00, control coverage 1.00, recall 0.889,
  one documented leak (YAML aliases).
- **Oracle audit (Sec 7):** the n=30 stratified sample used in the paper is in
  `results/oracle-audit-n30.txt`; independent adjudication matched the oracle on
  all 30, including every misapplied case. (The SUT stratum needs a KubeAstra
  checkout beside this repo; the B2 strata do not.)

## The SUT dependency (KubeAstra)

The SUT imports the real pipeline (`gitops.index` / `gitops.locate` /
`gitops.edit`) from KubeAstra. `find_backend()` searches this file's ancestors
and their siblings, so the simplest setup is to clone KubeAstra **next to** this
repo:

```
some-dir/
  kubeastra-bench/     (this repo)
  KubeAstra/           git clone https://github.com/astraverse-io/KubeAstra
```

Then the SUT and integration tests find `KubeAstra/ui/backend/gitops`
automatically. Without it, SUT/integration tests skip; `oracle.py`,
`appliers.py`, `applier_study.py`, and `corpus_builder.py` need only PyYAML.

## Reproduce the full matrix (spends API credits)

```bash
export ANTHROPIC_API_KEY=...        # for claude-sonnet-5
export GOOGLE_API_KEY=...           # for gemini-3.7-flash
python run.py --systems SUT,B1,B2,B3 --seeds 5 --model claude-sonnet-5
python run.py --systems SUT,B1,B2,B3 --seeds 5 --model gemini-3.7-flash
```

`--batch` uses the Anthropic Message Batches API (async, 50% off). Results land
in `results/<name>/rows.jsonl` + `summary.json`. **Set `--max-tokens` above the
largest file** (default 16000): B1 must re-emit the whole file, and a low cap
silently truncates it (paper Sec 6.5).

## Results data

`results/` holds the scored data behind the paper's tables:

- `real-sonnet5-s5-clean/`, `real-flash-s5-clean/` — the clean runs
  (`max_tokens=16000`, 5 seeds, 415 runs/system) reported in Tables 1-4.
- `appliers/raw-diffs.jsonl` — the 415 captured Sonnet B2 diffs; scoring is free
  and reproducible (`applier_study.py` re-reads this rather than re-spending).

The earlier low-cap run that truncated B1 (paper Sec 6.5) is not included; the
`-clean` runs supersede it.

## Corpus

`corpus-real/` is drawn from license-verified, SHA-pinned, Apache-2.0 public
repositories (Google Cloud's *Online Boutique* and the Kustomize *helloWorld*
example); see `corpus-real/provenance.json` and [`NOTICE`](NOTICE). This is a
small, source-limited corpus (~80 of 83 tasks edit one multi-resource file); the
paper is explicit about that (Sec 6.2), and a larger multi-repo corpus is future
work.

## License

Apache-2.0 (see [`LICENSE`](LICENSE)), matching KubeAstra. Bundled third-party
manifests retain their upstream Apache-2.0 license; attribution is in
[`NOTICE`](NOTICE).
