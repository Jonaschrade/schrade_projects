#!/bin/bash
# The run tags on the SCC (RUNBOOK § Pulling models). Two modes:
#
#   pull   on scc2, no GPU:  bash ~/projects/thesis/simulation/pull_models.sh pull
#          Pulls the base tags, builds the run tags (draft off, one common sampler, no
#          SYSTEM), prints each one's parameters and Modelfile head, and captures them,
#          the tag list and the package list for simulation/environment/.
#   check  as a GPU job:     qsub -v MODE=check ~/projects/thesis/simulation/pull_models.sh
#          Loads each run tag on the GPU under the job's own server settings: no
#          speculative decoding, think=false accepted, no <think> in the text, the
#          prompt-token count of a bare prompt and whether the template renders each
#          system prompt, whether one seed repeats its text, whether a per-call min_p
#          is applied, what a prompt over num_ctx does, both letters in a
#          declaration's top logprobs, one declaration, turn and probe through
#          agents.agent, and the decode rate, cold and warm.
#
# Both print everything the review needs; send the output back as it is.

#$ -q gpu@scc214
#$ -l h_vmem=96G,rtx_6000=1,gpu=1
#$ -l h_rt=01:00:00
#$ -N pull_models_check
#$ -o /data/scc/jonas.schrade/thesis-runs/ -j y

set -euo pipefail

BASES="qwen3.5:4b gemma4:e4b granite4.2:3b"
# The tags runs use, one per base in the same order.
RUN_TAGS="qwen3.5-nodraft:4b gemma4-nodraft:e4b granite4.2-nodraft:3b"
# One sampler for every tag, so none inherits its base's: temperature 1 with min_p
# 0.05 as the only cutoff (Nguyen et al. 2025; llama.cpp's default), top_k and top_p
# off, repetition penalties neutral.
SAMPLER="temperature 1
min_p 0.05
top_k 0
top_p 1
repeat_penalty 1
presence_penalty 0
frequency_penalty 0"
CAPTURE=~/v2_env   # copied to simulation/environment/ after review
SIM=~/projects/thesis/simulation

MODE="${1:-${MODE:-}}"
case "$MODE" in pull|check) ;; *)
  echo "usage: bash $0 pull   |   qsub -v MODE=check $0"; exit 1 ;;
esac

# setup_thesis_env.sh appends to unset variables and dies under -u.
set +u
source ~/setup_thesis_env.sh
set -u
which ollama | grep -q '/data/scc' || { echo "ABORT: wrong ollama binary: $(which ollama)"; exit 1; }

wait_for_server() {
  for _ in $(seq 1 60); do
    curl -sf "http://$OLLAMA_HOST/api/version" > /dev/null && return 0
    sleep 1
  done
  echo "ABORT: no server at $OLLAMA_HOST, see $1"; exit 1
}

if [ "$MODE" = pull ]; then
  export OLLAMA_HOST=127.0.0.1:11990
  ollama serve > ~/v2_pull_serve.log 2>&1 &
  trap 'kill $! 2>/dev/null' EXIT
  wait_for_server ~/v2_pull_serve.log
  curl -s "http://$OLLAMA_HOST/api/version"; echo "   (must be 0.40.13)"

  for m in $BASES; do
    echo "=== pull $m"; ollama pull "$m"
  done
  # Draft off plus the common sampler: blobs shared, no SYSTEM added. The draft
  # parameter costs nothing where there is no draft.
  set -- $RUN_TAGS
  for b in $BASES; do
    { echo "FROM $b"; echo "PARAMETER draft_num_predict 0"
      sed 's/^/PARAMETER /' <<< "$SAMPLER"; } > ~/Modelfile.v2
    echo "=== create $1 from $b"; ollama create "$1" -f ~/Modelfile.v2
    shift
  done

  mkdir -p "$CAPTURE"
  for m in $BASES $RUN_TAGS; do
    echo "=== parameters $m"; ollama show "$m" --parameters
  done
  BAD=""
  for m in $RUN_TAGS; do
    f="$CAPTURE/modelfile_${m/:/_}.txt"
    ollama show "$m" --modelfile > "$f"
    # Everything above the licence: blobs, DRAFT, TEMPLATE, SYSTEM, PARAMETER. A
    # TEMPLATE that writes text when .System is empty is a default system prompt.
    echo "=== modelfile $m (to LICENSE)"; sed '/^LICENSE/q' "$f"
    echo "--- $m: DRAFT lines $(grep -c '^DRAFT' "$f" || true), SYSTEM lines $(grep -c '^SYSTEM' "$f" || true)"
    while read -r p; do
      grep -qx "PARAMETER $p" "$f" || BAD="$BAD $m:${p%% *}"
    done <<< "$SAMPLER"
    grep -qx 'PARAMETER draft_num_predict 0' "$f" || BAD="$BAD $m:draft"
    if grep -q '^SYSTEM' "$f"; then BAD="$BAD $m:SYSTEM"; fi
  done
  # The rest of environment/'s record, same commands as its § Capture.
  curl -s "http://$OLLAMA_HOST/api/tags" > "$CAPTURE/ollama_tags.json"
  { which python; python --version; pip freeze; ollama --version; } > "$CAPTURE/scc_env.txt" 2>&1
  echo "=== list"; ollama list
  echo "Captured to $CAPTURE:"; ls -l "$CAPTURE"
  [ -z "$BAD" ] || { echo "FAIL:$BAD"; exit 1; }
  echo "all run tags: common sampler, draft off, no SYSTEM"
  exit 0
fi

# ---- check: the job's own server, exactly as run_stage.sh starts it ----------
if [ -n "${SGE_GPU:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${SGE_GPU// /,}"
else
  # No starter method: a free GPU other than index 0 (the discovery trap).
  export CUDA_VISIBLE_DEVICES=$(nvidia-smi \
    --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '$1!=0 && $2<1000 {print $1; exit}')
  [ -n "$CUDA_VISIBLE_DEVICES" ] || { echo "ABORT: no free GPU other than 0"; exit 1; }
fi
export OLLAMA_HOST=127.0.0.1:$((11500 + JOB_ID % 400))
export LLAMA_ARG_THREADS=${NSLOTS:-1}
export OLLAMA_LOAD_TIMEOUT=30m
LOG="$HOME/ollama_$JOB_ID.log"
GGML_CUDA_NO_VMM=1 ollama serve > "$LOG" 2>&1 &
trap 'kill $! 2>/dev/null' EXIT
for _ in $(seq 1 60); do
  grep -q "inference compute" "$LOG" && break
  sleep 2
done
grep -qi "library=cuda" "$LOG" || { echo "ABORT: Ollama not on CUDA, see $LOG"; exit 1; }
wait_for_server "$LOG"
echo "node $(hostname), GPU $CUDA_VISIBLE_DEVICES, Ollama $(curl -s "http://$OLLAMA_HOST/api/version")"

cd "$SIM"
STATUS=0
python -u - $RUN_TAGS <<'EOF' || STATUS=1
"""Per tag: raw decode stats, the three Ollama checks, then one exchange's calls via agents."""
import sys
import time

import ollama

from agents import prompts
from agents.agent import Agent
from agents.llm import LLM, TOP_LOGPROBS
from config import LLM_NUM_CTX, OLLAMA_HOST

client = ollama.Client(host=OLLAMA_HOST)
BARE = "In two sentences, what is a river?"


def generate(tag: str, prompt: str = BARE, system: str = "", **options):
    """One raw call with think=false; options beside num_ctx."""
    return client.generate(model=tag, prompt=prompt, system=system, think=False, stream=False,
                           options={"num_ctx": LLM_NUM_CTX, **options})


failed = []
for i, tag in enumerate(sys.argv[1:]):
    print(f"\n=== {tag}")
    try:
        out = generate(tag)
    except ollama.ResponseError as e:
        print(f"FAIL raw generate with think=false: {e}")
        failed.append(tag)
        continue
    rate = out["eval_count"] / (out["eval_duration"] / 1e9)
    print(f"load {out['load_duration'] / 1e9:.1f}s, cold decode {rate:.1f} tok/s, "
          f"prompt {out['prompt_eval_count']} tokens, thinking field "
          f"{'present' if out.get('thinking') else 'absent'}")
    # A default system prompt shows as hundreds of prompt tokens on this bare prompt;
    # each system prompt must change the count, or the template drops the system slot.
    # A negative one is ours replacing a default; every call sends one.
    added = {k: generate(tag, system=s, num_predict=1)["prompt_eval_count"]
             - out["prompt_eval_count"]
             for k, s in (("turn", prompts.TURN_SYSTEM), ("answer", prompts.ANSWER_SYSTEM))}
    print("tokens added per system prompt: " + ", ".join(f"{k} {n:+d}" for k, n in added.items())
          + (" (replaces the template's default)" if max(added.values()) < 0 else ""))
    if 0 in added.values():
        print("FAIL: a system prompt leaves the prompt unchanged")
        failed.append(tag)
    # Reported, not gated: how far a run's seed reproduces its text on this node. The
    # first call after a load is cold, so the second's rate is the steady one.
    same = [generate(tag, seed=1) for _ in range(2)]
    print(f"same seed, same text: {'yes' if same[0]['response'] == same[1]['response'] else 'NO'}")
    print(f"warm decode {same[1]['eval_count'] / (same[1]['eval_duration'] / 1e9):.1f} tok/s")
    # A per-call min_p is applied iff 0.99 (near-greedy) makes two seeds agree that
    # min_p 0 lets differ.
    greedy = {generate(tag, seed=s, min_p=0.99)["response"] for s in (1, 2)}
    free = {generate(tag, seed=s, min_p=0.0)["response"] for s in (1, 2)}
    print(f"per-call min_p: 0.99 gives {len(greedy)} text(s) over two seeds, 0 gives {len(free)}")
    if len(greedy) != 1 or len(free) != 2:
        print("FAIL: a per-call min_p in options is not applied")
        failed.append(tag)
    # Reported, not gated: whether a prompt over num_ctx is cut or refused.
    try:
        r = client.generate(model=tag, prompt=" ".join(["river"] * 3000), think=False,
                            stream=False, options={"num_ctx": 1024, "num_predict": 1})
        print(f"prompt over num_ctx=1024: accepted, prompt_eval_count {r['prompt_eval_count']}"
              " (silently truncated if <= 1024)")
    except ollama.ResponseError as e:
        print(f"prompt over num_ctx=1024: refused ({e})")

    agent = Agent(f"Check{i}", LLM(model=tag))
    t0 = time.time()
    decl = agent.declare([], "full")
    t1 = time.time()
    turn = agent.speak(decl["x"] or 1, prompts.opener_message(prompts.draw_opener()[0]))
    t2 = time.time()
    print(f"declaration {t1 - t0:.1f}s: x={decl['x']}, logprobs {decl['logprobs']}")
    print(f"turn {t2 - t1:.1f}s")
    if len(decl["logprobs"]) < len(prompts.LETTERS):
        print(f"FAIL: {len(prompts.LETTERS) - len(decl['logprobs'])} letter(s) outside the top {TOP_LOGPROBS} logprobs")
        failed.append(tag)
    text = turn["text"]
    probe = agent.perceive("Mary", text)
    print(f"probe {time.time() - t2:.1f}s: x={probe['x']}, logprobs {probe['logprobs']}")
    print("turn text:", text.replace("\n", " ")[:400])
    if "<think>" in text:
        print("FAIL: <think> in the turn")
        failed.append(tag)
print(f"\n{'FAILED: ' + ' '.join(sorted(set(failed))) if failed else 'all tags passed'}")
sys.exit(1 if failed else 0)
EOF

# Must print no --spec-type / --model-draft for any load.
echo; echo "=== llama-server starts (spec flags must be absent)"
grep "starting llama-server" "$LOG" \
  | grep -oE -- '--model [^ ]*|--spec-type [^ ]*|--model-draft [^ ]*' || true
grep -c "draft-mtp" "$LOG" | sed 's/^/draft-mtp mentions in the server log: /' || true
exit $STATUS
