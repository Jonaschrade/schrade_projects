# Environment record

Where the SCC capture goes: what a run's `params.json` does not carry, the packages above all. **No capture is filed:** the one under Ollama 0.40.13 is still to come (§ Capture), and the retired v2 schedule's, under 0.34.0, is in git history (last at `fef9c586`). Until then, a run's model and server are on record in its own `params.json` (`model_digest`, `ollama_version`, `node`), and the sampler in [`../pull_models.sh`](../pull_models.sh), which sets and checks it.

Packages are not logged per run; they rest on the environment staying as captured. [`../requirements.txt`](../requirements.txt) pins the direct dependencies; [`../../analysis/requirements.txt`](../../analysis/requirements.txt) pins the local analysis.

## Capture

`pull_models.sh pull` writes the run tags' Modelfiles (`modelfile_<tag>.txt`), `ollama_tags.json` (`/api/tags`) and `scc_env.txt` (interpreter, `pip freeze`, Ollama version) to `~/v2_env/` on the login node, against a temporary server with no GPU (`../RUNBOOK.md` § Pulling models). They are filed here after review, under the same names.

When the schedule ends, the capture is re-run without the pull and diffed against the filed one:

```bash
source ~/setup_thesis_env.sh
D=~/v2_end; SIM=~/projects/thesis/simulation; mkdir -p $D
export OLLAMA_HOST=127.0.0.1:11990
ollama serve > $D/serve.log 2>&1 &
sleep 5
curl -s http://$OLLAMA_HOST/api/tags > $D/ollama_tags.json
for m in qwen3.5-nodraft:4b gemma4-nodraft:e4b granite4.2-nodraft:3b; do ollama show "$m" --modelfile > $D/modelfile_${m/:/_}.txt; done
{ which python; python --version; pip freeze; ollama --version; } > $D/scc_env.txt 2>&1
kill %1; unset OLLAMA_HOST
for f in $D/*.txt $D/*.json; do diff -q "$f" "$SIM/environment/$(basename "$f")"; done
```
