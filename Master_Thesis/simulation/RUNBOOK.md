# RUNBOOK: Thesis-Simulation auf SCC

Wiederkehrender Startvorgang nach Login auf `scc2` oder nach einem Crash. Das
Setup ist einmalig erfolgt und wird hier nicht beschrieben.

## Terminologie

Vier Ebenen, einheitlich in `../CLAUDE.md`, `run_stage.sh`, hier und unter
`analysis/` (`analysis/runs.py` liest sie aus einem Logpfad):

| Ebene | Beispiel | Was es ist |
|---|---|---|
| **Stage** | `1` | Block des Schedules (Stage 1 = Phasenübergangs-Scan) |
| **Sub-Stage** | `1-01` | ein Parameterpunkt darin — das nimmt `run_stage.sh` als `SUBSTAGE`, und das rechnet **ein** Job |
| **Run** | `run_1-01_r1` | eine Replikation davon: ein Lauf von `main_network.py`, ein Logverzeichnis |
| **Kohorte** | `m25`, `m100` | nur im (abgeschlossenen) Pilot: ein Durchlauf des Schedules bei fixem Setup; variiert, was der Schedule selbst festhält (der Horizont `NETWORK_MAX_ROUNDS`) |

Logpfad Pilot: `analysis/logs/pilot/Stage<stage>/<kohorte>/run_<sub-stage>_r<n>/`.
Logpfad finaler Schedule: `analysis/logs/Stage<stage>/run_<sub-stage>_r<n>/` —
**ohne** Kohortenverzeichnis: der finale Schedule hält Populationsgröße und
Horizont (gesetzt durch Screen `A-00`) durchgehend fest, es gibt nichts, das eine
Kohorte trennen müsste.

> **Beim Einsortieren eines finalen Jobs kein Kohortenverzeichnis anlegen.**
> `runs.discover()` erkennt das Layout am Elternverzeichnis (`Stage<n>` direkt =
> keine Kohorte); ein falsches Kohortenverzeichnis crasht nicht, sondern legt die
> Läufe unter einem falschen Kohortennamen ab. Stage-Nummern sind über beide
> Schedules **nicht eindeutig** (Pilot-Stage 6 = Memory-Ablation, finale Stage 6 =
> Arm D1): unterscheiden über `cohort` (`"m100"` vs. `None`).

`STAGE=` bleibt veralteter Alias für `SUBSTAGE=`, damit eingereihte Jobs nicht brechen.

## Kurzfassung

Der SCC ist ein **Sun-Grid-Engine-Cluster**. Man arbeitet nie direkt auf einem
Rechenknoten, sondern reicht Jobs vom Frontend `scc2` in eine Queue ein — für
uns immer die Queue `gpu`. `$HOME` und `/data/scc` sind auf allen Knoten
dasselbe Dateisystem, deshalb lässt sich ein laufender Job bequem von `scc2`
aus mitlesen.

**Der interaktive Weg (`qrsh`) ist seit dem Cluster-Server-Update defekt**, ssh
auf Knoten ebenfalls. Alles läuft daher über Batch-Jobs:
`qsub -v SUBSTAGE=<sub-stage> -N substage_<sub-stage> run_stage.sh`. Das Skript setzt aus der
Sub-Stage alle Parameter, belegt eine GPU, startet einen Ollama-Server und rechnet
`NREP` Runs sequentiell dagegen. Folge-Sub-Stages lassen sich sofort
mitschicken, per `-hold_jid` verkettet.

Zwei Eigenheiten des Clusters bestimmen fast alles Weitere:

1. **GPUs werden nicht von SGE zugeteilt**, sondern von einer `starter_method`
   über Lock-Verzeichnisse in `/tmp` des Knotens. SGEs Zähler (`qc:gpu`) und die
   Realität können auseinanderlaufen — hinterlässt ein hart beendeter Fremdjob
   sein Lock, ist die Karte für alle dauerhaft blockiert, während SGE sie als
   frei meldet. Jobs, die dorthin dispatcht werden, sterben **vor** der ersten
   Skriptzeile. Details und Gegenmittel: [GPU-Zuteilung](#hintergrund-gpu-zuteilung-auf-der-gpu-queue).
2. **`h_vmem` ist ein Consumable**, kein Hinweis. SGE rechnet mit den
   *angeforderten* Werten laufender Jobs; ein Knoten kann laut `qhost` leer
   aussehen und trotzdem kein Kontingent haben. Fremdjobs halten hier gern
   188 G bei 30 Tagen Laufzeit — Wartezeiten von Tagen bis Wochen sind normal.

**Bewährte Herangehensweise bei belegtem Cluster:** pro brauchbarem Knoten
einen Wartejob absetzen (scc213, scc214, scc192 auf Vorrat), jeweils mit
eigenem `#$ -N`, aber **identischer Sub-Stage und identischem `REP_OFFSET`**.
Wer zuerst startet, gewinnt; die anderen brechen dank Sub-Stage-Lock von selbst
ab, statt in dieselben Ausgabeordner zu schreiben.
Danach `qstat` beobachten und die Verlierer per `qdel` entfernen.

**Mehrere eigene Jobs auf demselben Knoten sind normaler Betrieb, nicht nur ein
Kollisionsfall der Wartejob-Strategie oben.** `gpu=1` reserviert eine von acht
(scc213, scc214) bzw. vier (scc192) physischen Karten; der Lock-Mechanismus gibt
jedem Job seine eigene Karte, und der job-eigene `OLLAMA_HOST`-Port (siehe
`run_stage.sh`) hält die Ollama-Server auseinander. Die Parallelitäts-Obergrenze
ist damit die GPU-Zahl **des Knotens** und das `h_vmem`-Kontingent, nicht 1
(bestätigt: 5 gleichzeitige `substage_*`-Jobs auf `scc214`). Bei freier Kapazität
also mehrere *verschiedene* Sub-Stages auf denselben Knoten pinnen
(`-q gpu@sccXXX` je `qsub`).

**Wenn etwas schiefgeht, entscheidet die erste Logzeile** — die Tabelle in
[Batch-Betrieb](#die-erste-logzeile-entscheidet) ordnet sie zu.

---

## Voraussetzungen

- venv: `/data/scc/jonas.schrade/envs/thesis`
- Ollama-User-Installation: `/data/scc/jonas.schrade/ollama/install` (**0.34.0**,
  für den ganzen finalen Schedule fixiert; Upgrade von 0.30.10 siehe
  [Modelle pullen](#modelle-pullen) — nicht die System-Binary `/software/bin/ollama`,
  die ist 0.1.26 und unbrauchbar)
- Modelle unter `/data/scc/jonas.schrade/ollama/models`: `qwen3.8-nodraft:27b`
  (primär, finaler Schedule), `devstral-small-2:24b` und `gemma4-nodraft:26b`
  (Klammer E: Stage 8 und Classifier-Audit), `qwen3.5:35b` (Pilot), `qwen2.5:32b`
  und `qwen2.5:14b` (Fallbacks, letzteres deutlich schneller). Die beiden
  `-nodraft`-Tags sind lokal gebaut, nicht gepullt: `qwen3.8:27b` bzw. `gemma4:26b`
  ohne Speculative Decoding. Ein Modell muss vorliegen, **bevor** ein Job es
  anfordert ([Modelle pullen](#modelle-pullen)).
- Embeddings laufen über `sentence-transformers` auf der CPU, kein Ollama-Modell nötig
- Setup-Skript `~/setup_thesis_env.sh`, Code unter `~/projects/thesis/simulation`
- Output-Verzeichnis `/data/scc/jonas.schrade/thesis-runs`

---

## Modelle pullen

Einmalig vor dem finalen Schedule, auf `scc2`, **bevor** ein Job ein neues Modell
anfordert. Ein fehlendes Modell fällt sonst erst nach der Wartezeit in der Queue
auf: `run_classifier.sh` bricht dann sofort ab, `run_stage.sh` lässt jedes
Replikat scheitern.

**1. Ollama auf 0.34.0 heben.** Die User-Installation war 0.30.10; `qwen3.8` läuft
erst ab 0.32.12. Die Version bleibt für den ganzen Schedule fixiert, und jeder Run
schreibt sie als `ollama_version` in seine `params.json`. Nur ohne laufende eigene
Jobs, und erst herunterladen, dann die alten Bibliotheken entfernen — sonst mischen
sich zwei Versionen, oder ein abgebrochener Download hinterlässt gar keine:

```bash
I=/data/scc/jonas.schrade/ollama/install
A=/data/scc/jonas.schrade/ollama-0.34.0.tar.zst
curl -fL -o "$A" https://github.com/ollama/ollama/releases/download/v0.34.0/ollama-linux-amd64.tar.zst
rm -rf "$I/lib/ollama" && tar x --zstd -f "$A" -C "$I"
source ~/setup_thesis_env.sh && ollama --version      # muss 0.34.0 zeigen
```

Ohne laufenden Server meldet `ollama --version` nur `Warning: client version is 0.34.0`
plus `could not connect to a running Ollama instance` — beides erwartet, der Schritt
ist damit bestanden.

`tar --zstd` braucht `zstd` auf `scc2` (`which zstd`); fehlt es, das Archiv lokal
entpacken und per `scp` hochladen.

**2. Die drei Modelle pullen.** Ein eigener Server auf einem freien Port reicht;
geladen wird dabei nichts, eine GPU braucht es nicht. Zusammen ~52 GB unter
`/data/scc/jonas.schrade/ollama/models`:

```bash
source ~/setup_thesis_env.sh
export OLLAMA_HOST=127.0.0.1:11990
ollama serve > ~/ollama_pull.log 2>&1 &
sleep 5
for m in qwen3.8:27b devstral-small-2:24b gemma4:26b; do ollama pull "$m"; done
ollama list                                            # alle drei Tags müssen erscheinen
kill %1
```

Die Tags nennen ihre Größe (`devstral-small-2:24b`, nicht `latest`), damit ein
verschobenes `latest` die Gewichte nicht still austauscht. Welchen Stand ein Run
tatsächlich hatte, steht als `model_digest` in seiner `params.json`.

**3. Speculative Decoding abschalten: die `-nodraft`-Tags bauen.** Ollama 0.34.0
schaltet für Modelle mit mitgeliefertem Draft-Modell Speculative Decoding
automatisch ein — `qwen3.8:27b` hat einen eingebauten MTP-Draft, `gemma4:26b` einen
separaten `DRAFT`-Blob. Der llama-server startet dann mit
`--spec-type draft-mtp --spec-draft-n-max N --spec-draft-backend-sampling` und
stirbt direkt nach dem Laden mit `CUDA error: out of memory`, trotz ~94 GiB freiem
VRAM (Signatur: [Fehlersuche](#fehlersuche)). Beobachtet auf scc214 bei 3/3
qwen- und 1/1 gemma-Versuchen; scc213 ungeprüft, weil dort die Discovery-Falle
vorher greift. `OLLAMA_FLASH_ATTENTION=0` hilft nicht, eine Umgebungsvariable
dafür gibt es nicht. Der Modellparameter `draft_num_predict 0` schaltet es ab:

```bash
source ~/setup_thesis_env.sh
export OLLAMA_HOST=127.0.0.1:11990
ollama serve > ~/ollama_create.log 2>&1 &
sleep 5
printf 'FROM qwen3.8:27b\nPARAMETER draft_num_predict 0\n' > ~/Modelfile.qwen-nodraft
printf 'FROM gemma4:26b\nPARAMETER draft_num_predict 0\n'  > ~/Modelfile.gemma-nodraft
ollama create qwen3.8-nodraft:27b -f ~/Modelfile.qwen-nodraft
ollama create gemma4-nodraft:26b  -f ~/Modelfile.gemma-nodraft
ollama show qwen3.8-nodraft:27b --parameters   # draft_num_predict 0, Sampling wie Basis
ollama show gemma4-nodraft:26b  --parameters
kill %1; unset OLLAMA_HOST
```

Die abgeleiteten Tags verweisen auf dieselben Gewichts-Blobs (kein zusätzlicher
Plattenplatz) und erben alle Sampling-Parameter der Basis; verifiziert identisch
(qwen: `temperature 1`, `top_k 20`, `top_p 0.95`; gemma: `temperature 1`,
`top_k 64`, `top_p 0.95`). Speculative Decoding mit Verifikation lässt die
Ausgabeverteilung unverändert, das Abschalten kostet also höchstens Tempo, nicht
Inhalt. `gemma4-nodraft:26b` trägt den `DRAFT`-Blob weiter im Modelfile, der
Parameter unterdrückt ihn trotzdem. Kontrolle nach dem Start eines Jobs — muss
**leer** bleiben:

```bash
grep -m1 "starting llama-server" ~/ollama_<jobid>.log | grep -oE -- '--spec-type [^ ]*|--model-draft [^ ]*'
```

Der `model_digest` eines `-nodraft`-Tags unterscheidet sich vom Basis-Tag (anderes
Manifest, gleiche Gewichte); `params.json` protokolliert also den Tag, der
wirklich lief. `devstral-small-2:24b` hat keinen Draft und läuft unverändert.

---

## Betriebsart wählen

**Batch (`qsub`) ist der Standard** — wartet in der Queue, überlebt Disconnects,
braucht keinen Node-Login.

**Interaktiv (`qrsh`)** wird eingeplant, öffnet aber keine Shell (Exit 0,
Rücksprung auf scc2). Ob es wieder geht, klärt ein Test:

```bash
qrsh -verbose -q gpu -l h_vmem=128G,tesla_l40=1,gpu=1 ; echo "Exit: $?"
```

Ändert sich der Prompt auf einen Knotennamen, ist der interaktive Weg zurück.
`-verbose` ist Pflicht — ohne das Flag verschluckt `qrsh` auch Fehlermeldungen.

---

## Ressourcen anfordern

Die Ressourcen-Zeile ist bei beiden Betriebsarten identisch (`#$ -l` im Skript
bzw. `-l` bei `qrsh`/`qsub`):

```bash
-l h_vmem=128G,tesla_l40=1,gpu=1   # scc213 (L40S)
-l h_vmem=96G,rtx_6000=1,gpu=1     # scc214 (Blackwell); Kontingent knapp
-l h_vmem=128G,gpu=1               # beide, Scheduler wählt
```

| Knoten | GPU | VRAM | RAM | Feature | Anmerkung |
|---|---|---|---|---|---|
| scc213 | 8× L40S | 48 GiB | 1024 GB | `tesla_l40=1` | Reichlich `h_vmem`-Kontingent. GPU 0 oft fremdbelegt → Discovery-Falle; anfällig für verwaiste Locks. |
| scc214 | 8× RTX PRO 6000 Blackwell | ~96 GB (mit `nvidia-smi --query-gpu=memory.total` verifizieren) | 1536 GB | `rtx_6000=1` | Blackwell (sm_120) läuft mit unserem Ollama-Build verifiziert auf GPU — `devstral-small-2:24b` sowie die `-nodraft`-Tags; `qwen3.8:27b`/`gemma4:26b` mit Speculative Decoding crashen beim Laden ([Modelle pullen](#modelle-pullen), Schritt 3). `h_vmem=128G` läuft dort, aber das Kontingent ist knapp: bei mehreren parallelen Jobs reicht es nicht, daher fordert der Schedule 96G. `run_stage.sh` ist per `-q gpu@scc214` hierher gepinnt, `run_classifier.sh` dagegen auf scc192. |
| scc192 | 4× L40 | 45 GiB | 1024 GB | `tesla_l40=1` | VRAM reicht. Meist `disabled` (`qstat -f -q gpu@scc192`, Spalte `states` = `d`), dabei oft **völlig leer**. Lohnt einen Wartejob auf Vorrat mit `#$ -q gpu@scc192`: Batch-Jobs bleiben auf disabled Queues in `qw` und starten bei Freigabe. Root-eigener ollama auf Port 11434. |
| scc146 | 4× Tesla V100 | 16 **oder** 32 GB — ungeprüft, nur 32 GB trägt qwen3.8 (18 GB) | 188 GB | `tesla_v100=1` | Meist `adu`: das `u` heißt, der execd antwortet nicht. Nicht einplanbar. |
| scc195-199 | 8× RTX 2080 Ti | 11 GiB | 128 GB | `rtx_2080ti=1` | **Zu klein** für jedes Modell des finalen Schedules. |

**Verfügbarkeit vor dem Absetzen prüfen:**

```bash
qstat -F gpu -q gpu                  # 'qc:gpu=N' = freie Karten laut SGE
qhost -F h_vmem -h scc213,scc214     # 'hc:h_vmem' = freies Speicherkontingent
qstat -u "*" -q gpu                  # wer läuft gerade
qstat -u "*" -s p -q gpu             # wer wartet, nach Priorität sortiert
```

Bei Wartenden lohnt der Blick auf ihre Anforderung — ein auf `gpu@sccXXX`
gepinnter Fremdjob konkurriert nur dort:

```bash
qstat -j <fremdjob-id> | grep -E 'hard resource_list|hard_queue_list'
```

**Vier Regeln, die je einen Fehlversuch kosten:**

- **`gpu=1` mitgeben — und nie mehr.** Der Complex hat Default 0, seine Kapazität
  hängt an der **Queue** (`qconf -sq gpu`), nicht am Host; deshalb zeigen
  `qhost -F gpu` und `qconf -se scc213` nichts an. Er steuert, **wohin** SGE
  dispatcht. Ein höherer Wert lässt die `starter_method` entsprechend viele
  **physische** Karten verlangen und scheitern.
- **`h_vmem` ist der Filter gegen zu kleine Knoten.** 128G hält den Job von
  scc195-199 fern (nur 128 GB Host-RAM). Nur zusammen mit einem Feature-Constraint
  darf der Wert sinken — `rtx_6000=1` pinnt dann selbst auf scc214.
- **Kein Host-Pin** (`-l hostname=…`): scheitert ohne Retry, sobald der Knoten
  belegt ist. Auf eine Queue-Instanz pinnen (`-q gpu@sccXXX`) ist dagegen
  unproblematisch.
- **`-l` auf der Kommandozeile ersetzt `#$ -l` im Skript nicht, es wird damit
  vereinigt.** `qsub -l rtx_6000=1` auf ein Skript mit `tesla_l40=1` ergibt beide
  Features → `no suitable queues`. Skript kopieren und per `sed` umschreiben,
  danach `grep -n '^#\$' <skript>` kontrollieren.

---

## Batch-Betrieb

Das Skript liegt versioniert unter `simulation/run_stage.sh` und kommt per
`git pull` auf den Cluster (**nicht ins Terminal pasten** — Heredoc-Paste
zerlegt lange Blöcke).

**Das Skript wird nicht mehr editiert.** Die Sub-Stage kommt beim Absetzen mit, die
Sub-Stage-Tabelle im Skript setzt daraus alle `SIM_*`-Parameter:

```bash
cd ~/projects/thesis/simulation && git pull
S=1-04; qsub -v SUBSTAGE=$S -N substage_$S ~/projects/thesis/simulation/run_stage.sh
qstat -u jonas.schrade                       # qw → r
```

`SUBSTAGE` und `-N` aus **einer** Variablen zu setzen ist Absicht: der Jobname ist
nur ein Etikett, `SUBSTAGE` bestimmt die Physik. Getrennt getippt driften sie
auseinander, und das Log heißt dann nach einer Sub-Stage, die es nicht enthält.

**Warum die Parameter nicht in `config.py` stehen:** `config.py` wird gelesen,
wenn der Job **startet**, nicht wenn er eingereiht wird; bei mehreren
eingereihten Sub-Stages wäre die Datei längst umgesetzt. Was ein Job rechnet,
steht deshalb in der versionierten Tabelle von `run_stage.sh` plus der
Parameterzeile im Job-Log; `config.py` liefert nur Defaults für interaktive Läufe.

> **Ein veralteter Checkout bleibt die gefährlichste Fehlerquelle.** Eine alte
> Fassung schreibt unter altem `SIM_RUN_ID` in die Ordner einer fertigen
> Sub-Stage: `events.jsonl` wird **angehängt**, `personas.json` und
> `network_rounds/*.json` **überschrieben**. Also immer erst `git pull`, dann die
> Parameterzeile prüfen, die der Job tatsächlich geladen hat:
>
> ```bash
> grep -m1 'Sub-Stage' $RUNDIR/substage_<S>.o<jobid>
> ```

**Mehrere Knoten gleichzeitig bewerben**, wenn der Cluster voll ist. Die Kopien
unterscheiden sich nur noch in der Ressourcenzeile — Name und Sub-Stage kommen von
der Kommandozeile:

```bash
cd ~/projects/thesis/simulation && git pull && cd ~   # zuerst! sonst kopiert man eine alte Fassung
for v in 213 214 192; do cp ~/projects/thesis/simulation/run_stage.sh ~/run_wait_$v.sh; done

# run_wait_214.sh bleibt wie im Repo (scc214). Die beiden L40-Knoten umschreiben:
sed -i -e 's|^#\$ -q gpu@scc214$|#$ -q gpu@scc213|' \
       -e 's|^#\$ -l h_vmem=96G,rtx_6000=1,gpu=1$|#$ -l h_vmem=128G,tesla_l40=1,gpu=1|' ~/run_wait_213.sh
sed -i -e 's|^#\$ -q gpu@scc214$|#$ -q gpu@scc192|' \
       -e 's|^#\$ -l h_vmem=96G,rtx_6000=1,gpu=1$|#$ -l h_vmem=96G,tesla_l40=1,gpu=1|' ~/run_wait_192.sh
grep -n '^#\$' ~/run_wait_*.sh               # Kontrolle vor dem Absetzen: sed meldet
                                             # NICHTS, wenn das Muster nicht passt —
                                             # der Job liefe dann auf dem falschen Knoten

S=1-04
for v in 213 214 192; do qsub -v SUBSTAGE=$S -N substage_${S}_$v ~/run_wait_$v.sh; done
```

Die `#$ -o`-Zeile bleibt unangetastet: Sie zeigt auf ein Verzeichnis, SGE
benennt die Logs nach `<jobname>.o<jobid>`. Sobald einer auf `r` springt, die
übrigen per `qdel` entfernen.

### Mehrere Sub-Stages auf Vorrat einreihen

Bei Wartezeiten von Tagen lohnt es, die Folge-Sub-Stages sofort mitzuschicken: ein
wartender Job sammelt Wartezeit-Priorität, ein nicht abgesetzter nicht. Parallel
laufen sie problemlos (eigene Karte, eigener Port, siehe Kurzfassung); wo die
Reihenfolge zählt, etwa auf einem knappen Knoten, verkettet `-hold_jid`:

```bash
J=$(qsub -terse -v SUBSTAGE=1-04 -N substage_1-04 ~/projects/thesis/simulation/run_stage.sh)
J=$(qsub -terse -v SUBSTAGE=1-05 -N substage_1-05 -hold_jid $J ~/projects/thesis/simulation/run_stage.sh)
qstat -u jonas.schrade                       # hqw = wartet auf Vorgänger
```

Der Hold löst sich, sobald der Vorgänger **endet** — unabhängig vom Exit-Status.
Stirbt eine Sub-Stage an einem Phantom-Lock, rückt die nächste also nach, statt zu
blockieren. Was ein `hqw`-Job rechnet, steht fest: die Sub-Stage-Tabelle im Skript
ist zum Zeitpunkt des `qsub` festgeschrieben.

**Eine Stage ist bewusst unvollständig** und bricht beim Start ab, wenn die
inhaltliche Angabe fehlt: Stage 3 braucht die **Gegenposition**
desselben Topics (`config.TOPIC_TEXT_OPPOSITE`). Stage 5 bringt ihre beiden Topics
in der Tabelle mit (`5-01`…`5-03` `renewables`, `5-04`…`5-06` `coffee_tea`)
und wird ohne Topic-Angabe abgesetzt. Jedes Topic-Override, das eine Sub-Stage
nicht vorsieht, bricht den Job ab — an der falschen Sub-Stage ersetzte es deren
Topic still:

```bash
qsub -v SUBSTAGE=3-01,SIM_TOPIC_TEXT='<Gegenposition>' -N substage_3-01 $S
qsub -v SUBSTAGE=5-04 -N substage_5-04 $S
```

Die Kopplung muss **nie** mitgegeben werden: Leiter, Anker, β-Gitter und `w`
stehen in der Präambel des Skripts und kommen aus den Screens dieses Schedules.
Solange ein Screen noch nicht gelaufen ist, stehen dort Platzhalter — und das
Skript **bricht ab**, statt sie zu rechnen (`SCREENED_LADDER`, `SCREENED_MED`,
`SCREENED_BETA`, `SCREENED_D2`, alle auf 0). Ein Platzhalter-Lauf erzeugt sonst
ein vollständiges, plausibles Log, und nichts darin sagt später, dass der Wert
nur ein Kandidat war.

**`1-00`, `1-01` und `1-06` hängen an keinem Screen** (L0, L1 und L6 sind exakt);
`1-02`…`1-05` hängen an `A-01`, das gelaufen ist (`SCREENED_LADDER=1`). Stage 1 ist
damit vollständig absetzbar. Alles andere wartet auf sein Screen-Ergebnis; die
Reihenfolge steht in `../analysis/final_schedule.md` § Submission order.

### r = 10: zwei Jobs je Sub-Stage

Jede Sub-Stage mit r = 10 wird **zweimal** abgesetzt, der zweite Job mit
`REP_OFFSET=5`. Ohne den Offset überschreibt der zweite Job `run_<S>_r1..r5`;
mit ihm rechnet er `r6..r10`. Zwei Jobs à ~22 h statt einem à ~44 h, und der
Sub-Stage-Lock trägt den Offset im Namen, blockiert sich also nicht selbst.

```bash
S=~/projects/thesis/simulation/run_stage.sh
qsub -v SUBSTAGE=1-01               -N substage_1-01_a $S
qsub -v SUBSTAGE=1-01,REP_OFFSET=5  -N substage_1-01_b $S
```

`7-01`/`7-02` tragen `NREP=6` und laufen genauso zweimal (`REP_OFFSET=6`), falls
die Kappung r = 12 aus Regel R3 greift.

### Ungleiche Teiljobs: `NREP_JOB`

`NREP_JOB` begrenzt, wie viele Replikate *dieser* Job rechnet, und darf den
Tabellenwert nur verkleinern (`REP_OFFSET + NREP_JOB <= NREP`, sonst Abbruch).
Gebraucht, wenn die Replikate zusammen `h_rt` sprengen: `0-01` braucht bei
H = 400 ~55 h je Run, also höchstens zwei je Job (final_schedule.md § Screen A-00):

```bash
S=~/projects/thesis/simulation/run_stage.sh
qsub -v SUBSTAGE=0-01,NREP_JOB=2              -N substage_0-01_a $S   # r1..r2
qsub -v SUBSTAGE=0-01,REP_OFFSET=2,NREP_JOB=2 -N substage_0-01_b $S   # r3..r4
qsub -v SUBSTAGE=0-01,REP_OFFSET=4,NREP_JOB=1 -N substage_0-01_c $S   # r5
```

Der Lock trägt den Offset, die Teiljobs laufen also gleichzeitig.

### Seeds: nie einer für den ganzen Job

`SIM_SEED` gilt für **jeden** Run eines Jobs und würde alle Replikate auf einen
Graphen festnageln — die Regime-Fraktion wäre dann auf eine Topologie bedingt
(Regel 2). Das Skript **bricht deshalb ab**, wenn `SIM_SEED` gesetzt ist. Die
Blockung nach Regel 3 kommt als Seed-Block, ein Seed je Replikat, indiziert über
`r`:

```bash
qsub -v SUBSTAGE=7-01,SEED_BLOCK='11 22 33 44 55 66' -N substage_7-01 $S
```

Damit teilen zwei Arme an derselben Kopplung ihre Topologien replikatweise,
während die Replikate innerhalb eines Arms unabhängige Züge bleiben.

### Vorher lokal prüfen

Kostet Sekunden und fängt genau die Fehler, die die Tabelle sonst erst auf der
GPU zeigt:

```bash
bash -n simulation/run_stage.sh
bash simulation/check_substages.sh            # alle 36 Sub-Stages, alle Punkte verschieden
DRY_RUN=1 SUBSTAGE=7-02 bash simulation/run_stage.sh   # eine einzelne auflösen
```

`DRY_RUN=1` löst die Tabelle auf, druckt eine Zeile mit allen `SIM_*`-Werten und
beendet sich **vor** Lock, Ollama und jedem Run; `OWED=` nennt die noch fehlenden
Screens. `check_substages.sh` prüft zusätzlich, dass keine zwei Sub-Stages auf
denselben Parameterpunkt fallen — sonst wird ein Arm zweimal bezahlt und ein
„Kontrast“ ist eine Konfiguration gegen sich selbst.

Was ein Job **tatsächlich** gerechnet hat, steht danach in seiner eigenen
`params.json`, inklusive `git_commit`, `model_digest` und `ollama_version` — die Tabelle sagt nur,
was er rechnen *wird*.

Einsortiert wird nach `analysis/logs/Stage<n>/run_<sub-stage>_r<n>/`: **kein
Kohortenverzeichnis**, weil der finale Schedule Horizont und Populationsgröße
durchgehend fixiert. Die Pilotläufe liegen daneben unter
`analysis/logs/pilot/Stage<n>/<kohorte>/` und werden nie mit ihnen gepoolt.

### Das Skript

Maßgeblich ist `simulation/run_stage.sh`, die Werte begründet
`../analysis/final_schedule.md`. **Was die Blöcke tun:**

- **Präambel.** Jeder gefittete Betriebspunkt steht **genau einmal** dort: Leiter,
  Anker, β-Gitter, `w`, R1s Kopplung. Eine Sprosse ist damit an einer Stelle
  änderbar, und die vier `SCREENED_*`-Flags sagen, welche davon noch Platzhalter
  sind. `analysis/benchmark/make_nulls.py` liest dieselbe Präambel, damit die
  CPU-Nullmodelle dieselbe Leiter fahren wie die LLM-Arme.
- **Sub-Stage-Tabelle.** Der einzige Ort, an dem Label und Physik zusammenkommen.
  Ein Tippfehler in `SUBSTAGE` trifft keinen `case`-Zweig und bricht ab, statt die
  Werte der Vorgänger-Sub-Stage zu erben. Weil die Werte im Job-Environment
  stehen und nicht in `config.py`, ist festgeschrieben, was ein Job rechnet,
  sobald er eingereiht ist — die Voraussetzung dafür, mehrere Stages auf Vorrat
  einzureihen. Die `echo`-Zeile schreibt die effektiven Werte ins Log; sie ist
  der Beleg, den man nach dem Start prüft.
- **Guards.** Jeder gefangene Fehler fiele sonst erst beim Vergleich Monate
  später auf: ein Platzhalter-Lauf, ein falscher Horizont (`NEED_HORIZON`, per
  `sed` gelesen, weil vor dem Env-Setup kein brauchbares `python` existiert), ein
  Anker, der keine Sprosse ist, ein nicht aufsteigendes β-Gitter, ein `w` außerhalb
  des Gitters, `7-03` bei gefittetem w = 1.0, ein fehlendes oder falsches
  Topic-Override, ein Seed für den ganzen Job, ein `NREP_JOB` über `NREP` hinaus.

- **`#$`-Header.** SGE liest diese Zeilen beim Absetzen, nicht die Shell. `h_rt`
  ist die maximale Laufzeit — die `gpu`-Queue erlaubt mindestens 30 Tage
  (`h_rt=2592000`, so laufen Fremdjobs dort); danach killt SGE den Job. `-j y`
  legt stderr mit ins stdout-Log. `-o` zeigt auf ein **Verzeichnis**, SGE
  benennt die Datei dann `<jobname>.o<jobid>` — pro Job eindeutig, damit man nie
  versehentlich das Log des vorigen Laufs liest.
- **`set -euo pipefail` + Binary-Check.** Der Job soll früh und laut scheitern.
  Die falsche `ollama`-Binary (System-0.1.26) ist die teuerste Fehlerquelle
  überhaupt, weil sie erst beim Modell-Laden stillschweigend abstürzt.
- **Stage-Lock und Traps.** Wer auf mehreren Knoten parallel wartet, riskiert
  zwei gleichzeitige Läufe in denselben Ausgabeordnern und damit vermischte
  `events.jsonl`. `mkdir` ist atomar, auch über NFS — der zweite Job bricht
  sauber ab. Der Lock heißt `.substage_<S>_o<REP_OFFSET>.lock`: die zwei Jobs
  einer r=10-Sub-Stage schreiben disjunkte Replikate und dürfen deshalb parallel
  laufen, zwei Jobs mit demselben Offset nicht. Die Traps räumen Lock **und** Ollama-Daemon auf (SIGTERM →
  `exit 143` → EXIT-Trap) — aber **nicht verlässlich bei `qdel`**: Bash führt
  Traps erst aus, wenn das Vordergrundkommando zurückkehrt, und hängt `python`
  in einem LLM-Call, kommt SGEs SIGKILL zuerst. Nach jedem `qdel` also
  prüfen: `ls -d $RUNDIR/.substage_*.lock`, ggf. `rmdir`. Ein verwaister Lock
  blockiert nur die *eigene* Stage; die Abbruchmeldung nennt den Pfad.
- **GPU-Zuteilung.** `$SGE_GPU` kommt von der `starter_method` und hat Vorrang.
  `setup_thesis_env.sh` setzt `CUDA_VISIBLE_DEVICES=7` nur, wenn die Variable
  leer ist, überschreibt die Zuteilung also nicht — die Reihenfolge (erst
  sourcen, dann `SGE_GPU` setzen) sichert das zusätzlich ab. Der
  `nvidia-smi`-Fallback greift nur außerhalb der `gpu`-Queue und überspringt
  Index 0 bewusst, weil dort die Discovery-Falle lauert.
- **Ollama-Start.** Port `11500 + JOB_ID % 400` (11434 kann root gehören, ein
  fester Port kollidiert mit dem Nachbarjob), `GGML_CUDA_NO_VMM=1` gegen den
  256-GB-VM-Pool, `LLAMA_ARG_THREADS=${NSLOTS:-1}` gegen einen Thread je
  Knotenkern (siehe [Fehlersuche](#fehlersuche)), Log pro Job unter
  `~/ollama_$JOB_ID.log`. Der anschließende
  Check **wartet** auf Ollamas `inference compute`-Zeile (bis zu 120 s) und
  verlangt darin `library=CUDA`, sonst Abbruch. (Ein blindes `sleep 10` hatte
  einmal ein noch leeres Log als Erfolg gewertet und eine Sub-Stage auf CPU
  rechnen lassen.)
- **Replikat-Schleife.** Alle `NREP` Läufe teilen sich **einen** Server, das
  Modell bleibt geladen. Ein fehlgeschlagenes Replikat stoppt die Schleife nicht
  (`if … else`), sonst würde ein einzelner Ausreißer die ganze Stage kosten.

### Die erste Logzeile entscheidet

| Zeile | Bedeutung |
|---|---|
| `Knoten … GPU N` → `Ollama auf GPU bereit.` | Ollamas Discovery hat `library=CUDA` gemeldet — die Runs laufen auf der GPU. |
| `ERROR: Could only reserve 0 of 1 requested devices.` | Kommt **vor** jeder Skriptzeile, aus der `starter_method`: keine freie Karte laut Lock-Mechanismus. SGE reiht einmal neu ein, der Job stirbt erneut und verschwindet lautlos aus `qstat`. Siehe [GPU-Zuteilung](#hintergrund-gpu-zuteilung-auf-der-gpu-queue). |
| `ABBRUCH: Ollama nicht auf CUDA` | Discovery-Falle: GPU 0 des Knotens ist fremdbelegt. Der Job beendet sich absichtlich sofort. Auf einen anderen Knoten ausweichen — `rtx_6000=1` (scc214) war bisher immer sauber, scc213 ist der Wiederholungstäter. |
| `ABBRUCH: Sub-Stage … läuft bereits` | Ein Parallel-Wartejob war schneller. Genau so gedacht — diesen Job löschen. |
| `… unbound variable` | `set -u` gegen ein unvollständiges Environment; siehe Fehlersuche. |

### Andere Sub-Stage

Nichts editieren, nur `-v SUBSTAGE=` und `-N` mitgeben (siehe oben). Die
Zahl der Runs kommt aus der Sub-Stage-Tabelle. Ein Ollama-Server bedient alle
Runs **sequentiell** — nicht parallel starten (GPU-Contention,
KV-Cache-OOM). Ergebnis: `logs/run_<SUBSTAGE>_r1/` … `_r<NREP>/`.

> `SIM_RUN_ID` benennt nur den Log-Ordner und setzt **keine** Parameter. Dass
> `1-04_r3` wirklich Sub-Stage 1-04 ist, belegt die Parameterzeile im Job-Log, nicht
> der Ordnername — sie zeigt, was der Prozess geladen hat.

---

## Fortschritt verfolgen

Vier Ebenen, von grob nach fein. `python -u` im Runner sorgt dafür, dass alles
ungepuffert und damit live erscheint.

**Job lebt und was er verbraucht** — `maxvmem` ist zugleich die Zahl, an der
sich `h_vmem` für künftige Stages bemessen lässt:

```bash
qstat -j <job-id> | grep -E 'usage|maxvmem'
```

**Welcher Run läuft:**

```bash
grep -E '=== Run|Run .* OK' /data/scc/jonas.schrade/thesis-runs/<jobname>.o<jobid>
```

**Live mitlesen.** Die volle Ausgabe ist geschwätzig (20 Personas je
Run-Start, 20 Interaktionszeilen je Runde), deshalb meist gefiltert:

```bash
tail -f /data/scc/jonas.schrade/thesis-runs/<jobname>.o<jobid>

tail -f /data/scc/jonas.schrade/thesis-runs/<jobname>.o<jobid> \
  | grep --line-buffered -E '=== Run|Reflection phase|Round .* completed|Simulation complete|Run .* OK'
```

`--line-buffered` ist nicht optional — sonst puffert `grep` blockweise und es
kommt minutenlang nichts. Auf NFS kann `tail -f` nachhängen; dann `tail -F`.
`Round N completed in X.Xs` ist die Zeile für die Hochrechnung:
`NETWORK_MAX_ROUNDS` Runden je Run (derzeit 100, siehe `config.py`), `NREP` Runs
je Sub-Stage.

**Fortschritt im Dateisystem**, unabhängig vom Log:

```bash
D=/data/scc/jonas.schrade/thesis-runs/logs/run_1-03_r1
ls $D/network_rounds/ | tail -1     # round_0007.json = Runde 7 von 100
wc -l < $D/events.jsonl             # wächst laufend
```

Stagniert `events.jsonl` über Minuten bei unverändertem Zeitstempel, hängt ein
LLM-Call.

---

## Interaktiver Betrieb

Nur nutzbar, wenn der Test oben eine Shell öffnet. `run_stage.sh` automatisiert
genau diese Schritte.

**1. Knoten anfordern** — siehe [Ressourcen](#ressourcen-anfordern):

```bash
qrsh -verbose -q gpu -l h_vmem=128G,tesla_l40=1,gpu=1
```

**2. Environment laden und Binary prüfen** (häufigste, teuerste Fehlerquelle):

```bash
source ~/setup_thesis_env.sh
which ollama && ollama --version
```

Muss `/data/scc/jonas.schrade/ollama/install/bin/ollama` und `0.34.0` zeigen.

**3. GPU-Zuteilung übernehmen.** In der `gpu`-Queue hat die `starter_method`
bereits eine Karte reserviert; eine eigene Wahl wäre ein Zugriff auf fremd
reservierte Hardware:

```bash
echo "$SGE_GPU"
# nur falls leer: freie Karte selbst suchen (memory.used deutlich unter 1 GB)
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
export CUDA_VISIBLE_DEVICES=<idx>
```

**4. Ollama starten.** Eigener Port, weil auf manchen Knoten ein root-eigener
Daemon auf 11434 läuft. Nur **eigene** Prozesse killen — hier zulässig, weil
genau ein interaktiver Server läuft; `run_stage.sh` darf das nicht und leitet
seinen Port stattdessen aus `$JOB_ID` ab (siehe [Das Skript](#das-skript)):

```bash
export OLLAMA_HOST=127.0.0.1:11500
pkill -u jonas.schrade -f ollama; sleep 2; pgrep -u jonas.schrade -af ollama
export LLAMA_ARG_THREADS=${NSLOTS:-1}      # sonst ein Thread je Knotenkern (Fehlersuche)
GGML_CUDA_NO_VMM=1 ollama serve > ~/ollama.log 2>&1 &
sleep 5; ollama list
grep "inference compute" ~/ollama.log      # erwartet: library=CUDA …
```

> `GGML_CUDA_NO_VMM=1` ist Pflicht: ggml reserviert sonst einen 256-GB-VM-Pool
> (`cuMemAddressReserve`), der am harten `ulimit -Hv` von 128 GB scheitert. Ohne
> das Flag lädt kein Modell.

**5. Simulation starten.** Outputs landen relativ zum CWD in `logs/run_<id>/`:

```bash
cd /data/scc/jonas.schrade/thesis-runs
python ~/projects/thesis/simulation/main_pairwise.py   # Smoke-Test, 2 Agents
nohup python -u ~/projects/thesis/simulation/main_network.py \
  > run_$(date +%Y%m%d_%H%M%S).out 2>&1 &               # voller Lauf, disconnect-fest
```

**6. Sauber beenden:**

```bash
pkill -u jonas.schrade -f ollama; sleep 2
exit
```

---

## config.py-Pflichtwerte

```python
LLM_MODEL   = "qwen3.8-nodraft:27b"  # via SIM_LLM_MODEL überschreibbar (Stage 8)
LLM_NUM_CTX = 4096               # via os.getenv("LLM_NUM_CTX") überschreibbar
OLLAMA_HOST = "127.0.0.1:11500"  # Port wie beim Serverstart
```

`num_ctx=4096` ist kein Detail: ohne Cap wählt ggml 32 768 Tokens KV-Cache und
treibt den VRAM-Bedarf um mehrere GiB hoch (beim Pilotmodell ~27 statt ~20 GiB).

`reasoning=False` muss im `OllamaLLM(...)`-Konstruktor stehen (mappt auf Ollamas
`think: false`). Sonst liefern Thinking-Modelle wie qwen3.8 `<think>…</think>`-Blöcke, und
`_score_importance()` parst den ersten Token zu einer Zahl → stiller
Fallback-Wert 5.0. Der Klassifikator ist über das JSON-Schema robust dagegen,
spart aber Generationszeit.

---

## Ergebnisse

```bash
ls /data/scc/jonas.schrade/thesis-runs/logs/run_<id>/
```

- `params.json` — die Parameter, die dieser Lauf tatsächlich aufgelöst hat, plus Commit, Modell-Digest und Ollama-Version; der Beleg dafür, was er gerechnet hat
- `events.jsonl` — eine Zeile pro Diskussion, Kantenänderung und Reflexion; primäre Analyse-Daten
- `personas.json` — die gezogenen Personas
- `network_rounds/round_NNNN.json` — Snapshot nach jeder Runde, `round_0000` ist der Ausgangszustand

Herunterladen (im **lokalen** Terminal):

```bash
scp -r jonas.schrade@scc2.uni-konstanz.de:/data/scc/jonas.schrade/thesis-runs/logs/run_<id> .
```

---

## Jobs verwalten

```bash
qstat -u jonas.schrade              # eigene Jobs clusterweit
qdel <job-id> [<job-id> …]          # Jobs entfernen (qw wie r); -f erzwingt bei 'dr'
qstat -j <job-id> | tail -30        # Block 'scheduling info:' = warum noch qw
qacct -j <job-id>                   # abgeschlossener Job: hostname, exit_status, maxvmem
qalter -N <neuer-name> <job-id>     # Namen nachträglich ändern
```

`qstat` schneidet die Namensspalte bei zehn Zeichen ab; die vollen Namen zeigt
`qstat -r`. Manuelles Aufräumen entfällt: Die Traps im Runner beenden den
Ollama-Daemon, mit dem Job-Ende gibt SGE Knoten und GPU frei.

---

## Hintergrund: GPU-Zuteilung auf der gpu-Queue

Die Queue hat `prolog NONE`, aber eine `starter_method`
(`/data/sw/sge/scripts/start-gpu.sh` → `start-job.sh`, sourct
`start-gpu_helper.sh`; Basis: [kyamagu/sge-gpuprolog](https://github.com/kyamagu/sge-gpuprolog)).
Sie umschließt jeden Job und reserviert Karten **nicht über SGE**, sondern über
Lock-Verzeichnisse `/tmp/sge-gpu-lock-<jobid>.<taskid>-<dev>` auf dem Knoten.
Danach exportiert sie `SGE_GPU` und `CUDA_VISIBLE_DEVICES`.

Daraus folgen drei Dinge:

- Die Wunschzahl liest der Helper über `qstat.orig` (Logzeile
  `Number of requested GPUS: N`). Die Binary ist unzuverlässig und liefert
  teils nichts; dann greift der Fallback **1** — für uns der richtige Wert.
  Ein höherer `gpu=`-Request wird jedoch ausgelesen und lässt den Helper
  entsprechend viele **physische** Karten verlangen: `gpu=3` endet in
  `Could only reserve 1 of 3` und verschenkt die eine Karte, die frei war.
  **Für diese Simulation gilt immer `gpu=1`.**
- Findet der Helper keine freie Karte, bricht er **vor** dem Job-Skript ab.
  `qacct` zeigt dann `failed 0`, `exit_status 1` — formal ein Job-Exit, kein
  Prolog-Fehler.
- Der Epilog entfernt nur Locks des **eigenen** Jobs. Hart abgebrochene
  Fremdjobs hinterlassen dauerhafte Locks, die eine Karte für immer sperren.

**Verwaiste Locks erkennen.** `qstat -F gpu -q gpu` meldet freie Karten
(`qc:gpu>0`), der Helper findet trotzdem keine. Da jeder Job genau eine Karte
reserviert, muss gelten: *belegte Slots* (aus `qhost -q -h <knoten>`) + *freie
Karten* (`qc:gpu`) = *Gesamtzahl GPUs*. Geht die Rechnung nicht auf, ist die
Differenz Müll. **Kein Userspace-Fix** — die Locks gehören fremden Usern;
Betreiber-Ticket mit Job-ID, `qacct -j <id>` und den beiden Zählungen als Beleg.
Eigendiagnose nur über einen Probe-Job in eine **Nicht-GPU-Queue desselben
Knotens** (`qhost -q -h <knoten>`), da der Helper nur bei `$QUEUE == gpu` läuft;
dort ist `ls -ld /tmp/sge-gpu-lock-*` möglich. Ist die einzige andere Queue des
Knotens `disabled`, entfällt auch das.

**Trotz Phantom-Locks warten statt sterben.** Solange SGE Karten zählt, die der
Helper nicht vergeben kann, wird ein `gpu=1`-Job sofort dispatcht und stirbt.
Als Wartebedingung eignet sich nur eine Ressource, die der Helper **nicht**
liest — praktisch `h_vmem`. Den Wert so wählen, dass er jetzt unerfüllbar ist,
aber erfüllt wird, sobald ein Fremdjob endet: Dann wird zeitgleich dessen
GPU-Lock frei, und der Helper findet die Karte.

```bash
qhost -F h_vmem -h <knoten>                         # hc:h_vmem = frei
qstat -j <fremdjob-id> | grep 'hard resource_list'  # was ein Job dort hält
```

Beispiel: frei 252 G, jeder Fremdjob hält 188 G → `h_vmem` zwischen 253 G und
440 G wählen, etwa 300 G. Kostet keinen echten Speicher (`h_vmem` ist eine
Obergrenze), belegt aber Kontingent — also nicht unnötig hoch ansetzen. Gegen
Konkurrenten, die auf denselben Knoten warten, hilft der Kniff nicht; das
vorher mit `qstat -u "*" -s p -q gpu` prüfen.

---

## Fehlersuche

| Symptom | Ursache | Gegenmittel |
|---|---|---|
| `qrsh`/`qlogin` meldet „successfully scheduled", öffnet aber keine Shell | Interaktiver Attach seit Server-Update defekt (PTY-/Transport-Handoff) | Kein Userspace-Fix → Batch. Regression an den Support melden. |
| `qrsh` springt kommentarlos zurück | Ohne `-verbose` unterdrückt `qrsh` alle Scheduling-Meldungen | Immer `qrsh -verbose … ; echo $?` |
| `qrsh … could not be scheduled` / `got select timeout` | Interaktive Jobs brauchen Sofort-Dispatch ohne Reservierung; scheitert bei Last, wo ein Batch-Job durchkommt | Als `qsub`-Batch einreihen |
| `ssh sccXXX` verlangt Passwort | Host-/Key-Auth zwischen Knoten kaputt; Node-SSH bräuchte zusätzlich einen laufenden Job dort | Node-Login vermeiden → Batch |
| `error: no suitable queues` | Knoten hat das Feature nicht, **oder** `-l` von Kommandozeile und Skript wurden vereinigt | Features prüfen (`tesla_l40` vs. `rtx_6000`); Skript kopieren statt per Kommandozeile übersteuern |
| Job bleibt lange in `qw`, ohne Fehler | Meist `h_vmem`: ein Consumable, SGE rechnet mit den *angeforderten* Werten laufender Jobs, nicht mit `qhost`-MEMUSE | `qstat -j <id> \| tail -30` → `cannot run at host … offers only hc:h_vmem=<bytes>`. Warten ist korrekt, der Job startet automatisch. |
| Job verschwindet binnen Sekunden aus `qstat`, ohne zu laufen | Dispatcht und in der `starter_method` gestorben | Log lesen — der Fehler steht dort **vor** jeder Skriptausgabe. `qacct -j <id>` zeigt `failed 0`, `exit_status 1`. |
| `ERROR: Could only reserve N of M requested devices.` | Keine (bzw. zu wenige) freie GPU laut Lock-Mechanismus | `gpu=1` verwenden, nie mehr. Sonst siehe [GPU-Zuteilung](#hintergrund-gpu-zuteilung-auf-der-gpu-queue). |
| `library=cpu` bzw. `CUDA … device(s) is/are busy or unavailable` trotz freier GPU | **Discovery-Falle:** GPUs im `Exclusive_Process`-Modus; Ollamas Discovery-Probe erzwingt intern `CUDA_VISIBLE_DEVICES=0` und legt dort einen Kontext an — ist GPU 0 fremdbelegt, crasht sie, **bevor** deine Auswahl greift. Weder anderer Index/UUID noch `OLLAMA_LLM_LIBRARY=cuda_v12` helfen. | Kein Userspace-Fix. Knoten mit freier GPU 0 wählen. `run_stage.sh` erkennt es und bricht sofort ab. Prüfen: `nvidia-smi --query-gpu=index,compute_mode --format=csv,noheader` |
| `setup_thesis_env.sh: line N: LD_LIBRARY_PATH: unbound variable` | Das Setup-Skript hängt an nicht gesetzte Variablen an; `set -u` macht daraus einen Abbruch — **nach** erfolgreicher GPU-Reservierung, die Karte ist also verschenkt | In `run_stage.sh` behoben (`set +u` nur ums Sourcen). Bei eigenen Skripten genauso verfahren. |
| `offloaded 0/NN layers to GPU` | Context-/Layout-Größe, **nicht** `h_vmem` | `LLM_NUM_CTX=4096` prüfen, Server sauber neu starten |
| `ggml_aligned_malloc: insufficient memory` | `h_vmem` zu niedrig fürs Modell | Höher anfordern; realer Bedarf aus `qacct -j <id> \| grep maxvmem` eines gelungenen Laufs ablesen |
| `CUDA error: out of memory` bei `cuMemAddressReserve` | ggml reserviert 256-GB-VM-Pool gegen `ulimit -Hv` = 128 GB | `GGML_CUDA_NO_VMM=1` beim Serverstart |
| `ggml-cuda.cu:108: CUDA error: out of memory` trotz ~94 GiB frei, direkt nach `adding speculative implementation 'draft-mtp'` (gemma: `Gemma4Assistant requires ctx_other to be set`); Retry mit `--no-mmproj-offload` scheitert gleich; Python sieht einen `ResponseError` beim ersten Aufruf | Automatisches Speculative Decoding mit Draft-Modell (`--spec-type draft-mtp`). Kein Kapazitätsproblem, Flash Attention ist es auch nicht | `-nodraft`-Tag verwenden ([Modelle pullen](#modelle-pullen), Schritt 3). Prüfen: `grep -m1 "starting llama-server" ~/ollama_<jobid>.log \| grep -c draft-mtp` muss `0` sein |
| Generierung extrem langsam (~0,5 s/Token, `tokens per second` ≈ 2) bei `offloaded NN/NN layers to GPU` und `graph splits = 2`; alle Modelle gleich betroffen; `qstat -j <id>` zeigt CPU-Zeit ≈ Laufzeit | llama-server startet ohne `--threads` mit einem Thread je Knotenkern (`n_threads = 288` im Log), der Job hat aber nur seine Slots: die Threads teilen sich einen Kern (2026-09-14, scc214, Job 1366208: 0,01 Calls/s) | `export LLAMA_ARG_THREADS=${NSLOTS:-1}` vor `ollama serve` (steht in `run_stage.sh` und `run_classifier.sh`; danach 0,43 Calls/s). Prüfen: `grep -m1 "n_threads =" ~/ollama_<jobid>.log` muss `n_threads = 1` zeigen |
| `ollama --version` zeigt `0.1.26` / `/software/bin` | Environment nicht gesourced | `source ~/setup_thesis_env.sh` |
| `address already in use` (11434) | root-Ollama belegt den Default-Port | `export OLLAMA_HOST=127.0.0.1:11500` |
| `pkill: Operation not permitted` | Versuch, den root-Prozess zu killen | `-u jonas.schrade` ergänzen |
| `could not connect to ollama app` | Daemon abgestürzt oder nicht gestartet | `tail -n 30 ~/ollama_<jobid>.log`, Port und Binary prüfen |
| Python meldet 3.13 / ImportError | System-Python statt venv | `source ~/setup_thesis_env.sh`; `which python` muss auf `/data/scc/.../thesis/bin/python` zeigen |
| `SyntaxError: Non-ASCII character '\xc2' in file config.py ... no encoding declared`, Job endet binnen Sekunden | Ein Guard rief `python` **vor** `source ~/setup_thesis_env.sh`: das ist System-Python **2.x** und importiert die UTF-8-Datei (`§` im Docstring) nicht | Behoben — das Horizont-Gate liest `NETWORK_MAX_ROUNDS` per `sed` statt per Import. Guards vor dem Env-Setup nie mit `python` bauen (2026-09-23, Jobs 1506773/1506774; `0-01` war die erste Sub-Stage mit `NEED_HORIZON`, der Pfad war nie gelaufen) |
| Heredoc-Paste erzeugt kaputte Datei | Erster Paste ließ das Heredoc offen, der zweite wurde verschluckt | Skripte nicht ins Terminal pasten, per `git pull` holen. `head -1` muss `#!/bin/bash` sein. |
| Laufender Job endet ohne Traceback mit `exit_status 1`, `failed 0`; `cpu ≈ ru_wallclock`, Ollama lieferte bis Sekunden davor `200` | Die `.o<jobid>`-Datei des **laufenden** Jobs wurde geloescht. Ueber NFS wird sie silly-renamed (`.nfsXXXX`), und der naechste Schreibversuch kann `ESTALE` liefern: der Run stirbt, das `echo` im `else`-Zweig scheitert ebenfalls, und `set -e` beendet das Skript mit 1 — die Schleife haette einen reinen Run-Fehler sonst abgefangen | Logs laufender Jobs nur lesen (`tail`, `less`), nie loeschen; aufgeraeumt wird erst nach `qstat`-Leerlauf. Die Datei ist ausserdem die einzige Kopie der Parameterzeile und der Laufzeiten (2026-09-23, Job 1506780, Run r1 nach ~2,8 h verloren) |
| VSCode Remote-SSH verbindet nicht | Server-Update | In den Settings `"remote.SSH.useExecServer": false` |
