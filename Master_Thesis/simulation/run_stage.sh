#!/bin/bash
# Batch-Runner fuer eine Sub-Stage des finalen Schedules: EIN GPU-Ollama-Server,
# NREP Runs von main_network.py sequentiell dagegen (Modell bleibt geladen).
#
# Die Sub-Stage kommt beim Absetzen mit; editiert wird nur die Praeambel (Screen-
# Ergebnisse). Die Tabelle setzt alle SIM_*-Parameter, weil config.py erst beim
# Jobstart gelesen wird. Begruendung jedes Werts: analysis/final_schedule.md; bei
# Widerspruch gilt diese Tabelle.
#
# Absetzen:  S=1-04; qsub -v SUBSTAGE=$S -N substage_$S ~/projects/thesis/simulation/run_stage.sh
#   r=10:    zweiter Job derselben Sub-Stage mit -v SUBSTAGE=$S,REP_OFFSET=5
#   Teiljob: -v SUBSTAGE=$S,REP_OFFSET=<n>,NREP_JOB=<k> rechnet r=n+1..n+k (0-01)
# Verfolgen: tail -f /data/scc/jonas.schrade/thesis-runs/substage_<S>.o<jobid>
#
# STAGE=<wert> bleibt veralteter Alias fuer SUBSTAGE, fuer bereits eingereihte Jobs.

#$ -q gpu@scc214
#$ -l h_vmem=96G,rtx_6000=1,gpu=1
#$ -l h_rt=168:00:00
#$ -N thesis_stage
#$ -o /data/scc/jonas.schrade/thesis-runs/ -j y

set -euo pipefail
RUNDIR=/data/scc/jonas.schrade/thesis-runs

if [ -n "${SUBSTAGE:-}" ]; then
  :
elif [ -n "${STAGE:-}" ]; then
  echo "HINWEIS: STAGE=$STAGE ist veraltet — bitte SUBSTAGE=$STAGE verwenden."
  SUBSTAGE="$STAGE"
else
  echo "ABBRUCH: SUBSTAGE nicht gesetzt."
  echo "  S=1-04; qsub -v SUBSTAGE=\$S -N substage_\$S ~/projects/thesis/simulation/run_stage.sh"
  exit 1
fi

REP_OFFSET="${REP_OFFSET:-0}"   # zweiter Job einer r=10-Sub-Stage: REP_OFFSET=5

# ===================== Betriebspunkte (Praeambel) =========================
# Leitersprossen. Die Kopplungsachse ist eine EXAKTE Kantenzahl, keine
# Wahrscheinlichkeit: N=20 liegt fest, also 100 Cross-Paare und 45 Paare je
# Block. L0/L1/L6 sind exakt, L2..L5 kommen aus Screen A-01.
M_INTRA=27   # Kanten JE BLOCK, von 45. Dichte exakt 0.6 -> L6 geht ohne Rest auf.
L0=0         # keine Cross-Kante, zwei getrennte Bloecke. Deckel f(0). Nie ein Anker.
L1=1         # genau EINE Cross-Kante: Kopplungs-Untergrenze.
L2=9         # \
L3=10        #  > klammern p_half des binaeren Nullmodells (A-01: 10.09)
L4=11        # /
L5=20        # Boden-Check: letzte Kopplung mit Nullmodell-Fraktion > 0 (A-01)
L6=60        # Cross-Dichte = Block-Dichte (60/100 = 27/45): ER-Nullmodell.
FIT_RUNGS="$L2 $L3 $L4 $L5"       # nur diese gehen in den p_half-Fit ein
ANCHOR_RUNGS="$L1 $FIT_RUNGS $L6" # L0 ausgeschlossen: keine Cross-Exposure

# Anker: die beiden Leiterenden plus die Schwelle. Muessen Sprossen sein, sonst
# verliert jeder daran verankerte Arm still seinen kostenlosen r=10-Kontrollarm.
P_LO=$L1
P_MED=$L3    # Sprosse am naechsten an p_half des graded-Nullmodells — PLATZHALTER (C-00)
P_HI=$L6

# Ausdrucks-Beta-Gitter (Stage 2) und die Sprosse, an der es laeuft.
BETA_LO=1.0  # Einsatzpunkt          \
BETA_MID=2.0 # steilste Antwort       > PLATZHALTER (C-03)
BETA_HI=3.0  # Anschluss an inf      /
P_BETA=$P_MED

# Rewiring: Zeitskala w (Regel R2) und die Kopplung R1 (Regel R1), beide aus
# EINEM Screen, D-01 (Kopplung x w). EIN w fuer Stage 6 UND 7-01/7-02, sonst ist
# die Stage-6-Sprosse bei R1 kein gueltiger Vergleichsarm mehr. 7-03 haelt w=1.0
# fest, unabhaengig von R2.
W=3.0        # PLATZHALTER (D-01)
W_GRID="1.0 1.5 2.0 3.0"
P_R1=$L1     # PLATZHALTER (D-01); R1 ist auf L1..L5 beschraenkt

# Screen-Gates: 0 = Platzhalter noch nicht durch das Screen-Ergebnis ersetzt.
# Jede Sub-Stage deklariert unten, welche sie braucht (NEED_SCREEN); der Job
# bricht ab statt einen Platzhalter zu rechnen. Ein Platzhalter-Run erzeugt ein
# vollstaendiges, plausibles Log, und nichts in params.json sagt hinterher,
# dass der Wert nur ein Kandidat war.
SCREENED_LADDER=1   # A-01  -> L2..L5
SCREENED_MED=0      # C-00  -> P_MED
SCREENED_BETA=0     # C-03  -> BETA_LO/MID/HI, P_BETA
SCREENED_D2=0       # D-01 unter R0-R3 -> P_R1, W

# Defaults = "Fixed"-Block von Bracket A. JEDER sweepbare Parameter wird gesetzt,
# auch wenn config.py denselben Default hat: sonst haengt die Physik eines Jobs
# am Stand des Checkouts zum Jobstart.
export SIM_SBM_M_INTRA=$M_INTRA
export SIM_SBM_M_INTER=$L1
export SIM_OPINION_BETA=inf
export SIM_RESPONDER_SELECTION_BETA=0.0
export SIM_GRAPH_DYNAMIC=0
export SIM_REWIRE_RULE=reward_local
export SIM_REWIRE_BETA=5.0
export SIM_REWIRE_TIMESCALE_RATIO=$W
export SIM_MEMORY_ENABLED=1
export SIM_LLM_MODEL=qwen3.8-nodraft:27b   # Default-Modell; nur Stage 8 weicht ab (-nodraft: RUNBOOK § Modelle pullen)

# ========================= Sub-Stage-Tabelle ==============================
case "$SUBSTAGE" in

  # ---- Screen A-00 / Stage 0: Horizont-Sonde ------------------------------
  # Setzt NETWORK_MAX_ROUNDS fuer den ganzen Schedule: L3, 5 Replikate, H=400
  # (Hand-Edit in config.py, NEED_HORIZON prueft ihn). Hoechstens zwei Replikate
  # je Job, per NREP_JOB (Guard 8; final_schedule.md § Screen A-00). Kein Arm:
  # aus der 36-Punkte-Pruefung ausgenommen.
  0-01) export SIM_SBM_M_INTER=$L3; NREP=5; NEED_HORIZON=400; NEED_SCREEN="LADDER" ;;

  # ---- Bracket A / Stage 1: RQ1-Leiter, aufsteigend in m_inter -------------
  # Jede r=10-Sub-Stage wird ZWEIMAL abgesetzt, der zweite Job mit REP_OFFSET=5.
  # 1-05 laeuft mit r=5, also nur EIN Job. 1-00 und 1-06 sind Grenzen, keine
  # Punkte auf der Kurve (FIT_RUNGS).
  1-00) export SIM_SBM_M_INTER=$L0; NREP=5 ;;
  1-01) export SIM_SBM_M_INTER=$L1; NREP=5 ;;
  1-02) export SIM_SBM_M_INTER=$L2; NREP=5; NEED_SCREEN="LADDER" ;;
  1-03) export SIM_SBM_M_INTER=$L3; NREP=5; NEED_SCREEN="LADDER" ;;
  1-04) export SIM_SBM_M_INTER=$L4; NREP=5; NEED_SCREEN="LADDER" ;;
  1-05) export SIM_SBM_M_INTER=$L5; NREP=5; NEED_SCREEN="LADDER" ;;
  1-06) export SIM_SBM_M_INTER=$L6; NREP=5 ;;

  # ---- Bracket C (2) / Stage 2: Ausdrucks-Beta -----------------------------
  # EINE Kopplung ($P_BETA): die Kopplungsabhaengigkeit von beta klaert Screen
  # C-03 auf 300 Reps, die GPU-Zeit kauft dafuer Power je Sprosse. Bracket A
  # liefert bei gleichem beta KEINEN kostenlosen Vergleichsarm (dort inf);
  # BETA_HI ist der Anschluss daran — ein Nullmodell-Ergebnis, kein LLM-Ergebnis.
  2-01) export SIM_SBM_M_INTER=$P_BETA; export SIM_OPINION_BETA=$BETA_LO;  NREP=5; NEED_SCREEN="LADDER BETA" ;;
  2-02) export SIM_SBM_M_INTER=$P_BETA; export SIM_OPINION_BETA=$BETA_MID; NREP=5; NEED_SCREEN="LADDER BETA" ;;
  2-03) export SIM_SBM_M_INTER=$P_BETA; export SIM_OPINION_BETA=$BETA_HI;  NREP=5; NEED_SCREEN="LADDER BETA" ;;

  # (beta_sel laeuft nur noch als CPU-Screen C-02 — kein Sub-Stage-Slot mehr.)

  # ---- Bracket C (2) / Stage 3: Stance-Flip, Gegenposition ----------------
  # Gleiches Topic, Gegenposition als Proposition (config.TOPIC_TEXT_OPPOSITE) —
  # kommt beim Absetzen als SIM_TOPIC_TEXT, sonst Abbruch. Alle drei Anker;
  # 3-03 braucht die ER-Sprosse, weil die Konsensrichtung nur dort fuer JEDE
  # Replikate definiert ist.
  3-01) export SIM_SBM_M_INTER=$P_LO;  NREP=5; NEED_TOPIC=opposite ;;
  3-02) export SIM_SBM_M_INTER=$P_MED; NREP=5; NEED_TOPIC=opposite; NEED_SCREEN="LADDER MED" ;;
  3-03) export SIM_SBM_M_INTER=$P_HI;  NREP=5; NEED_TOPIC=opposite ;;

  # ---- Bracket C (2) / Stage 4: Memory-Ablation ----------------------------
  # Alle drei Anker, r=5 je Anker (ein Job, kein REP_OFFSET). Kontrollarm ist je
  # Anker die Stage-1-Sprosse dort. EXPLORATIV: die Obergrenze setzt der
  # Kontrollarm mit r=10, nicht der Behandlungsarm (80 % erst ab d~1.1).
  4-01) export SIM_SBM_M_INTER=$P_LO;  export SIM_MEMORY_ENABLED=0; NREP=5 ;;
  4-02) export SIM_SBM_M_INTER=$P_MED; export SIM_MEMORY_ENABLED=0; NREP=5; NEED_SCREEN="LADDER MED" ;;
  4-03) export SIM_SBM_M_INTER=$P_HI;  export SIM_MEMORY_ENABLED=0; NREP=5 ;;

  # ---- Bracket C (2) / Stage 5: zwei weitere Topics ------------------------
  # ANDERE Propositionen, fest in der Tabelle (SIM_TOPIC aus config.TOPICS):
  # 5-01..03 ein stark polarisiertes Energie-Topic, 5-04..06 eine kaum
  # polarisierte, fast beliebige Dichotomie (final_schedule.md § Bracket C,
  # Stage 5). 5-03/5-06 brauchen die ER-Sprosse wie 3-03.
  5-01) export SIM_SBM_M_INTER=$P_LO;  export SIM_TOPIC=renewables; NREP=5; NEED_TOPIC=topic ;;
  5-02) export SIM_SBM_M_INTER=$P_MED; export SIM_TOPIC=renewables; NREP=5; NEED_TOPIC=topic; NEED_SCREEN="LADDER MED" ;;
  5-03) export SIM_SBM_M_INTER=$P_HI;  export SIM_TOPIC=renewables; NREP=5; NEED_TOPIC=topic ;;
  5-04) export SIM_SBM_M_INTER=$P_LO;  export SIM_TOPIC=coffee_tea; NREP=5; NEED_TOPIC=topic ;;
  5-05) export SIM_SBM_M_INTER=$P_MED; export SIM_TOPIC=coffee_tea; NREP=5; NEED_TOPIC=topic; NEED_SCREEN="LADDER MED" ;;
  5-06) export SIM_SBM_M_INTER=$P_HI;  export SIM_TOPIC=coffee_tea; NREP=5; NEED_TOPIC=topic ;;

  # ---- Bracket D (D1) / Stage 6: dynamische Leiter -------------------------
  # Dieselben Sprossen L1..L5 wie Stage 1, aber endogener Graph — D1 wird gegen
  # Stage 1 an DERSELBEN Kopplung gelesen. Genau EINE dieser Sub-Stages liegt auf
  # R1's Kopplung und traegt zugleich die beta_rw=5-Seite des Scoring-Kontrasts
  # gegen 7-02; WELCHE, entscheidet Screen D-01. Nicht vorab auf 6-01 festlegen.
  6-01) export SIM_SBM_M_INTER=$L1; export SIM_GRAPH_DYNAMIC=1; NREP=5; NEED_SCREEN="D2" ;;
  6-02) export SIM_SBM_M_INTER=$L2; export SIM_GRAPH_DYNAMIC=1; NREP=5; NEED_SCREEN="LADDER D2" ;;
  6-03) export SIM_SBM_M_INTER=$L3; export SIM_GRAPH_DYNAMIC=1; NREP=5; NEED_SCREEN="LADDER D2" ;;
  6-04) export SIM_SBM_M_INTER=$L4; export SIM_GRAPH_DYNAMIC=1; NREP=5; NEED_SCREEN="LADDER D2" ;;
  6-05) export SIM_SBM_M_INTER=$L5; export SIM_GRAPH_DYNAMIC=1; NREP=3; NEED_SCREEN="LADDER D2" ;;

  # ---- Bracket D (D2) / Stage 7: Reichweite vs. Triadenschluss -------------
  # 7-01/7-02 sind der Mechanismus-Test: beide Arme bewerten NICHT (beta_rw=0),
  # der einzige Unterschied ist der 2-Hop-Pool. NREP=6 ist die Budget-Obergrenze
  # je Job (zweiter Job mit REP_OFFSET=6 ergibt die Kappung r=12), nicht das
  # Ergebnis der Poweranalyse (Regel R3). 7-03 ist die w=1.0-Sonde, kein Test.
  7-01) export SIM_SBM_M_INTER=$P_R1; export SIM_GRAPH_DYNAMIC=1; export SIM_REWIRE_RULE=random;       export SIM_REWIRE_BETA=0.0; NREP=6; NEED_SCREEN="LADDER D2" ;;
  7-02) export SIM_SBM_M_INTER=$P_R1; export SIM_GRAPH_DYNAMIC=1; export SIM_REWIRE_RULE=reward_local; export SIM_REWIRE_BETA=0.0; NREP=6; NEED_SCREEN="LADDER D2" ;;
  7-03) export SIM_SBM_M_INTER=$P_R1; export SIM_GRAPH_DYNAMIC=1; export SIM_REWIRE_TIMESCALE_RATIO=1.0; NREP=3; NEED_SCREEN="LADDER D2" ;;

  # ---- Bracket E / Stage 8: Modell-Sensitivitaet ---------------------------
  # Drei Sprossen je Zusatzmodell (Boden / Schwelle / ER). Kontrollarme sind
  # 1-01, die Stage-1-Sub-Stage bei $P_MED und 1-06 — alle r=10 unter dem
  # Default-Modell, kosten also nichts. WELCHE Sub-Stage bei $P_MED liegt,
  # entscheidet erst C-00: nicht vorab auf 1-03 festlegen.
  # Die ER-Sprosse ist hier NICHT optional: nur dort ist die Konsensrichtung fuer
  # JEDE Replikate definiert, und genau die ist die Frage von Tier 0.
  8-01) export SIM_SBM_M_INTER=$P_LO;  export SIM_LLM_MODEL=devstral-small-2:24b; NREP=5 ;;
  8-02) export SIM_SBM_M_INTER=$P_MED; export SIM_LLM_MODEL=devstral-small-2:24b; NREP=5; NEED_SCREEN="LADDER MED" ;;
  8-03) export SIM_SBM_M_INTER=$P_HI;  export SIM_LLM_MODEL=devstral-small-2:24b; NREP=5 ;;
  8-04) export SIM_SBM_M_INTER=$P_LO;  export SIM_LLM_MODEL=gemma4-nodraft:26b;   NREP=5 ;;
  8-05) export SIM_SBM_M_INTER=$P_MED; export SIM_LLM_MODEL=gemma4-nodraft:26b;   NREP=5; NEED_SCREEN="LADDER MED" ;;
  8-06) export SIM_SBM_M_INTER=$P_HI;  export SIM_LLM_MODEL=gemma4-nodraft:26b;   NREP=5 ;;

  *) echo "ABBRUCH: unbekannte Sub-Stage '$SUBSTAGE'"; exit 1 ;;
esac

# =============================== Guards ===================================
# Jeder dieser Fehler faellt sonst erst auf, wenn der Vergleich Monate spaeter
# gemacht werden soll.

# 1. Screen-Gate: kein Platzhalter-Run. Unter DRY_RUN nur gemeldet, nicht
#    abgebrochen — check_substages.sh muss alle Sub-Stages aufloesen koennen,
#    auch solange die Platzhalter noch drinstehen.
OWED=""
for s in ${NEED_SCREEN:-}; do
  eval "flag=\${SCREENED_$s:-}"
  if [ "$flag" != "1" ]; then
    if [ -n "${DRY_RUN:-}" ]; then
      OWED="$OWED$s "
      continue
    fi
    echo "ABBRUCH: $SUBSTAGE braucht Screen-Ergebnis '$s' (SCREENED_$s=0)."
    echo "  Werte in der Praeambel eintragen, dann SCREENED_$s=1 setzen."
    echo "  LADDER=A-01, MED=C-00, BETA=C-03, D2=D-01 (Regeln R0-R3)."
    exit 1
  fi
done

# 1b. Horizont-Gate: eine Sub-Stage, die einen bestimmten NETWORK_MAX_ROUNDS
#     braucht, prueft ihn gegen config.py. Ohne Override kann der Wert nur per
#     Hand-Edit stimmen, und ein falscher Horizont faellt sonst erst in der
#     Auswertung auf.
#     Per sed, NICHT per python: vor dem Env-Setup ist `python` System-Python 2.x
#     und importiert config.py nicht (RUNBOOK § Fehlersuche).
if [ -n "${NEED_HORIZON:-}" ]; then
  if [ -n "${DRY_RUN:-}" ]; then
    OWED="$OWED horizon=$NEED_HORIZON "
  else
    HAVE_HORIZON="$(sed -n 's/^NETWORK_MAX_ROUNDS[[:space:]]*=[[:space:]]*\([0-9][0-9]*\).*/\1/p' \
                      ~/projects/thesis/simulation/config.py)"
    if [ -z "$HAVE_HORIZON" ]; then
      echo "ABBRUCH: NETWORK_MAX_ROUNDS steht in config.py nicht als einfaches Zahlen-Literal."
      echo "  Das Horizont-Gate liest es per sed (vor dem Env-Setup gibt es kein brauchbares python)."
      exit 1
    fi
    if [ "$HAVE_HORIZON" != "$NEED_HORIZON" ]; then
      echo "ABBRUCH: $SUBSTAGE braucht NETWORK_MAX_ROUNDS=$NEED_HORIZON, config.py hat $HAVE_HORIZON."
      echo "  Wert vor dem qsub in simulation/config.py eintragen (kein SIM_*-Override, absichtlich)."
      exit 1
    fi
  fi
fi

# 2. Anker und P_BETA muessen ankerfaehige Leitersprossen sein.
for a in "$P_LO" "$P_MED" "$P_HI" "$P_BETA"; do
  case " $ANCHOR_RUNGS " in
    *" $a "*) ;;
    *) echo "ABBRUCH: Anker $a ist keine ankerfaehige Sprosse ($ANCHOR_RUNGS)"; exit 1 ;;
  esac
done

# 3. R1 ist auf L1..L5 beschraenkt: bei L6 ist der Reichweiten-Kontrast ~0, bei
#    L0 maximal-aber-strukturell (kein gemeinsamer 2-Hop-Pfad).
case " $L1 $FIT_RUNGS " in
  *" $P_R1 "*) ;;
  *) echo "ABBRUCH: R1-Kopplung $P_R1 liegt nicht in L1..L5 ($L1 $FIT_RUNGS)"; exit 1 ;;
esac

# 4. beta-Gitter aufsteigend, sonst stimmt die Sub-Stage-Ordnung von 2-01..2-03 nicht.
if ! awk -v a="$BETA_LO" -v b="$BETA_MID" -v c="$BETA_HI" \
     'BEGIN { exit !(a < b && b < c) }'; then
  echo "ABBRUCH: beta-Gitter nicht aufsteigend ($BETA_LO $BETA_MID $BETA_HI)"; exit 1
fi

# 5. w muss ein gescreenter Gitterwert sein, sonst ist der gemeinsame
#    D1/D2-Betriebspunkt verloren.
case " $W_GRID " in
  *" $W "*) ;;
  *) echo "ABBRUCH: w=$W ist kein gescreenter Gitterwert ($W_GRID)"; exit 1 ;;
esac

# 5b. 7-03 ist die w=1.0-Sonde GEGEN den gefitteten w. Gibt D-01 selbst w=1.0
#     zurueck, ist sie kein Kontrast mehr, sondern exakt die Stage-6-Sprosse bei
#     $P_R1 — gleicher Parameterpunkt, doppelt bezahlt. check_substages.sh findet
#     das erst nach dem Screen; hier faellt es beim Absetzen auf.
if [ "$SUBSTAGE" = "7-03" ] && awk -v w="$W" 'BEGIN{exit !(w==1.0)}'; then
  echo "ABBRUCH: D-01 hat w=1.0 gefittet, also ist 7-03 identisch mit der"
  echo "  Stage-6-Sprosse bei P_R1=$P_R1 (6-0R1). Die Sonde entfaellt — 6-0R1"
  echo "  liest sie bereits. Nicht absetzen (De-Scoping-Leiter: 7-03 zuerst)."
  exit 1
fi

# 6. Was der Schedule offenlaesst, muss beim Absetzen kommen.
case "${NEED_TOPIC:-}" in
  opposite)
    if [ -z "${SIM_TOPIC_TEXT:-}" ]; then
      echo "ABBRUCH: $SUBSTAGE braucht die Gegenposition (config.TOPIC_TEXT_OPPOSITE)."
      echo "  qsub -v SUBSTAGE=$SUBSTAGE,SIM_TOPIC_TEXT='<Gegenposition>' ..."
      exit 1
    fi
    TOPIC_EXTRA="${SIM_TOPIC:-}${SIM_TOPIC_LABEL:-}${SIM_TOPIC_TEXT_OPPOSITE:-}" ;;
  topic)
    # Das Topic steht in der Tabelle (SIM_TOPIC); ein Text-Override ersetzte es still.
    TOPIC_EXTRA="${SIM_TOPIC_TEXT:-}${SIM_TOPIC_LABEL:-}${SIM_TOPIC_TEXT_OPPOSITE:-}" ;;
  *)
    TOPIC_EXTRA="${SIM_TOPIC:-}${SIM_TOPIC_TEXT:-}${SIM_TOPIC_LABEL:-}${SIM_TOPIC_TEXT_OPPOSITE:-}" ;;
esac
# Ein Topic-Override an der falschen Sub-Stage wuerde deren Topic still ersetzen.
if [ -n "$TOPIC_EXTRA" ]; then
  echo "ABBRUCH: $SUBSTAGE bekommt ein Topic-Override, das der Schedule nicht vorsieht."
  echo "  Stage 5 traegt ihr Topic in der Tabelle; nur Stage 3 nimmt SIM_TOPIC_TEXT."
  exit 1
fi

# 7. Ein fester Seed ueber alle Replikate eines Arms wuerde die Regime-Fraktion
#    auf EINEN Graphen bedingen (final_schedule.md § Seeding Regel 2). Ein
#    Seed-Block je Replikat ist der erlaubte Weg (Regel 3).
if [ -n "${SIM_SEED:-}" ]; then
  echo "ABBRUCH: SIM_SEED gilt fuer JEDEN Run dieses Jobs — das verletzt Regel 2."
  echo "  Fuer die Blockung nach Regel 3: SEED_BLOCK='s1 s2 ... sN' uebergeben."
  exit 1
fi
SEED_BLOCK="${SEED_BLOCK:-}"

# 8. Job-Splitting. NREP_JOB rechnet nur EINEN Teil der Replikate dieser
#    Sub-Stage, REP_OFFSET sagt welchen — fuer Sub-Stages, deren Replikate
#    zusammen h_rt sprengen (0-01 bei H=400). Es darf den Tabellenwert nur
#    verkleinern: die Tabelle bleibt der Datensatz darueber, aus wie vielen
#    Replikaten die Sub-Stage besteht, und r bleibt in 1..NREP.
if [ -n "${NREP_JOB:-}" ]; then
  case "$NREP_JOB" in ''|*[!0-9]*)
    echo "ABBRUCH: NREP_JOB muss eine positive ganze Zahl sein (war '$NREP_JOB')."; exit 1 ;;
  esac
  if [ "$NREP_JOB" -lt 1 ] || [ "$((REP_OFFSET + NREP_JOB))" -gt "$NREP" ]; then
    echo "ABBRUCH: NREP_JOB=$NREP_JOB mit REP_OFFSET=$REP_OFFSET laeuft ueber" \
         "NREP=$NREP von $SUBSTAGE hinaus."
    echo "  Die Tabelle legt die Replikatzahl fest; NREP_JOB teilt sie nur auf Jobs auf."
    exit 1
  fi
  NREP="$NREP_JOB"
fi

# DRY_RUN: eine maschinenlesbare Zeile mit dem aufgeloesten Parameterpunkt, dann
# Schluss — vor Lock, Ollama und jedem Run. Grundlage der Eindeutigkeitspruefung
# (check_substages.sh; final_schedule.md § Encoding).
if [ -n "${DRY_RUN:-}" ]; then
  echo "$SUBSTAGE NREP=$NREP TOPIC=${NEED_TOPIC:-none} OWED=${OWED:-none}" \
       "$(env | grep '^SIM_' | sort | tr '\n' ' ')"
  exit 0
fi

echo "Sub-Stage $SUBSTAGE  NREP=$NREP (Offset $REP_OFFSET)  Modell=$SIM_LLM_MODEL" \
     "m_intra=$SIM_SBM_M_INTRA/45  m_inter=$SIM_SBM_M_INTER/100" \
     "β=$SIM_OPINION_BETA  β_sel=$SIM_RESPONDER_SELECTION_BETA" \
     "dynamic=$SIM_GRAPH_DYNAMIC  rewire=$SIM_REWIRE_RULE  β_rw=$SIM_REWIRE_BETA  w=$SIM_REWIRE_TIMESCALE_RATIO" \
     "memory=$SIM_MEMORY_ENABLED  topic=${SIM_TOPIC:-<config.py>}${SIM_TOPIC_TEXT:+ (Text: $SIM_TOPIC_TEXT)}" \
     "seeds=${SEED_BLOCK:-<OS-Entropie>}"

# Schutz gegen Doppelläufe: Warten mehrere Jobs derselben Sub-Stage auf
# verschiedenen Knoten, würden sie sonst gleichzeitig in dieselben
# logs/run_<SUBSTAGE>_r*/ schreiben. mkdir ist atomar (auch über NFS).
# REP_OFFSET gehoert in den Lock-Namen: die zwei Jobs einer r=10-Sub-Stage
# schreiben disjunkte Replikate und muessen parallel laufen duerfen.
SUBSTAGE_LOCK="$RUNDIR/.substage_${SUBSTAGE}_o${REP_OFFSET}.lock"
mkdir "$SUBSTAGE_LOCK" 2>/dev/null || {
  echo "ABBRUCH: Sub-Stage $SUBSTAGE (Offset $REP_OFFSET) läuft bereits. Lock: $SUBSTAGE_LOCK"
  echo "Falls verwaist (Job hart abgebrochen): rmdir '$SUBSTAGE_LOCK'"
  exit 1
}
# Räumt Lock und Ollama-Daemon auf — auch bei qdel (SIGTERM → exit → EXIT-Trap).
trap 'kill "${OLLAMA_PID:-}" 2>/dev/null; rmdir "$SUBSTAGE_LOCK" 2>/dev/null' EXIT
trap 'exit 143' TERM INT

# setup_thesis_env.sh hängt an teils nicht gesetzte Variablen an (LD_LIBRARY_PATH)
# und stirbt deshalb unter 'set -u'. Nur fürs Sourcen abschalten.
set +u
source ~/setup_thesis_env.sh
set -u
which ollama | grep -q '/data/scc' || { echo "FALSCHE ollama-Binary!"; exit 1; }

# GPU-Zuteilung der starter_method übernehmen: sie reserviert die Karte (Lock
# unter /tmp/sge-gpu-lock-*) und legt ihren Index in SGE_GPU ab. Findet sie keine
# freie Karte, stirbt der Job schon vor dieser Zeile mit "Could only reserve
# 0 of 1 requested devices" (siehe RUNBOOK, "GPU-Zuteilung auf der gpu-Queue").
if [ -n "${SGE_GPU:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${SGE_GPU// /,}"
else
  # Fallback ohne starter_method: freie GPU wählen, Index 0 aus (Discovery-Falle)
  export CUDA_VISIBLE_DEVICES=$(nvidia-smi \
    --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '$1!=0 && $2<1000 {print $1; exit}')
  [ -n "$CUDA_VISIBLE_DEVICES" ] || { echo "Keine freie GPU (außer 0)!"; exit 1; }
fi
echo "Knoten $(hostname), GPU $CUDA_VISIBLE_DEVICES (SGE_GPU='${SGE_GPU:-unset}')"

# Eigener Port je Job: mehrere eigene Jobs teilen sich regelmaessig einen Knoten.
export OLLAMA_HOST=127.0.0.1:$((11500 + JOB_ID % 400))
# Sonst ein llama-server-Thread je Knotenkern auf den Slots des Jobs (RUNBOOK § Fehlersuche).
export LLAMA_ARG_THREADS=${NSLOTS:-1}
GGML_CUDA_NO_VMM=1 ollama serve > "$HOME/ollama_$JOB_ID.log" 2>&1 &
OLLAMA_PID=$!

# HARTER CPU-Fallback-Check: auf die Discovery-Zeile warten, nicht blind schlafen.
for _ in $(seq 1 60); do
  grep -q "inference compute" "$HOME/ollama_$JOB_ID.log" && break
  sleep 2
done
# Positiver Test: fehlende Evidenz ist kein Freibrief. Ollama schreibt
# 'library=CUDA' (GPU) bzw. 'library=cpu' (Discovery-Falle) — daher -i.
if ! grep -qi "library=cuda" "$HOME/ollama_$JOB_ID.log"; then
  echo "ABBRUCH: Ollama nicht auf CUDA (Discovery-Falle? GPU 0 des Knotens belegt?)."
  echo "  Siehe ollama_$JOB_ID.log. Auf einen Knoten mit freier GPU 0 ausweichen."
  kill "$OLLAMA_PID" 2>/dev/null || true; exit 1
fi
echo "Ollama auf GPU bereit."

cd "$RUNDIR"
# Replikate r = REP_OFFSET+1 .. REP_OFFSET+NREP. Der Offset ist, was den zweiten
# Job einer r=10-Sub-Stage davon abhaelt, run_<S>_r1..r5 zu ueberschreiben.
for r in $(seq $((REP_OFFSET + 1)) $((REP_OFFSET + NREP))); do
  if [ -n "$SEED_BLOCK" ]; then
    set -- $SEED_BLOCK
    if [ "$r" -gt "$#" ]; then
      echo "ABBRUCH: SEED_BLOCK hat nur $# Seeds, Replikat r$r braucht einen."; exit 1
    fi
    eval "export SIM_SEED=\${$r}"
    echo "=== Run $r (SIM_RUN_ID=${SUBSTAGE}_r${r}, seed=$SIM_SEED) ==="
  else
    echo "=== Run $r (SIM_RUN_ID=${SUBSTAGE}_r${r}) ==="
  fi
  if SIM_RUN_ID=${SUBSTAGE}_r${r} python -u ~/projects/thesis/simulation/main_network.py; then
    echo "Run $r OK"
  else
    echo "Run $r FEHLGESCHLAGEN (exit $?) — weiter mit nächstem"
  fi
done

echo "Sub-Stage $SUBSTAGE fertig."
