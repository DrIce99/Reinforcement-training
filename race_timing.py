# race_timing.py
"""
Cronometraggio della gara in stile MotoGP.

Ogni gate è un rilevamento intermedio: quando un pilota lo attraversa si registra il tempo.
Il distacco tra due piloti è la differenza dei loro tempi allo stesso rilevamento, come
fanno i sistemi di cronometraggio reali. Da qui si ricavano:
- posizioni in tempo reale (con isteresi, così due piloti appaiati non si "sorpassano" a ogni frame);
- intervallo con chi precede e distacco dal leader;
- sorpassi;
- tendenza dell'intervallo, misurata solo a fine settore (il giro è diviso in 4 settori di uguale
  lunghezza): chi si avvicina a chi lo precede e chi allunga su chi lo segue.
"""
import math
from collections import deque

STEP_SECONDS = 1 / 120        # Un passo di simulazione = 1/120 s: a velocità x1 la gara scorre in tempo reale
POSITION_HYSTERESIS = 0.3     # Vantaggio minimo (frazione del tratto tra due gate) per cambiare posizione
OVERTAKE_MIN_SPLITS = 3       # Niente avvisi di sorpasso nella confusione della partenza
OVERTAKE_CONFIRM_STEPS = 60   # Un sorpasso viene annunciato solo se regge per 0.5 s (niente scambi avanti e indietro)
SECTOR_COUNT = 4              # Settori in cui è diviso il giro
TREND_EVENT = 0.3             # Variazione in un settore (s) che fa scattare l'avviso "si avvicina" / "allunga"
TREND_MAX_INTERVAL = 3.0      # Avvisi solo per piloti a meno di 3 s: lotte per la posizione
TREND_MIN_INTERVAL = 0.1      # Sotto 0.1 s sono appaiati: conta il sorpasso, non il distacco
TREND_SHOWN = 0.05            # Variazione minima per mostrare la freccia di tendenza in classifica


class TimingEvent:
    """
    kind: "sorpasso" (ahead ha appena passato behind), "avvicina" (behind recupera su ahead),
          "allunga" (ahead aumenta il vantaggio su behind).
    Le posizioni sono quelle al momento dell'evento.
    """

    def __init__(self, kind, ahead, behind, ahead_pos, behind_pos, interval=None, change=None, sector=None):
        self.kind = kind
        self.sector = sector
        self.ahead = ahead
        self.behind = behind
        self.ahead_pos = ahead_pos
        self.behind_pos = behind_pos
        self.interval = interval
        self.change = change


def compute_sector_ends(gates, count=SECTOR_COUNT):
    """
    Divide il giro in `count` settori di lunghezza (quasi) uguale.
    Ritorna l'indice del gate che chiude ogni settore; l'ultimo è il traguardo.
    """
    n = len(gates)
    if n < count:
        return list(range(n))
    centers = gates.centers
    steps = [math.hypot(*(centers[j] - centers[j - 1])) for j in range(1, n)]
    cumulative = [0.0]
    for d in steps:
        cumulative.append(cumulative[-1] + d)
    total = cumulative[-1] + math.hypot(*(centers[0] - centers[-1]))
    ends = []
    for s in range(1, count):
        target = total * s / count
        j = next(i for i, c in enumerate(cumulative) if c >= target)
        ends.append(max(j, ends[-1] + 1 if ends else 0))
    ends.append(n - 1)
    return ends


class RaceTiming:
    def __init__(self, racers, gates, laps):
        self.racers = list(racers)
        self.gates = gates
        self.laps = laps
        self.sector_ends = compute_sector_ends(gates)
        self.order = list(racers)                      # Ordine mostrato in classifica
        self.track_pos = {r.id: 0.0 for r in racers}   # Rilevamenti passati + frazione del tratto in corso
        self.seen_splits = {r.id: 0 for r in racers}
        self.history = {r.id: deque(maxlen=16) for r in racers}  # A fine settore: (rilevamento, id di chi precede, intervallo)
        self.trend = {r.id: None for r in racers}      # Variazione dell'intervallo nell'ultimo settore (s)
        self.pending_overtakes = {}                    # (chi sorpassa, chi è sorpassato) -> passo del sorpasso
        self.step = 0
        self.events = []

    # --- POSIZIONI ---
    def _update_track_pos(self, r):
        if not r.alive or r.completed:
            return
        n = len(self.gates)
        prev_c = self.gates.centers[(r.next_cp - 1) % n]
        next_c = self.gates.centers[r.next_cp % n]
        # Avanzamento nel settore: proiezione lungo la direzione del tracciato, così la traiettoria
        # (interno o esterno curva) non cambia la posizione in classifica
        sx, sy = next_c[0] - prev_c[0], next_c[1] - prev_c[1]
        seg2 = sx * sx + sy * sy
        proj = ((r.pos.x - prev_c[0]) * sx + (r.pos.y - prev_c[1]) * sy) / seg2 if seg2 > 0 else 0.0
        frac = min(1.0, max(0.0, proj))
        self.track_pos[r.id] = len(r.splits) + frac

    def _key(self, r):
        """Arrivati (in ordine d'arrivo) > in gara (per posizione in pista) > ritirati."""
        if r.completed:
            return (2, -r.finish_time)
        if r.alive:
            return (1, self.track_pos[r.id])
        return (0, self.track_pos[r.id])

    def _should_swap(self, upper, lower):
        ku, kl = self._key(upper), self._key(lower)
        if ku[0] != kl[0]:
            return kl[0] > ku[0]
        if ku[0] == 1:
            return kl[1] > ku[1] + POSITION_HYSTERESIS
        return kl[1] > ku[1]

    def position(self, r):
        return self.order.index(r) + 1

    def sector_of_gate(self, gate_index):
        return next(s for s, end in enumerate(self.sector_ends, start=1) if gate_index <= end)

    def current_sector(self, r):
        """Settore in cui si trova il pilota (quello del prossimo gate da attraversare)."""
        return self.sector_of_gate(r.next_cp % len(self.gates))

    def update(self):
        """Da chiamare dopo ogni passo di simulazione."""
        self.step += 1
        for r in self.racers:
            self._update_track_pos(r)

        # Ordinamento a bolle sull'ordine precedente: ogni scambio è un cambio di posizione
        for _ in range(len(self.order)):
            swapped = False
            for i in range(len(self.order) - 1):
                upper, lower = self.order[i], self.order[i + 1]
                if not self._should_swap(upper, lower):
                    continue
                self.order[i], self.order[i + 1] = lower, upper
                swapped = True
                racing = all(x.alive and not x.completed for x in (upper, lower))
                if racing and len(lower.splits) >= OVERTAKE_MIN_SPLITS:
                    if (upper.id, lower.id) in self.pending_overtakes:
                        # Ripassato subito: nessuno dei due sorpassi viene annunciato
                        del self.pending_overtakes[(upper.id, lower.id)]
                    else:
                        self.pending_overtakes[(lower.id, upper.id)] = self.step
            if not swapped:
                break
        self._confirm_overtakes()

        for idx, r in enumerate(self.order):
            k = len(r.splits)
            if k != self.seen_splits[r.id]:
                self.seen_splits[r.id] = k
                self._new_split(idx, r, k)

    def _confirm_overtakes(self):
        by_id = {r.id: r for r in self.racers}
        for pair, step in list(self.pending_overtakes.items()):
            if self.step - step < OVERTAKE_CONFIRM_STEPS:
                continue
            del self.pending_overtakes[pair]
            ahead, behind = by_id[pair[0]], by_id[pair[1]]
            racing = all(x.alive and not x.completed for x in (ahead, behind))
            ahead_pos, behind_pos = self.position(ahead), self.position(behind)
            if racing and ahead_pos < behind_pos:
                self.events.append(TimingEvent("sorpasso", ahead=ahead, behind=behind,
                                               ahead_pos=ahead_pos, behind_pos=behind_pos))

    # --- DISTACCHI ---
    def split_gap(self, ahead, behind):
        """
        Distacco (s) di `behind` da `ahead` all'ultimo rilevamento di `behind`.
        Ritorna ("tempo", secondi), ("giri", n) se doppiato, oppure None se non ancora misurabile.
        """
        k = len(behind.splits)
        n = len(self.gates)
        laps_down = (len(ahead.splits) - k) // n if n else 0
        if laps_down >= 1:
            return ("giri", laps_down)
        if k == 0 or len(ahead.splits) < k:
            return None
        # Mai negativo: chi è dietro in classifica può essere passato prima al rilevamento per pochi passi
        return ("tempo", max(0.0, (behind.splits[k - 1] - ahead.splits[k - 1]) * STEP_SECONDS))

    def interval(self, r):
        """Distacco da chi precede in classifica."""
        idx = self.order.index(r)
        return None if idx == 0 else self.split_gap(self.order[idx - 1], r)

    def gap_to_leader(self, r):
        idx = self.order.index(r)
        return None if idx == 0 else self.split_gap(self.order[0], r)

    def _new_split(self, idx, r, k):
        """
        Nuovo rilevamento di `r`. Solo se chiude un settore si confronta l'intervallo con chi lo precede
        con quello di fine settore precedente: un solo aggiornamento per settore, niente raffiche di avvisi.
        """
        gate_index = (k - 1) % len(self.gates)
        if gate_index not in self.sector_ends:
            return
        sector = self.sector_of_gate(gate_index)
        if idx == 0 or not r.alive:
            self.trend[r.id] = None
            return
        ahead = self.order[idx - 1]
        gap = self.split_gap(ahead, r)
        if gap is None or gap[0] != "tempo":
            self.trend[r.id] = None
            return
        interval = gap[1]
        hist = self.history[r.id]
        previous = hist[-1] if hist else None
        hist.append((k, ahead.id, interval))

        # Confronto con la fine del settore precedente, solo se davanti c'era lo stesso pilota
        if previous is None or previous[1] != ahead.id:
            self.trend[r.id] = None
            return
        change = interval - previous[2]
        self.trend[r.id] = change

        if abs(change) >= TREND_EVENT and TREND_MIN_INTERVAL <= interval <= TREND_MAX_INTERVAL and ahead.alive:
            kind = "avvicina" if change < 0 else "allunga"
            self.events.append(TimingEvent(kind, ahead=ahead, behind=r, ahead_pos=idx, behind_pos=idx + 1,
                                           interval=interval, change=change, sector=sector))

    def pop_events(self):
        events, self.events = self.events, []
        return events

    def final_order(self):
        return sorted(self.racers, key=self._key, reverse=True)

    def leader_lap_text(self):
        leader = self.order[0]
        if leader.completed:
            return "BANDIERA A SCACCHI"
        lap = min(leader.laps + 1, self.laps)
        return "ULTIMO GIRO" if lap == self.laps and self.laps > 1 else f"GIRO {lap}/{self.laps}"
