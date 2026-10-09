# track.py
# Funzioni condivise su piste, gate (checkpoint) e navigazione,
# usate da create_track.py, generate_track.py, training.py e simulation.py.
import math
import os
import pickle

import numpy as np
import pygame

TRACK_WIDTH = 75
GATE_SPACING = 100                     # Distanza tra due gate lungo il tracciato (px)
GATE_WIDTH = TRACK_WIDTH - 10
NEIGHBOR_ARC = 4 * TRACK_WIDTH         # Oltre questa distanza lungo il tracciato un punto è "un altro tratto"
CROSSING_CLEARANCE = 1.5 * TRACK_WIDTH # Nessun gate così vicino a un altro tratto di pista (incroci)

SENSOR_COUNT = 5
NAV_INPUTS = 2                         # Navigatore: seno e coseno dell'angolo verso i prossimi gate
INPUT_COUNT = SENSOR_COUNT + NAV_INPUTS


def track_image_path(name):
    return f"{name}.png"


def track_config_path(name):
    return os.path.join("tracks_config", f"{name}.pkl")


# --- GEOMETRIA DEL TRACCIATO ---
class TrackPath:
    """Linea centrale della pista parametrizzata per lunghezza d'arco."""

    def __init__(self, points, closed=True):
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        # Rimuove i punti doppi (es. giunzioni tra curve di Bézier)
        keep = np.ones(len(pts), dtype=bool)
        keep[1:] = np.hypot(*np.diff(pts, axis=0).T) > 1e-6
        pts = pts[keep]
        if closed and len(pts) > 1 and np.hypot(*(pts[-1] - pts[0])) < 1e-6:
            pts = pts[:-1]

        self.closed = closed
        self.points = pts
        self.ext = np.vstack([pts, pts[:1]]) if closed else pts
        seg = np.diff(self.ext, axis=0)
        self.cum = np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
        self.length = float(self.cum[-1])
        self.point_arc = self.cum[:len(pts)]

    def point(self, s):
        s = s % self.length if self.closed else min(max(s, 0.0), self.length)
        return np.array([np.interp(s, self.cum, self.ext[:, 0]), np.interp(s, self.cum, self.ext[:, 1])])

    def direction(self, s, h=3.0):
        d = self.point(s + h) - self.point(s - h)
        n = math.hypot(d[0], d[1])
        return d / n if n > 0 else np.array([1.0, 0.0])

    def clearance(self, s):
        """Distanza dai tratti di pista lontani lungo il tracciato (es. l'altra strada di un incrocio)."""
        arc = np.abs(self.point_arc - s)
        if self.closed:
            arc = np.minimum(arc, self.length - arc)
        far = arc > NEIGHBOR_ARC
        if not far.any():
            return math.inf
        return float(np.hypot(*(self.points[far] - self.point(s)).T).min())

    def gate_at(self, s, width=GATE_WIDTH):
        p = self.point(s)
        d = self.direction(s)
        normal = np.array([-d[1], d[0]]) * (width / 2)
        a, b = p + normal, p - normal
        return ((float(a[0]), float(a[1])), (float(b[0]), float(b[1])))


def build_checkpoint_gates(points, closed=True, spacing=GATE_SPACING, start=0.0, finish_offset=None):
    """
    Gate perpendicolari al tracciato ogni `spacing` px, a partire dall'ascissa `start`.
    Con `finish_offset` (piste chiuse) l'ultimo gate è il traguardo, `finish_offset` px prima del punto 0.
    I gate vicini a un altro tratto di pista vengono saltati: in un incrocio verrebbero
    attraversati anche da chi percorre l'altra strada, falsando il controllo del percorso.
    """
    path = TrackPath(points, closed)
    if path.length < spacing:
        return []

    end = path.length
    if finish_offset is not None:
        end -= finish_offset + spacing / 2
    elif closed:
        end -= spacing / 2  # Evita due gate attaccati a cavallo del punto di partenza

    gates = [path.gate_at(s) for s in np.arange(start, end, spacing)
             if path.clearance(s) >= CROSSING_CLEARANCE]
    if finish_offset is not None:
        gates.append(path.gate_at(path.length - finish_offset))
    return gates


def order_checkpoints_from_spawn(checkpoints, spawn_pos, base_angle):
    """
    Riordina i gate in modo che seguano il verso di marcia (base_angle) e che il primo
    sia quello subito davanti allo spawn. L'editor li genera partendo dal primo punto
    della pista, che in generale non coincide con lo spawn.
    """
    checkpoints = list(checkpoints)
    n = len(checkpoints)
    if n < 2:
        return checkpoints

    centers = [(pygame.Vector2(p1) + pygame.Vector2(p2)) / 2 for p1, p2 in checkpoints]
    rad = math.radians(base_angle)
    heading = pygame.Vector2(math.cos(rad), math.sin(rad))

    nearest = min(range(n), key=lambda i: centers[i].distance_squared_to(spawn_pos))
    if nearest < n - 1:
        track_dir = centers[nearest + 1] - centers[nearest]
    else:
        track_dir = centers[nearest] - centers[nearest - 1]

    # Pista disegnata nel verso opposto a quello di partenza
    if track_dir.dot(heading) < 0:
        checkpoints.reverse()
        centers.reverse()
        nearest = n - 1 - nearest

    # Se lo spawn ha già superato il gate più vicino si parte dal successivo
    if (spawn_pos - centers[nearest]).dot(heading) > 0:
        nearest = (nearest + 1) % n

    return checkpoints[nearest:] + checkpoints[:nearest]


# --- GATE IN GARA ---
class GateSet:
    """Gate in ordine di percorrenza, con controllo vettoriale degli attraversamenti."""

    def __init__(self, gates):
        self.gates = [((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))) for a, b in gates]
        arr = np.array(self.gates, dtype=float).reshape(-1, 2, 2)
        self.a = arr[:, 0]
        self.e = arr[:, 1] - arr[:, 0]
        self.centers = arr.mean(axis=1)
        # Punto verso cui punta il navigatore: a metà tra il prossimo gate e quello successivo,
        # così la direzione non oscilla quando si è vicini al gate
        self.targets = (self.centers + np.roll(self.centers, -1, axis=0)) / 2

    def __len__(self):
        return len(self.gates)

    def __getitem__(self, i):
        return self.gates[i]

    def crossed(self, p1, p2):
        """Indici dei gate attraversati dal segmento p1 -> p2."""
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        apx, apy = self.a[:, 0] - p1[0], self.a[:, 1] - p1[1]
        ex, ey = self.e[:, 0], self.e[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = dx * ey - dy * ex
            t = (apx * ey - apy * ex) / denom   # posizione lungo lo spostamento
            u = (apx * dy - apy * dx) / denom   # posizione lungo il gate
            hit = (denom != 0) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
        return np.flatnonzero(hit)

    def check_progress(self, p1, p2, next_cp):
        """
        Ritorna (prossimo_gate_superato, strada_sbagliata).
        Attraversare un gate diverso dal prossimo significa aver preso la strada sbagliata
        a un incrocio, aver tagliato la pista o andare contromano. Il gate appena superato
        è tollerato (partenza leggermente arretrata), quelli prima no.
        """
        n = len(self.gates)
        if n == 0:
            return False, False
        hits = self.crossed(p1, p2)
        if len(hits) == 0:
            return False, False
        target = next_cp % n
        previous = (next_cp - 1) % n
        passed = bool(np.any(hits == target))
        wrong = bool(np.any((hits != target) & (hits != previous)))
        return passed, wrong


def navigation_inputs(pos, angle, gates, next_cp):
    """Direzione verso i prossimi gate rispetto al muso dell'auto: (seno, coseno)."""
    if len(gates) == 0:
        return np.array([0.0, 1.0])
    tx, ty = gates.targets[next_cp % len(gates)]
    rel = math.atan2(ty - pos[1], tx - pos[0]) - math.radians(angle)
    return np.array([math.sin(rel), math.cos(rel)])


def adapt_weights(weights):
    """
    Adatta pesi salvati con un numero diverso di ingressi (es. modelli addestrati prima
    del navigatore): le righe mancanti partono da zero, quindi il comportamento iniziale
    resta identico e l'evoluzione impara gradualmente a usare i nuovi ingressi.
    """
    w = np.asarray(weights, dtype=float)
    if w.shape[0] == INPUT_COUNT:
        return w.copy()
    adapted = np.zeros((INPUT_COUNT, w.shape[1]))
    rows = min(INPUT_COUNT, w.shape[0])
    adapted[:rows] = w[:rows]
    return adapted


# --- CARICAMENTO / SALVATAGGIO ---
def load_track(name):
    """
    Ritorna (immagine non convertita, spawn_pos, base_angle, GateSet).
    base_angle è in coordinate schermo (y verso il basso).
    """
    image = pygame.image.load(track_image_path(name))
    with open(track_config_path(name), "rb") as f:
        config = pickle.load(f)
    spawn_pos = pygame.Vector2(config["spawn_pos"])
    # Il file salva l'angolo con l'asse y verso l'alto (convenzione dell'editor)
    base_angle = -config["base_angle"]
    gates = config["checkpoints"]
    if not config.get("gates_ordered", False):
        gates = order_checkpoints_from_spawn(gates, spawn_pos, base_angle)
    return image, spawn_pos, base_angle, GateSet(gates)


def save_track(name, surface, spawn_pos, base_angle_math, gates, gates_ordered=False, **extra):
    """base_angle_math: angolo di partenza con l'asse y verso l'alto (convenzione dell'editor)."""
    os.makedirs("tracks_config", exist_ok=True)
    pygame.image.save(surface, track_image_path(name))
    config = {
        "checkpoints": gates,
        "spawn_pos": (float(spawn_pos[0]), float(spawn_pos[1])),
        "base_angle": float(base_angle_math),
        "gates_ordered": gates_ordered,
    }
    config.update(extra)
    with open(track_config_path(name), "wb") as f:
        pickle.dump(config, f)
