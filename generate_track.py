# generate_track.py
"""
Generatore automatico di piste per training.py e simulation.py.

Senza incroci la pista è una curva chiusa casuale (somma di armoniche). Con gli incroci
è una "cinghia" avvolta su una catena di cerchi, percorsa all'andata e al ritorno: ogni
volta che due cerchi consecutivi sono avvolti in verso opposto le due tangenti tra loro
si incrociano. Così il raggio minimo di curva è garantito e gli incroci cadono sui rettilinei.

Gli incroci sono quasi perpendicolari e su tratti dritti; i gate vicini agli incroci vengono
omessi e chi imbocca la strada sbagliata attraversa un gate fuori sequenza e viene eliminato,
quindi l'evoluzione premia chi segue il navigatore e tira dritto.

Uso:
    python generate_track.py                          anteprima interattiva
    python generate_track.py --nome pista_x --incroci 2 --seed 42 --salva

Comandi dell'anteprima:
    SPAZIO = nuova pista    0-2 = numero di incroci    R = incroci casuali
    INVIO = salva           ESC = esci
Poi: python training.py <nome>   e   python simulation.py <nome>
"""
import argparse
import math
import random

import numpy as np
import pygame

from track import (TRACK_WIDTH, GATE_SPACING, NEIGHBOR_ARC, TrackPath,
                   build_checkpoint_gates, save_track)
from window_state import restore_window_position, close_window

WIDTH, HEIGHT = 1600, 900
WINDOW_KEY = "generate_track"
DEFAULT_NAME = "pista_auto"

MARGIN = TRACK_WIDTH            # Distanza minima della linea centrale dal bordo dell'immagine
STEP = 4                        # Passo di campionamento del tracciato (px)
COARSE_EVERY = 3                # Per i controlli O(n²) si usa un punto ogni 3 (12 px)
CURVATURE_WINDOW = 10           # Campioni su cui si misura la curvatura (40 px)
MIN_RADIUS = 100                # Raggio minimo di curva percorribile dalle auto
MIN_CROSS_ANGLE = 60            # Gradi: incroci quasi perpendicolari, la strada giusta è "dritto"
MAX_CROSSINGS = 2               # Con più incroci non c'è spazio in 1600x900 rispettando i vincoli
CROSS_ZONE = 220                # Raggio attorno a un incrocio in cui i due tratti possono avvicinarsi
CROSS_STRAIGHT_ZONE = 130       # Tratto prima e dopo un incrocio che deve essere quasi dritto...
CROSS_STRAIGHT_RADIUS = 300     # ...cioè con raggio di curva almeno questo
MIN_GAP = 2.2 * TRACK_WIDTH     # Distanza minima tra tratti diversi lontano dagli incroci
MIN_CROSS_DIST = 400            # Distanza minima tra due incroci
SPAWN_CLEARANCE = 400           # Distanza minima dello spawn dagli incroci
SPAWN_STRAIGHT_RADIUS = 350     # Lo spawn va su un tratto quasi dritto
FINISH_OFFSET = 30              # Traguardo (ultimo gate) poco prima dello spawn
MIN_LENGTH, MAX_LENGTH = 2800, 9000
MAX_ATTEMPTS = 20000


class GeneratedTrack:
    def __init__(self, seed, points, spawn_pos, spawn_angle, gates, crossings):
        self.seed = seed
        self.points = points            # Linea centrale, dal punto di spawn nel verso di marcia
        self.spawn_pos = spawn_pos
        self.spawn_angle = spawn_angle  # Gradi, coordinate schermo (y verso il basso)
        self.gates = gates
        self.crossings = crossings      # Lista di (x, y, angolo in gradi)
        self.length = TrackPath(points).length
        self.surface = render_track(points, gates[-1])


# --- GENERAZIONE ---
def fit_to_image(pts, stretch):
    """Centra la curva nell'immagine. stretch=True la deforma per riempirla, altrimenti scala uniforme."""
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    box = np.array([WIDTH - 2 * MARGIN, HEIGHT - 2 * MARGIN], dtype=float)
    scale = box / (hi - lo) if stretch else (box / (hi - lo)).min()
    return (pts - (lo + hi) / 2) * scale + np.array([WIDTH, HEIGHT]) / 2


def harmonic_curve(rng, n=720):
    """Curva chiusa senza incroci: ellisse deformata da armoniche casuali deboli."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    strength = rng.uniform(0.15, 0.45)
    x = np.cos(t) * rng.uniform(0.8, 1.2)
    y = np.sin(t) * rng.uniform(0.8, 1.2)
    for k in range(2, int(rng.integers(2, 5)) + 1):
        amp = strength / k ** 0.7
        x += amp * (rng.normal() * np.cos(k * t) + rng.normal() * np.sin(k * t))
        y += amp * (rng.normal() * np.cos(k * t) + rng.normal() * np.sin(k * t))
    return fit_to_image(np.column_stack([x, y]), stretch=True)


def belt_tangent(c1, r1, s1, c2, r2, s2):
    """
    Tangente percorsa dal cerchio 1 al cerchio 2. s = +1/-1 indica da che lato della
    direzione di marcia sta il centro (cioè il verso in cui il cerchio viene avvolto).
    Con versi uguali la tangente è esterna, con versi opposti è interna (e incrocia la sua gemella).
    Ritorna i due punti di contatto, o None se i cerchi sono troppo vicini.
    """
    delta = c2 - c1
    dist = math.hypot(delta[0], delta[1])
    dr = s2 * r2 - s1 * r1
    if abs(dr) >= dist:
        return None
    theta = math.atan2(delta[1], delta[0]) - math.asin(dr / dist)
    normal = np.array([-math.sin(theta), math.cos(theta)])
    return c1 - s1 * r1 * normal, c2 - s2 * r2 * normal


def belt_path(centers, radii, signs, sequence):
    """Linea centrale chiusa che avvolge i cerchi nell'ordine `sequence`."""
    m = len(sequence)
    tangents = []
    for k in range(m):
        i, j = sequence[k], sequence[(k + 1) % m]
        t = belt_tangent(centers[i], radii[i], signs[i], centers[j], radii[j], signs[j])
        if t is None:
            return None
        tangents.append(t)

    pts = []
    for k in range(m):
        i = sequence[k]
        c, r, s = centers[i], radii[i], signs[i]
        p_in, p_out = tangents[k - 1][1], tangents[k][0]
        a_in = math.atan2(p_in[1] - c[1], p_in[0] - c[0])
        a_out = math.atan2(p_out[1] - c[1], p_out[0] - c[0])
        sweep = ((a_out - a_in) * s) % (2 * math.pi)
        if sweep > 1.9 * math.pi:  # Giro quasi completo: l'arco si richiuderebbe su se stesso
            return None
        for a in np.linspace(0, sweep, max(2, int(sweep * r / 2)), endpoint=False):
            pts.append(c + r * np.array([math.cos(a_in + s * a), math.sin(a_in + s * a)]))
        p0, p1 = tangents[k]
        for t in np.linspace(0, 1, max(2, int(math.hypot(*(p1 - p0)) / 2)), endpoint=False):
            pts.append(p0 + (p1 - p0) * t)
    return np.array(pts)


def chain_curve(rng, crossings):
    """
    Catena di cerchi disposta in orizzontale e percorsa andata e ritorno.
    Ogni collegamento tra cerchi avvolti in verso opposto produce esattamente un incrocio.
    """
    m = crossings + 1 + int(rng.integers(0, 2))
    radii = rng.uniform(MIN_RADIUS * 1.05, 200, m)
    flips = np.zeros(m - 1, dtype=bool)
    flips[rng.choice(m - 1, crossings, replace=False)] = True
    signs = np.ones(m)
    for k in range(1, m):
        signs[k] = -signs[k - 1] if flips[k - 1] else signs[k - 1]

    centers = [np.zeros(2)]
    heading = rng.uniform(-0.25, 0.25)
    for k in range(1, m):
        heading += rng.uniform(-0.5, 0.5)
        if flips[k - 1]:
            # Distanza tra 1.7 e 2 volte la somma dei raggi: incrocio tra 60° e 72°
            dist = rng.uniform(1.7, 2.0) * (radii[k - 1] + radii[k])
        else:
            dist = radii[k - 1] + radii[k] + rng.uniform(1.6, 4.0) * TRACK_WIDTH
        centers.append(centers[-1] + dist * np.array([math.cos(heading), math.sin(heading)]))
    centers = np.array(centers)

    gaps = np.hypot(*(centers[:, None] - centers[None]).transpose(2, 0, 1)) - radii[:, None] - radii[None, :]
    np.fill_diagonal(gaps, np.inf)
    if gaps.min() < 1.5 * TRACK_WIDTH:
        return None

    sequence = list(range(m)) + list(range(m - 2, 0, -1))
    pts = belt_path(centers, radii, signs, sequence)
    return None if pts is None else fit_to_image(pts, stretch=False)


def candidate_curve(rng, crossings):
    return harmonic_curve(rng) if crossings == 0 else chain_curve(rng, crossings)


def resample(points, step):
    path = TrackPath(points)
    s = np.arange(0, path.length, step)
    return np.column_stack([np.interp(s, path.cum, path.ext[:, 0]), np.interp(s, path.cum, path.ext[:, 1])])


def curvature_radius(pts):
    """Raggio di curvatura (px) in ogni punto di una curva chiusa campionata a passo STEP."""
    seg = np.roll(pts, -1, axis=0) - pts
    heading = np.arctan2(seg[:, 1], seg[:, 0])
    turn = (np.roll(heading, -1) - heading + np.pi) % (2 * np.pi) - np.pi
    # Curvatura media su una finestra centrata
    kernel = np.ones(CURVATURE_WINDOW) / (CURVATURE_WINDOW * STEP)
    padded = np.concatenate([turn[-CURVATURE_WINDOW:], turn, turn[:CURVATURE_WINDOW]])
    curvature = np.convolve(padded, kernel, mode="same")[CURVATURE_WINDOW:-CURVATURE_WINDOW]
    with np.errstate(divide="ignore"):
        return 1.0 / np.abs(curvature)


def find_crossings(pts):
    """Incroci tra segmenti non adiacenti: lista di (punto, angolo in gradi, indice_a, indice_b)."""
    a = pts
    b = np.roll(pts, -1, axis=0)
    d = b - a
    n = len(pts)
    # Tutte le coppie i < j non adiacenti
    i, j = np.triu_indices(n, k=2)
    keep = ~((i == 0) & (j == n - 1))
    i, j = i[keep], j[keep]

    di, dj = d[i], d[j]
    ap = a[j] - a[i]
    denom = di[:, 0] * dj[:, 1] - di[:, 1] * dj[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (ap[:, 0] * dj[:, 1] - ap[:, 1] * dj[:, 0]) / denom
        u = (ap[:, 0] * di[:, 1] - ap[:, 1] * di[:, 0]) / denom
    hit = (denom != 0) & (t >= 0) & (t < 1) & (u >= 0) & (u < 1)

    result = []
    for k in np.flatnonzero(hit):
        p = a[i[k]] + t[k] * di[k]
        cos_angle = abs(np.dot(di[k], dj[k])) / (np.linalg.norm(di[k]) * np.linalg.norm(dj[k]))
        angle = math.degrees(math.acos(min(1.0, cos_angle)))
        result.append((p, angle, int(i[k]), int(j[k])))
    return result


def evaluate(raw, rng):
    """Controlla una curva candidata. Ritorna (punti, indice_spawn, incroci) oppure None."""
    pts = resample(raw, STEP)
    length = len(pts) * STEP
    if not MIN_LENGTH <= length <= MAX_LENGTH:
        return None

    radius = curvature_radius(pts)
    if radius.min() < MIN_RADIUS:
        return None

    coarse = pts[::COARSE_EVERY]
    found = find_crossings(coarse)
    crossings = []
    for p, angle, ia, ib in found:
        if angle < MIN_CROSS_ANGLE:
            return None
        crossings.append((p, angle, ia * COARSE_EVERY, ib * COARSE_EVERY))

    # Incroci ben distanziati tra loro
    for k in range(len(crossings)):
        for m in range(k + 1, len(crossings)):
            if np.linalg.norm(crossings[k][0] - crossings[m][0]) < MIN_CROSS_DIST:
                return None

    # Strada dritta vicino agli incroci: così la scelta giusta è sempre "tirare dritto"
    zone = int(CROSS_STRAIGHT_ZONE / STEP)
    n = len(pts)
    for _, _, ia, ib in crossings:
        for idx in (ia, ib):
            window = np.arange(idx - zone, idx + zone) % n
            if radius[window].min() < CROSS_STRAIGHT_RADIUS:
                return None

    # Tratti diversi della pista devono restare separati, tranne in corrispondenza degli incroci
    m = len(coarse)
    arc_step = STEP * COARSE_EVERY
    idx = np.arange(m)
    arc = np.abs(idx[:, None] - idx[None, :]) * arc_step
    arc = np.minimum(arc, m * arc_step - arc)
    dist = np.hypot(coarse[:, None, 0] - coarse[None, :, 0], coarse[:, None, 1] - coarse[None, :, 1])
    close_pairs = np.argwhere((arc > NEIGHBOR_ARC) & (dist < MIN_GAP))
    if len(close_pairs):
        if not crossings:
            return None
        cross_pts = np.array([c[0] for c in crossings])
        near_cross = np.hypot(*(coarse[:, None, :] - cross_pts[None, :, :]).transpose(2, 0, 1)).min(axis=1) < CROSS_ZONE
        if not np.all(near_cross[close_pairs[:, 0]] & near_cross[close_pairs[:, 1]]):
            return None

    # Spawn su un tratto dritto e lontano dagli incroci
    straight = np.ones(n, dtype=bool)
    spawn_zone = int(150 / STEP)
    for shift in range(-spawn_zone, spawn_zone + 1, 5):
        straight &= np.roll(radius, shift) >= SPAWN_STRAIGHT_RADIUS
    if crossings:
        cross_pts = np.array([c[0] for c in crossings])
        far = np.hypot(*(pts[:, None, :] - cross_pts[None, :, :]).transpose(2, 0, 1)).min(axis=1) >= SPAWN_CLEARANCE
        straight &= far
    candidates = np.flatnonzero(straight)
    if len(candidates) == 0:
        return None

    return pts, int(rng.choice(candidates)), crossings


def generate_track(seed=None, crossings=None):
    """
    Genera una pista. crossings = numero di incroci desiderato (None = casuale tra 0 e MAX_CROSSINGS).
    Stesso seed e stessi parametri -> stessa pista.
    """
    if seed is None:
        seed = random.randrange(1_000_000)
    rng = np.random.default_rng(seed)
    if crossings is None:
        crossings = int(rng.choice([0, 1, 1, 2, 2]))
    crossings = min(crossings, MAX_CROSSINGS)

    for _ in range(MAX_ATTEMPTS):
        raw = candidate_curve(rng, crossings)
        if raw is None:
            continue
        result = evaluate(raw, rng)
        if result is None or len(result[2]) != crossings:
            continue
        pts, spawn_idx, found = result

        # La pista parte dallo spawn; il verso di marcia è casuale
        pts = np.roll(pts, -spawn_idx, axis=0)
        if rng.random() < 0.5:
            pts = np.vstack([pts[:1], pts[:0:-1]])

        direction = pts[3] - pts[0]
        spawn_angle = math.degrees(math.atan2(direction[1], direction[0]))
        gates = build_checkpoint_gates(pts, closed=True, start=GATE_SPACING / 2, finish_offset=FINISH_OFFSET)
        crossing_info = [(float(p[0]), float(p[1]), float(angle)) for p, angle, _, _ in found]
        return GeneratedTrack(seed, pts, (float(pts[0][0]), float(pts[0][1])), spawn_angle, gates, crossing_info)

    raise RuntimeError(f"Nessuna pista valida con {crossings} incroci dopo {MAX_ATTEMPTS} tentativi (seed {seed})")


# --- DISEGNO E SALVATAGGIO ---
def render_track(points, finish_gate):
    """Pista bianca su sfondo nero (il nero è muro), con il traguardo verde."""
    surf = pygame.Surface((WIDTH, HEIGHT))
    surf.fill((0, 0, 0))
    for x, y in points:
        pygame.draw.circle(surf, (255, 255, 255), (int(x), int(y)), TRACK_WIDTH // 2)

    a, b = pygame.Vector2(finish_gate[0]), pygame.Vector2(finish_gate[1])
    center = (a + b) / 2
    half = (b - a).normalize() * (TRACK_WIDTH // 2)
    pygame.draw.line(surf, (0, 255, 0), center - half, center + half, 10)
    return surf


def save_generated(track, name):
    save_track(name, track.surface, track.spawn_pos, -track.spawn_angle, track.gates,
               gates_ordered=True, seed=track.seed, crossings=track.crossings)
    print(f"Pista salvata: {name}.png + tracks_config/{name}.pkl "
          f"(seed {track.seed}, {len(track.crossings)} incroci, {len(track.gates)} gate)")
    print(f"Allenamento: python training.py {name}   |   Gara: python simulation.py {name}")


# --- ANTEPRIMA INTERATTIVA ---
def draw_preview(screen, font, track, crossings_setting, name, message):
    screen.blit(track.surface, (0, 0))

    for i, (a, b) in enumerate(track.gates):
        color = (0, 200, 0) if i == len(track.gates) - 1 else (70, 130, 255)
        pygame.draw.line(screen, color, a, b, 2)

    for x, y, _ in track.crossings:
        pygame.draw.circle(screen, (255, 200, 0), (int(x), int(y)), TRACK_WIDTH, 2)

    rad = math.radians(track.spawn_angle)
    start = pygame.Vector2(track.spawn_pos)
    pygame.draw.line(screen, (0, 255, 0), start, start + pygame.Vector2(math.cos(rad), math.sin(rad)) * 60, 6)
    pygame.draw.circle(screen, (0, 255, 0), start, 10)

    setting = "casuali" if crossings_setting is None else str(crossings_setting)
    lines = [
        f"Seed {track.seed} | Incroci {len(track.crossings)} (impostazione: {setting}) | "
        f"Lunghezza {track.length:.0f} px | Gate {len(track.gates)}",
        f"SPAZIO nuova pista | 0-{MAX_CROSSINGS} incroci | R casuali | INVIO salva come '{name}' | ESC esci",
    ]
    if message:
        lines.append(message)
    for i, text in enumerate(lines):
        img = font.render(text, True, (255, 255, 255))
        bg = pygame.Surface((img.get_width() + 12, img.get_height() + 4), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 180))
        screen.blit(bg, (6, 6 + i * 26))
        screen.blit(img, (12, 8 + i * 26))


def show_generating(screen, font):
    img = font.render("Generazione pista...", True, (255, 255, 0))
    screen.fill((0, 0, 0))
    screen.blit(img, ((WIDTH - img.get_width()) // 2, HEIGHT // 2))
    pygame.display.flip()


def run_preview(name, crossings, seed):
    restore_window_position(WINDOW_KEY)
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Generatore di piste")
    font = pygame.font.SysFont("Arial", 18, bold=True)
    clock = pygame.time.Clock()

    show_generating(screen, font)
    track = generate_track(seed, crossings)
    message = ""

    while True:
        regenerate = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type != pygame.KEYDOWN:
                continue
            if event.key == pygame.K_ESCAPE:
                return
            if event.key == pygame.K_SPACE:
                regenerate = True
            elif event.key == pygame.K_r:
                crossings, regenerate = None, True
            elif pygame.K_0 <= event.key <= pygame.K_0 + MAX_CROSSINGS:
                crossings, regenerate = event.key - pygame.K_0, True
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                save_generated(track, name)
                message = f"Salvata come '{name}'. Allena con: python training.py {name}"

        if regenerate:
            show_generating(screen, font)
            track = generate_track(None, crossings)
            message = ""

        draw_preview(screen, font, track, crossings, name, message)
        pygame.display.flip()
        clock.tick(30)


def main():
    parser = argparse.ArgumentParser(description="Generatore automatico di piste")
    parser.add_argument("--nome", default=DEFAULT_NAME, help=f"nome della pista (default: {DEFAULT_NAME})")
    parser.add_argument("--incroci", type=int, choices=range(0, MAX_CROSSINGS + 1), default=None,
                        help=f"numero di incroci (default: casuale tra 0 e {MAX_CROSSINGS})")
    parser.add_argument("--seed", type=int, default=None, help="seme per riprodurre una pista")
    parser.add_argument("--salva", action="store_true", help="genera e salva senza anteprima")
    args = parser.parse_args()

    pygame.init()
    if args.salva:
        save_generated(generate_track(args.seed, args.incroci), args.nome)
    else:
        run_preview(args.nome, args.incroci, args.seed)


if __name__ == "__main__":
    try:
        main()
    finally:
        close_window(WINDOW_KEY)
