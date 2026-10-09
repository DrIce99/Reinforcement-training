# generate_track.py
"""
Generatore automatico di piste per training.py e simulation.py.

Le piste sono "cinghie" avvolte su cerchi (pulegge): il raggio minimo di curva è garantito
dai cerchi e tra un cerchio e l'altro ci sono rettilinei.
- Anello: cerchi sparsi nell'immagine e percorsi in ordine; i cerchi avvolti al contrario
  creano rientranze, tornanti e chicane.
- Catena: cerchi in fila percorsi all'andata e al ritorno; ogni volta che due cerchi
  consecutivi sono avvolti in verso opposto le due tangenti tra loro si incrociano.
Tra alcune piste valide viene tenuta la più tortuosa.

Le piste sono grandi WIDTH x HEIGHT pixel (più della finestra): training e gara le mostrano ridotte,
mentre auto e sensori lavorano alla grandezza reale.

Gli incroci sono quasi perpendicolari e su tratti dritti; i gate vicini agli incroci vengono
omessi e chi imbocca la strada sbagliata attraversa un gate fuori sequenza e viene eliminato,
quindi l'evoluzione premia chi segue il navigatore e tira dritto.

Uso:
    python generate_track.py                          anteprima interattiva
    python generate_track.py --salva --quante 5       genera e salva 5 piste (pista_auto_1, _2, ...)
    python generate_track.py --nome pista_x --incroci 2 --seed 42 --salva

Comandi dell'anteprima:
    SPAZIO = nuova pista    0-2 = numero di incroci    R = incroci casuali
    INVIO = salva (ogni pista con un nuovo nome)     ESC = esci
Poi: python training.py <nome> per ogni pista, e python simulation.py per il campionato su tutte.
"""
import argparse
import math
import random

import numpy as np
import pygame

from track import (TRACK_WIDTH, GATE_SPACING, NEIGHBOR_ARC, TrackPath,
                   build_checkpoint_gates, save_track, next_free_track_name, view_scale, scaled_view)
from window_state import restore_window_position, close_window

WIDTH, HEIGHT = 2400, 1350       # Grandezza reale della pista (la finestra la mostra ridotta)
WINDOW_KEY = "generate_track"
DEFAULT_NAME = "pista_auto"   # Le piste salvate si chiamano pista_auto_1, pista_auto_2, ...

MARGIN = TRACK_WIDTH            # Distanza minima della linea centrale dal bordo dell'immagine
STEP = 4                        # Passo di campionamento del tracciato (px)
COARSE_EVERY = 3                # Per i controlli O(n²) si usa un punto ogni 3 (12 px)
CURVATURE_WINDOW = 10           # Campioni su cui si misura la curvatura (40 px)
MIN_RADIUS = 100                # Raggio minimo di curva percorribile dalle auto
MIN_CROSS_ANGLE = 60            # Gradi: incroci quasi perpendicolari, la strada giusta è "dritto"
MAX_CROSSINGS = 2               # Con più incroci non c'è spazio rispettando i vincoli
CROSS_ZONE = 220                # Raggio attorno a un incrocio in cui i due tratti possono avvicinarsi
CROSS_STRAIGHT_ZONE = 130       # Tratto prima e dopo un incrocio che deve essere quasi dritto...
CROSS_STRAIGHT_RADIUS = 300     # ...cioè con raggio di curva almeno questo
MIN_GAP = 2.2 * TRACK_WIDTH     # Distanza minima tra tratti diversi lontano dagli incroci
MIN_CROSS_DIST = 400            # Distanza minima tra due incroci
SPAWN_CLEARANCE = 400           # Distanza minima dello spawn dagli incroci
SPAWN_STRAIGHT_RADIUS = 350     # Lo spawn va su un tratto quasi dritto
FINISH_OFFSET = 30              # Traguardo (ultimo gate) poco prima dello spawn
MIN_LENGTH, MAX_LENGTH = 5000, 14000
MAX_ATTEMPTS = 20000
CANDIDATES = 4                  # Piste valide tra cui scegliere la più tortuosa
EXTRA_ATTEMPTS = 1500           # Tentativi concessi dopo la prima pista valida per trovarne altre


class GeneratedTrack:
    def __init__(self, seed, points, spawn_pos, spawn_angle, gates, crossings):
        self.seed = seed
        self.points = points            # Linea centrale, dal punto di spawn nel verso di marcia
        self.spawn_pos = spawn_pos
        self.spawn_angle = spawn_angle  # Gradi, coordinate schermo (y verso il basso)
        self.gates = gates
        self.crossings = crossings      # Lista di (x, y, angolo in gradi)
        self.length = TrackPath(points).length
        self.complexity = complexity(points)
        self.surface = render_track(points, gates[-1])
        self.preview = None             # Anteprima ridotta per la finestra, creata al primo disegno
        self.saved_as = None


# --- GENERAZIONE ---
def fit_to_image(pts, stretch, max_scale=None):
    """
    Centra la curva nell'immagine. stretch=True la deforma per riempirla, altrimenti scala uniforme.
    max_scale limita l'ingrandimento (ingrandire allarga le curve e rende la pista più semplice).
    """
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    box = np.array([WIDTH - 2 * MARGIN, HEIGHT - 2 * MARGIN], dtype=float)
    scale = box / (hi - lo) if stretch else (box / (hi - lo)).min()
    if max_scale is not None:
        scale = np.minimum(scale, max_scale)
    return (pts - (lo + hi) / 2) * scale + np.array([WIDTH, HEIGHT]) / 2


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
    m = crossings + 1 + int(rng.integers(1, 4))
    radii = rng.uniform(MIN_RADIUS * 1.05, 180, m)
    flips = np.zeros(m - 1, dtype=bool)
    flips[rng.choice(m - 1, crossings, replace=False)] = True
    signs = np.ones(m)
    for k in range(1, m):
        signs[k] = -signs[k - 1] if flips[k - 1] else signs[k - 1]

    centers = [np.zeros(2)]
    heading = rng.uniform(-0.25, 0.25)
    for k in range(1, m):
        heading += rng.uniform(-0.6, 0.6)
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
    return None if pts is None else fit_to_image(pts, stretch=False, max_scale=1.0)


def ring_curve(rng):
    """
    Anello di cerchi sparsi nell'immagine, percorsi in ordine attorno al centro.
    I cerchi interni vengono spesso avvolti al contrario: la pista rientra verso il centro
    formando tornanti e chicane (e a volte incroci).
    """
    m = int(rng.integers(8, 13))
    radii = rng.uniform(MIN_RADIUS * 1.05, 180, m)
    centers = np.empty((0, 2))
    for k, r in enumerate(radii):
        # 200 posizioni candidate in un colpo solo: si tiene la prima abbastanza lontana dagli altri cerchi
        cand = np.column_stack([rng.uniform(MARGIN + r, WIDTH - MARGIN - r, 200),
                                rng.uniform(MARGIN + r, HEIGHT - MARGIN - r, 200)])
        if len(centers):
            gaps = np.hypot(*(cand[:, None, :] - centers[None, :, :]).transpose(2, 0, 1)) - r - radii[:k]
            free = np.flatnonzero((gaps >= 2 * TRACK_WIDTH).all(axis=1))
            if len(free) == 0:
                return None
            cand = cand[free]
        centers = np.vstack([centers, cand[:1]])

    mid = centers.mean(axis=0)
    order = list(np.argsort(np.arctan2(centers[:, 1] - mid[1], centers[:, 0] - mid[0])))
    dist = np.hypot(*(centers - mid).T)
    inner = 1 - (dist - dist.min()) / (np.ptp(dist) + 1e-9)   # 1 = cerchio più vicino al centro
    signs = np.where(rng.random(m) < 0.6 * (0.3 + inner), -1.0, 1.0)
    return belt_path(centers, radii, signs, order)


def candidate_curve(rng, crossings):
    # L'anello dà le piste più tortuose ma raramente il numero esatto di incroci:
    # ogni tanto si prova la catena, che li garantisce
    if crossings == 0 or rng.random() < 0.9:
        return ring_curve(rng)
    return chain_curve(rng, crossings)


def complexity(pts):
    """Quanto è tortuosa la pista: sterzata totale in giri completi (un ovale vale 1)."""
    seg = np.roll(pts, -1, axis=0) - pts
    heading = np.arctan2(seg[:, 1], seg[:, 0])
    turn = (np.roll(heading, -1) - heading + np.pi) % (2 * np.pi) - np.pi
    return float(np.abs(turn).sum() / (2 * np.pi))


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

    valid = []
    first_valid_at = None
    for attempt in range(MAX_ATTEMPTS):
        if first_valid_at is not None and attempt - first_valid_at > EXTRA_ATTEMPTS:
            break
        raw = candidate_curve(rng, crossings)
        if raw is None:
            continue
        result = evaluate(raw, rng)
        if result is None or len(result[2]) != crossings:
            continue
        valid.append(result)
        if first_valid_at is None:
            first_valid_at = attempt
        if len(valid) >= CANDIDATES:
            break

    if valid:
        pts, spawn_idx, found = max(valid, key=lambda res: complexity(res[0]))

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


def target_name(explicit_name):
    """Nome scelto dall'utente, altrimenti il primo pista_auto_N libero."""
    return explicit_name or next_free_track_name(DEFAULT_NAME)


def save_generated(track, name):
    save_track(name, track.surface, track.spawn_pos, -track.spawn_angle, track.gates,
               gates_ordered=True, seed=track.seed, crossings=track.crossings)
    track.saved_as = name
    print(f"Pista salvata: {name}.png + tracks_config/{name}.pkl "
          f"(seed {track.seed}, {len(track.crossings)} incroci, {len(track.gates)} gate)")
    print(f"  Allenamento: python training.py {name}")


# --- ANTEPRIMA INTERATTIVA ---
def build_preview(track):
    """Pista con gate, incroci e partenza, ridotta alla grandezza della finestra."""
    surf = track.surface.copy()
    for i, (a, b) in enumerate(track.gates):
        color = (0, 200, 0) if i == len(track.gates) - 1 else (70, 130, 255)
        pygame.draw.line(surf, color, a, b, 3)

    for x, y, _ in track.crossings:
        pygame.draw.circle(surf, (255, 200, 0), (int(x), int(y)), TRACK_WIDTH, 3)

    rad = math.radians(track.spawn_angle)
    start = pygame.Vector2(track.spawn_pos)
    pygame.draw.line(surf, (0, 255, 0), start, start + pygame.Vector2(math.cos(rad), math.sin(rad)) * 80, 8)
    pygame.draw.circle(surf, (0, 255, 0), start, 14)
    return scaled_view(surf, view_scale(surf.get_size()))


def draw_preview(screen, font, track, crossings_setting, save_name, message):
    if track.preview is None:
        track.preview = build_preview(track)
    screen.blit(track.preview, (0, 0))

    setting = "casuali" if crossings_setting is None else str(crossings_setting)
    lines = [
        f"Seed {track.seed} | Incroci {len(track.crossings)} (impostazione: {setting}) | "
        f"Lunghezza {track.length:.0f} px | Tortuosità {track.complexity:.1f} | Gate {len(track.gates)}",
        f"SPAZIO nuova pista | 0-{MAX_CROSSINGS} incroci | R casuali | "
        + (f"Salvata come '{track.saved_as}'" if track.saved_as else f"INVIO salva come '{save_name}'")
        + " | ESC esci",
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
    screen.blit(img, ((screen.get_width() - img.get_width()) // 2, screen.get_height() // 2))
    pygame.display.flip()


def run_preview(name, crossings, seed):
    restore_window_position(WINDOW_KEY)
    scale = view_scale((WIDTH, HEIGHT))
    screen = pygame.display.set_mode((round(WIDTH * scale), round(HEIGHT * scale)))
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
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and not track.saved_as:
                saved = target_name(name)
                save_generated(track, saved)
                message = (f"Salvata come '{saved}'. Allena con: python training.py {saved} | "
                           f"Campionato su tutte le piste: python simulation.py")

        if regenerate:
            show_generating(screen, font)
            track = generate_track(None, crossings)
            message = ""

        draw_preview(screen, font, track, crossings, target_name(name), message)
        pygame.display.flip()
        clock.tick(30)


def main():
    parser = argparse.ArgumentParser(description="Generatore automatico di piste")
    parser.add_argument("--nome", default=None,
                        help=f"nome della pista (default: {DEFAULT_NAME}_1, {DEFAULT_NAME}_2, ... senza sovrascrivere)")
    parser.add_argument("--incroci", type=int, choices=range(0, MAX_CROSSINGS + 1), default=None,
                        help=f"numero di incroci (default: casuale tra 0 e {MAX_CROSSINGS})")
    parser.add_argument("--seed", type=int, default=None, help="seme per riprodurre una pista")
    parser.add_argument("--salva", action="store_true", help="genera e salva senza anteprima")
    parser.add_argument("--quante", type=int, default=1, help="con --salva: quante piste generare (default: 1)")
    args = parser.parse_args()

    pygame.init()
    if args.salva:
        for i in range(max(1, args.quante)):
            seed = None if args.seed is None else args.seed + i
            if args.nome and args.quante > 1:
                name = next_free_track_name(args.nome)
            else:
                name = target_name(args.nome)
            save_generated(generate_track(seed, args.incroci), name)
        print("Campionato su tutte le piste: python simulation.py")
    else:
        run_preview(args.nome, args.incroci, args.seed)


if __name__ == "__main__":
    try:
        main()
    finally:
        close_window(WINDOW_KEY)
