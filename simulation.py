import argparse
import glob
import pygame
import numpy as np
import math
import pickle
import os
import json
import random

from physics import compute_step
from speed_control import SpeedControl
from track import (SENSOR_COUNT, list_tracks, load_track, navigation_inputs, adapt_weights,
                   view_scale, scaled_view)
from window_state import restore_window_position, close_window

# --- CONFIGURAZIONE GARA ---
NUM_RACERS = 20  # Numero di partecipanti alla gara
LAPS_TO_WIN = 7  # Giri per gara (modificabile con --giri)
MAX_FRAMES_WITHOUT_GATE = 1000  # Chi non raggiunge il gate successivo in tempo (es. gira in tondo) è fuori
POINTS_SYSTEM = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
LEADERBOARD_FILE = "leaderboard.json"

# Uso:
#   python simulation.py                       campionato su tutte le piste in tracks_config/
#   python simulation.py pista_a pista_b       campionato su queste piste, in quest'ordine
#   python simulation.py pista_a --giri 3      gara singola da 3 giri
WINDOW_KEY = "simulation"

# --- CLASSE PILOTA GARA ---
class Racer:
    def __init__(self, weights, color, racer_id, spawn_pos, base_angle):
        self.weights = weights.copy()
        self.color = color
        self.id = racer_id
        self.pos = pygame.Vector2(spawn_pos)
        self.angle = base_angle
        self.alive = True
        self.completed = False
        self.score = 0
        self.time = 0

        self.laps = 0
        self.next_cp = 0  # Prossimo gate da attraversare: il giro conta solo passandoli tutti in ordine
        self.finish_time = 0
        self.last_gate_time = 0

        self.learning_rate = 0.01  # Quanto velocemente cambia idea
        self.prev_dist_to_walls = 0

        self.velocity = 0
        self.prev_steer = 0

        # Personalità del pilota
        self.aggressiveness = random.uniform(0.9, 1.15)   # quanto spinge
        self.precision = random.uniform(0.85, 1.1)        # quanto è pulito nello sterzo
        self.risk_taking = random.uniform(0.9, 1.2)       # quanto rischia in curva
        self.confidence = 0.0

        self.weights += np.random.normal(0, 0.02, self.weights.shape)

        # --- STATO DELLE GOMME ---
        self.tyre_health = 100.0       # Da 100 (perfette) a 0 (distrutte)
        self.tyre_fragility = random.uniform(0.9, 1.1) # Alcune auto consumano di più
        self.prev_speed_raw = 0

        # --- CAOS E APPRENDIMENTO ---
        self.chaos_factor = random.uniform(0.02, 0.12)  # Quanto è "sloppy" nello sterzo
        self.imitation_rate = random.uniform(0.001, 0.01)  # Quanto impara dagli altri

    def progress(self, gate_count):
        return self.laps * gate_count + self.next_cp

    def predict(self, sensors, nav):
        output = np.dot(np.concatenate([sensors, nav]), self.weights)
        self.confidence = float(np.mean(sensors) - np.std(sensors))

        # 1. Caos base del pilota + Caos dovuto alle gomme usurate
        # Se le gomme sono al 50%, il caos raddoppia. A 0%, triplica.
        tyre_chaos = self.chaos_factor * (1.0 + (1.0 - self.tyre_health / 100.0) * 2.0)
        noise = np.random.normal(0, tyre_chaos, output.shape)

        action = np.tanh(output + noise)
        return action

    def update(self, track_image, gates, total_laps, racers):
        if not self.alive or self.completed:
            return

        self.time += 1
        sensors = self.get_sensors(track_image)
        self.learn_on_the_fly(sensors)
        action = self.predict(sensors, navigation_inputs(self.pos, self.angle, gates, self.next_cp))

        steer_raw, speed_raw = action[0], action[1]

        # --- AGGRESSIVITÀ DINAMICA E APPRENDIMENTO SOCIALE ---
        boost = 1.0
        best_nearby_racer = None
        obs_range = 150
        min_obs_dist = obs_range

        for other in racers:
            if other is self or not other.alive: continue
            dist = self.pos.distance_to(other.pos)

            if dist < min_obs_dist:
                rad = math.radians(self.angle)
                forward = pygame.Vector2(math.cos(rad), math.sin(rad))
                direction_vec = other.pos - self.pos

                if direction_vec.length_squared() > 0.0001:
                    if forward.dot(direction_vec.normalize()) > 0.5:
                        boost += 0.02 * (1 - dist / obs_range)

                        my_progress = (self.progress(len(gates)), self.score)
                        other_progress = (other.progress(len(gates)), other.score)

                        if other_progress > my_progress:
                            best_nearby_racer = other
                            min_obs_dist = dist

        if best_nearby_racer is not None:
            mask = np.random.rand(*self.weights.shape) < self.imitation_rate
            self.weights[mask] = best_nearby_racer.weights[mask]

        # --- USURA GOMME ---
        # 1. Usura costante per la velocità
        wear_base = self.velocity * 0.0005

        # 2. Usura per le curve ad alta velocità (stress laterale)
        wear_cornering = abs(steer_raw) * self.velocity * 0.008

        # 3. Usura per accelerazioni violente
        wear_accel = max(0, speed_raw) * 0.002

        # 4. Usura per FRENATE IMPROVVISE (se il comando accel cala drasticamente)
        brake_intensity = max(0, self.prev_speed_raw - speed_raw)
        wear_braking = brake_intensity * 0.015  # Le staccate violente consumano molto!
        self.prev_speed_raw = speed_raw  # Salva per il frame prossimo

        # Applichiamo l'usura
        self.tyre_health -= (wear_base + wear_cornering + wear_accel + wear_braking) * self.tyre_fragility
        self.tyre_health = max(0, self.tyre_health) # Non scende sotto lo 0%

        # Effetto delle gomme sulle performance (potenza motore)
        tyre_penalty = 0.5 + 0.5 * (self.tyre_health / 100.0)

        # Sbandata macroscopica se le gomme sono messe male
        if self.tyre_health < 50 and random.random() < (50 - self.tyre_health) * 0.0008:
            steer_raw += random.uniform(-1.0, 1.0)
            self.score -= 50

        # --- FISICA UNIFICATA CON GOMME ---
        effective_aggressiveness = self.aggressiveness * min(boost, 1.25) * tyre_penalty

        # Passiamo tyre_health a compute_step!
        self.angle, self.velocity = compute_step(
            angle=self.angle,
            velocity=self.velocity,
            steer_raw=steer_raw,
            speed_raw=speed_raw,
            confidence=self.confidence,
            precision=self.precision,
            aggressiveness=effective_aggressiveness,
            risk_taking=self.risk_taking,
            tyre_health=self.tyre_health
        )

        old_pos = self.pos.copy()
        rad = math.radians(self.angle)
        self.pos += pygame.Vector2(math.cos(rad), math.sin(rad)) * self.velocity

        # Score e penalità sterzo
        self.score += old_pos.distance_to(self.pos) * 0.5
        steer_change = abs(steer_raw - self.prev_steer)
        self.score -= steer_change * 2.0
        self.score -= abs(steer_raw) * 0.2
        self.prev_steer = steer_raw

        np.clip(self.weights, -2.0, 2.0, out=self.weights)

        # Controllo Collisioni Pista (Nero = Muro)
        try:
            pixel = track_image.get_at((int(self.pos.x), int(self.pos.y)))
            if pixel.r < 30 and pixel.g < 30 and pixel.b < 30:
                self.alive = False
                self.score -= 50 + self.velocity * 200
                return
        except IndexError:
            self.alive = False
            return

        # Conteggio giri tramite i gate: vanno attraversati tutti e in ordine
        passed, wrong_way = gates.check_progress(old_pos, self.pos, self.next_cp)
        if wrong_way:
            # Strada sbagliata a un incrocio, taglio di pista o contromano
            self.alive = False
            self.score -= 500
            return
        if passed:
            self.next_cp += 1
            self.last_gate_time = self.time
            if self.next_cp >= len(gates):
                self.next_cp = 0
                self.laps += 1
                if self.laps >= total_laps:
                    self.completed = True
                    self.finish_time = self.time
                    return

        if self.time - self.last_gate_time > MAX_FRAMES_WITHOUT_GATE:
            self.alive = False

    def get_sensors(self, track):
        sensors = []
        spread = 120
        max_dist = 150
        start_angle = self.angle - spread / 2
        step = spread / (SENSOR_COUNT - 1)

        for i in range(SENSOR_COUNT):
            ray_angle = math.radians(start_angle + i * step)
            dist = 0
            while dist < max_dist:
                dist += 4
                x, y = int(self.pos.x + math.cos(ray_angle) * dist), int(self.pos.y + math.sin(ray_angle) * dist)
                try:
                    pixel = track.get_at((x, y))
                    if pixel.r < 50 and pixel.g < 50 and pixel.b < 50:
                        break
                except IndexError: break
            sensors.append(dist / max_dist)
        return np.array(sensors)

    def learn_on_the_fly(self, sensors):
        """
        Apprendimento individuale: se i sensori dicono che siamo troppo vicini
        ai muri, applichiamo una piccola mutazione correttiva ai pesi.
        """
        # Se la somma dei sensori è bassa, siamo vicini ai muri
        current_safety = np.mean(sensors)

        if current_safety < 0.3: # Siamo in pericolo
            # Applichiamo una piccola 'scossa' casuale ai pesi per cercare una manovra diversa
            mutation = np.random.normal(0, self.learning_rate, self.weights.shape)
            self.weights += mutation

# --- FUNZIONI DI SUPPORTO ---
def load_best_weights(track_name):
    """
    Modello addestrato per la pista. Se manca usa il modello addestrato più recente
    di un'altra pista (la guida di base si trasferisce), avvisando l'utente.
    """
    path = f"tracks/{track_name}.pkl"
    if not os.path.exists(path):
        others = sorted(glob.glob("tracks/*.pkl"), key=os.path.getmtime, reverse=True)
        if not others:
            return None
        path = others[0]
        print(f"Attenzione: nessun addestramento per '{track_name}', uso {path}. "
              f"Per risultati migliori: python training.py {track_name}")
    with open(path, "rb") as f:
        return adapt_weights(pickle.load(f))

def driver_weights_path(track_name, racer_id):
    return f"single/{track_name}/driver_{racer_id}.pkl"

def get_driver_weights(track_name, racer_id, base_weights):
    filename = driver_weights_path(track_name, racer_id)
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return adapt_weights(pickle.load(f))
    # Se è nuovo, parte dal DNA del training + una piccola variazione casuale
    return base_weights + np.random.normal(0, 0.05, base_weights.shape)

def save_driver_weights(track_name, racer_id, weights):
    filename = driver_weights_path(track_name, racer_id)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(weights, f)

def safe_load_json(filename):
    """Carica JSON con gestione errori"""
    if not os.path.exists(filename):
        return {}

    try:
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:  # File vuoto
                return {}
            return json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError):
        print(f"File {filename} corrotto. Reset!")
        return {}  # Reset se corrotto

def safe_save_json(filename, data):
    temp_file = filename + ".backup"
    with open(temp_file, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    os.replace(temp_file, filename)

def readable(color, min_brightness=110):
    """Schiarisce i colori troppo scuri per essere letti sullo sfondo scuro."""
    brightness = max(color)
    if brightness >= min_brightness:
        return tuple(color)
    return tuple(int(c + (min_brightness - brightness) + 40) for c in color)

def race_points(position):
    return POINTS_SYSTEM[position] if position < len(POINTS_SYSTEM) else 0

BASE_COLORS = [
    (255, 0, 0),   # Rosso
    (0, 0, 255),   # Blu
    (255, 255, 0), # Giallo
    (255, 165, 0), # Arancione
    (0, 255, 255), # Ciano
    (255, 0, 255), # Magenta
    (128, 0, 128), # Viola
    (0, 128, 128), # Ottanio
    (128, 128, 128), # Grigio
    (200, 200, 200),  # Grigio chiaro
    (8, 250, 0),  # Verde
    (5, 77, 6),  # Verde scuro
    (221, 177, 73),  # Oro
    (194, 252, 156), # Verde Salvia
    (3, 29, 35), # Blu Avido
    (135, 80, 52), # Terracotta
    (89, 12, 44), # Porpora
    (109, 61, 9), # Marrone
    (232, 98, 64), # Salmone
    (153, 120, 181) # Lilla
]

class RaceTrack:
    """Una tappa del campionato: pista caricata e modello addestrato."""
    def __init__(self, name):
        self.name = name
        self.image, self.spawn_pos, self.base_angle, self.gates = load_track(name)
        self.weights = load_best_weights(name)
        self.scale = view_scale(self.image.get_size())  # Le piste grandi vengono mostrate ridotte
        self.window_size = (round(self.image.get_width() * self.scale), round(self.image.get_height() * self.scale))

def parse_args():
    parser = argparse.ArgumentParser(description="Gara / campionato tra i piloti IA")
    parser.add_argument("piste", nargs="*",
                        help="piste del campionato, in ordine (default: tutte quelle in tracks_config/)")
    parser.add_argument("--giri", type=int, default=LAPS_TO_WIN, help=f"giri per gara (default: {LAPS_TO_WIN})")
    return parser.parse_args()

# --- MAIN: CAMPIONATO ---
def main():
    args = parse_args()
    names = args.piste or list_tracks()
    if not names:
        print("Nessuna pista trovata: creane una con generate_track.py o create_track.py")
        return

    pygame.init()
    tracks = []
    for name in names:
        try:
            race_track = RaceTrack(name)
        except (pygame.error, FileNotFoundError, KeyError) as e:
            print(f"Pista '{name}' saltata: impossibile caricarla ({e})")
            continue
        if race_track.weights is None:
            print(f"Pista '{name}' saltata: nessun modello addestrato. Esegui: python training.py {name}")
            continue
        tracks.append(race_track)
    if not tracks:
        return

    print(f"Campionato di {len(tracks)} gare: {', '.join(t.name for t in tracks)} ({args.giri} giri ciascuna)")
    restore_window_position(WINDOW_KEY)
    screen = pygame.display.set_mode(tracks[0].window_size)
    for t in tracks:
        t.surface = t.image.convert()            # Pista a grandezza reale: la leggono i sensori
        t.view = scaled_view(t.surface, t.scale)  # Pista ridotta per la finestra
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)
    speed = SpeedControl(base_fps=120)  # Rimane impostata per tutto il campionato

    while True:  # Ogni giro di questo ciclo è un campionato completo
        championship = {}  # id pilota -> {"points", "wins", "color"}
        for index, race_track in enumerate(tracks):
            if screen.get_size() != race_track.window_size:
                screen = pygame.display.set_mode(race_track.window_size)
            pygame.display.set_caption(f"GRAN PREMIO IA - Gara {index + 1}/{len(tracks)}: {race_track.name}")

            race_info = f"Gara {index + 1}/{len(tracks)}: {race_track.name}"
            final_standings = run_race(screen, clock, font, speed, race_track, args.giri, race_info)
            if final_standings is None:  # Finestra chiusa durante la gara
                return
            finalize_race(race_track.name, final_standings)

            for i, r in enumerate(final_standings):
                entry = championship.setdefault(r.id, {"points": 0, "wins": 0, "color": r.color})
                entry["points"] += race_points(i)
                entry["wins"] += i == 0

            next_name = tracks[index + 1].name if index + 1 < len(tracks) else None
            if not show_post_race_screen(screen, clock, final_standings, championship,
                                         race_info, next_name, len(tracks)):
                return

def run_race(screen, clock, font, speed, race_track, laps, race_info):
    """Esegue una gara. Ritorna la classifica finale, oppure None se la finestra viene chiusa."""
    width = screen.get_width()
    track, gates, scale = race_track.surface, race_track.gates, race_track.scale
    spawn_pos, base_angle = race_track.spawn_pos, race_track.base_angle

    # Creazione Griglia di Partenza con colori diversi
    racers = []
    for i in range(NUM_RACERS):
        r_id = i + 1
        # Ogni pilota ha il proprio DNA salvato, oppure quello del training con una variazione casuale
        individual_dna = get_driver_weights(race_track.name, r_id, race_track.weights)
        color = BASE_COLORS[i % len(BASE_COLORS)]
        # Griglia di partenza: spostiamo ogni pilota di lato per non sovrapporli
        # (mai indietro, altrimenti ripasserebbe il traguardo al contrario)
        side = pygame.Vector2(1, 0).rotate(base_angle + 90)
        offset_pos = spawn_pos + side * random.uniform(-12, 12)
        racers.append(Racer(individual_dna, color, r_id, offset_pos, base_angle))

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            speed.handle_event(event)

        # Più passi di simulazione per frame quando la velocità è sopra x1
        for _ in range(speed.steps_per_frame):
            for r in racers:
                r.update(track, gates, laps, racers)

        screen.blit(race_track.view, (0, 0))
        for r in racers:
            pos = (int(r.pos.x * scale), int(r.pos.y * scale))
            # Colore pilota o grigio se morto
            color = r.color if r.alive else (50, 50, 50)
            pygame.draw.circle(screen, color, pos, 6)

            # DISEGNO STATO GOMME (Cerchio esterno)
            if r.alive:
                # Colore da Verde (100%) a Rosso (0%)
                tyre_color = (
                    int(255 * (1 - r.tyre_health / 100)),  # Rosso aumenta
                    int(255 * (r.tyre_health / 100)),      # Verde diminuisce
                    0
                )
                # Disegna un anello intorno alla macchina
                pygame.draw.circle(screen, tyre_color, pos, 9, 2)

        # Classifica aggiornata con i giri
        racers.sort(key=lambda x: (x.progress(len(gates)), x.score), reverse=True)
        for i, r in enumerate(racers[:5]):
            status = "VINTO!" if r.completed else ("OUT" if not r.alive else f"Giro {r.laps + 1}/{laps}")
            entry = font.render(f"{i+1}. Pilota {r.id}: {status}", True, readable(r.color))
            screen.blit(entry, (width - 200, 20 + i * 25))

        info = font.render(f"{race_info} | {speed.label()}", True, (255, 255, 255))
        pygame.draw.rect(screen, (0, 0, 0), (6, 6, info.get_width() + 12, info.get_height() + 6))
        screen.blit(info, (12, 9))

        pygame.display.flip()
        speed.tick(clock)

        if not any(r.alive and not r.completed for r in racers):
            return racers  # Già ordinati per giri e punteggio

def finalize_race(track_name, final_standings):
    """Evoluzione del DNA e punti nella classifica storica: va chiamata UNA sola volta per gara."""
    # 1. Evoluzione DNA (separato per ogni pista)
    winner = final_standings[0]
    print(f"\n[{track_name}] Vince Pilota {winner.id}!")

    for i, r in enumerate(final_standings):
        if i == 0:
            print(f"1° Pilota {r.id}: DNA conservato")
        elif i < 3:
            r.weights += np.random.normal(0, 0.01, r.weights.shape)
            print(f"{i+1}° Pilota {r.id}: DNA + mutazione")
        else:
            r.weights = (winner.weights * 0.8) + (r.weights * 0.2)
            r.weights += np.random.normal(0, 0.03, r.weights.shape)
            print(f"{i+1}° Pilota {r.id}: apprende dal vincitore")
        save_driver_weights(track_name, r.id, r.weights)

    # 2. Classifica storica (somma di tutte le gare mai disputate)
    leaderboard = safe_load_json(LEADERBOARD_FILE)

    print("\n--- PUNTI ASSEGNATI ---")
    for i, r in enumerate(final_standings):
        punti = race_points(i)
        r_name = f"Pilota {r.id}"

        entry = leaderboard.get(r_name, {"score": 0, "color": list(r.color)})
        entry["score"] += punti
        entry["color"] = list(r.color)
        leaderboard[r_name] = entry

        print(f"{i+1}° {r_name} ({r.color}): +{punti} pts (tot: {entry['score']})")

    safe_save_json(LEADERBOARD_FILE, leaderboard)

def show_post_race_screen(screen, clock, final_standings, championship, race_info, next_name, race_count):
    """
    Classifica della gara, del campionato in corso e storica.
    Ritorna True per proseguire (gara successiva o nuovo campionato), False per uscire.
    """
    width, height = screen.get_size()
    col_x = [50, width // 3 + 20, 2 * width // 3 + 10]

    leaderboard = safe_load_json(LEADERBOARD_FILE)
    storica = sorted(leaderboard.items(), key=lambda x: x[1]['score'], reverse=True)
    campionato = sorted(championship.items(), key=lambda x: (x[1]["points"], x[1]["wins"]), reverse=True)
    finished = next_name is None

    title_font = pygame.font.SysFont("Arial", 30, bold=True)
    banner_font = pygame.font.SysFont("Arial", 34, bold=True)
    pos_font = pygame.font.SysFont("Arial", 24, bold=True)
    data_font = pygame.font.SysFont("Arial", 20)
    hint_font = pygame.font.SysFont("Arial", 22)
    stats_font = pygame.font.SysFont("Arial", 18)

    total = len(final_standings)
    survived = len([r for r in final_standings if r.alive])
    winners = len([r for r in final_standings if r.completed])

    if finished:
        hint_text = "SPACE = Nuovo Campionato | ESC = Esci" if race_count > 1 else "SPACE = Nuova Gara | ESC = Esci"
    else:
        hint_text = f"SPACE = Prossima gara: {next_name} | ESC = Esci"

    def draw_row(x, i, name, line2, color, highlight):
        pos_text = pos_font.render(f"{i+1}°", True, highlight if i == 0 else (255, 255, 255))
        screen.blit(pos_text, (x + 10, 140 + i * 50))
        screen.blit(data_font.render(name, True, readable(color)), (x + 70, 135 + i * 50))
        screen.blit(data_font.render(line2, True, (220, 220, 220)), (x + 70, 157 + i * 50))
        pygame.draw.circle(screen, color, (x + 55, 148 + i * 50), 8)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                return True

        screen.fill((15, 15, 25))  # Sfondo scuro elegante

        # ========== BANNER ==========
        if finished and race_count > 1:
            champ_id, champ = campionato[0]
            banner = banner_font.render(f"CAMPIONATO CONCLUSO - Campione: Pilota {champ_id} ({champ['points']} pts)",
                                        True, champ["color"])
        else:
            banner = banner_font.render(race_info, True, (255, 255, 255))
        screen.blit(banner, ((width - banner.get_width()) // 2, 20))

        # ========== INTESTAZIONI ==========
        headers = [("CLASSIFICA GARA", (255, 215, 0)), ("CAMPIONATO", (0, 255, 150)), ("STORICA", (120, 180, 255))]
        for x, (text, color) in zip(col_x, headers):
            screen.blit(title_font.render(text, True, color), (x, 75))
            pygame.draw.line(screen, color, (x, 115), (x + 380, 115), 3)

        # ========== CLASSIFICA GARA ==========
        for i, r in enumerate(final_standings[:10]):
            draw_row(col_x[0], i, f"Pilota {r.id}", f"Score {r.score:.0f} | +{race_points(i)} pts", r.color, (255, 215, 0))

        # ========== CAMPIONATO IN CORSO ==========
        for i, (r_id, info) in enumerate(campionato[:10]):
            wins = f"{info['wins']} vittori{'a' if info['wins'] == 1 else 'e'}"
            draw_row(col_x[1], i, f"Pilota {r_id}", f"{info['points']} pts | {wins}",
                     info["color"], (0, 255, 150))

        # ========== CLASSIFICA STORICA ==========
        for i, (name, info) in enumerate(storica[:10]):
            draw_row(col_x[2], i, name, f"{info['score']} pts", tuple(info["color"]), (120, 180, 255))

        # ========== ISTRUZIONI ==========
        hint = hint_font.render(hint_text, True, (150, 200, 255))
        screen.blit(hint, ((width - hint.get_width()) // 2, height - 60))

        # Statistiche in basso
        stats = stats_font.render(f"Sopravvissuti: {survived}/{total} | Completati: {winners}/{total}", True, (255, 150, 150))
        screen.blit(stats, (50, height - 90))

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    try:
        main()
    finally:
        close_window(WINDOW_KEY)
