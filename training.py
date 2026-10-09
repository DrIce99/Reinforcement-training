import pygame
import numpy as np
import math
import random
import pickle
import os
import sys

from physics import compute_step
from track import (SENSOR_COUNT, INPUT_COUNT, load_track, navigation_inputs, adapt_weights,
                   view_scale, scaled_view)
from speed_control import SpeedControl
from window_state import restore_window_position, close_window

# --- CONFIG ---
POP_SIZE = 60

# Uso: python training.py [nome_pista]
TRACK_NAME = sys.argv[1] if len(sys.argv) > 1 else "pista_gara"
WINDOW_KEY = "training"
MAX_FRAMES_WITHOUT_CP = 1000  # Elimina chi gira a vuoto senza raggiungere il checkpoint successivo
WRONG_GATE_PENALTY = 3000     # Strada sbagliata a un incrocio, taglio o contromano
REFERENCE_GATES = 36          # Gate della pista su cui è stato tarato il premio velocità

# LOGICA DI SALVATAGGIO ---
def save_model(brain):
    os.makedirs("tracks", exist_ok=True)
    with open(f"tracks/{TRACK_NAME}.pkl", "wb") as f:
        pickle.dump(brain.weights, f)
    print(f">>> Progresso salvato in tracks/{TRACK_NAME}.pkl")

def load_model():
    if os.path.exists(f"tracks/{TRACK_NAME}.pkl"):
        with open(f"tracks/{TRACK_NAME}.pkl", "rb") as f:
            print(">>> Modello precedente caricato con successo!")
            return adapt_weights(pickle.load(f))
    return None

# --- BRAIN ---
class Brain:
    def __init__(self, spawn_pos, base_angle, weights=None, sector_weights=None):
        self.next_cp = 0
        
        # Pesi neurali: 5 sensori + 2 ingressi del navigatore, 2 decisioni in uscita
        if weights is None:
            self.weights = np.random.uniform(-1, 1, (INPUT_COUNT, 2))
        else:
            self.weights = weights
            
        # --- INIZIO MODIFICA PUNTO 2 ---
        # Pesi dei settori (uno per checkpoint)
        # Se non forniti, usa un array di 1 (peso uniforme)
        if sector_weights is None:
            # Nota: qui usiamo SENSOR_COUNT come fallback solo se non abbiamo info sui checkpoint,
            # ma idealmente dovrebbe essere len(checkpoints). 
            # Per sicurezza, lo inizializziamo a 1.0 e verrà sovrascritto in main()
            self.sector_weights = np.ones(10) # Metti un numero ragionevole o passalo sempre
        else:
            self.sector_weights = sector_weights.copy()
        
        # Per tracciare quanti piloti attraversano ogni settore (utile per adattivo)
        self.cp_crossed_flags = np.zeros(len(self.sector_weights))
        # --- FINE MODIFICA PUNTO 2 ---

        self.confidence = 0.0   # quanto è sicuro della direzione
        self.stuck_timer = 0
        self.reset(spawn_pos, base_angle)

    def reset(self, spawn_pos, base_angle):
        self.pos = spawn_pos.copy()
        self.angle = base_angle + random.uniform(-10, 10)
        self.alive = True
        self.score = 0
        self.next_cp = 0
        self.completed = False
        self.velocity = 0
        self.stuck_timer = 0
        self.frames_since_cp = 0
        # Resetta anche i flag dei checkpoint quando si resetta il cervello
        self.cp_crossed_flags[:] = 0 

    def predict(self, sensors, nav):
        output = np.dot(np.concatenate([sensors, nav]), self.weights)
        output = np.tanh(output)
        steer = output[0]
        speed = output[1]
        # CONFIDENCE = quanto il cervello “vede chiaro” (solo dai sensori di distanza)
        self.confidence = float(np.mean(sensors) - np.std(sensors))
        return np.array([steer, speed])

# --- SENSORI ---
def get_sensors(pos, angle, track):
    sensors = []
    spread = 120
    max_dist = 150

    start_angle = angle - spread / 2
    step = spread / (SENSOR_COUNT - 1)

    for i in range(SENSOR_COUNT):
        ray_angle = math.radians(start_angle + i * step)
        dist = 0

        while dist < max_dist:
            dist += 4
            x = int(pos.x + math.cos(ray_angle) * dist)
            y = int(pos.y + math.sin(ray_angle) * dist)

            try:
                color = track.get_at((x, y))
                if color[0] < 50 and color[1] < 50 and color[2] < 50:
                    break
            except:
                break

        sensors.append(dist / max_dist)

    return np.array(sensors)


# --- SIMULAZIONE ---
def step_brain(brain, track, gates):
    """Un passo di simulazione per un pilota. Ritorna True se ha appena completato il giro."""
    sensors = get_sensors(brain.pos, brain.angle, track)
    nav = navigation_inputs(brain.pos, brain.angle, gates, brain.next_cp)
    action = brain.predict(sensors, nav)

    steer_raw, speed_raw = action[0], action[1]
    old_pos = brain.pos.copy()

    # --- FISICA UNIFICATA ---
    brain.angle, brain.velocity = compute_step(
        angle=brain.angle,
        velocity=brain.velocity,
        steer_raw=steer_raw,
        speed_raw=speed_raw,
        confidence=brain.confidence
    )

    # Movimento reale
    rad = math.radians(brain.angle)
    brain.pos += pygame.Vector2(math.cos(rad), math.sin(rad)) * brain.velocity

    # Anti-incastro
    if old_pos.distance_to(brain.pos) < 0.5:
        brain.stuck_timer += 1
    else:
        brain.stuck_timer = 0

    if brain.stuck_timer > 20:
        brain.score -= 20
        brain.angle += random.uniform(-30, 30)
        brain.stuck_timer = 0

    # Punteggio di avanzamento e stabilità
    brain.score += old_pos.distance_to(brain.pos) * 0.5
    brain.score += (1.0 - abs(steer_raw)) * 0.1
    brain.score -= brain.velocity * 0.01

    just_completed = False

    # --- CONTROLLO ATTRAVERSAMENTO CHECKPOINT (GATE INTERSECTION) ---
    if brain.next_cp < len(gates):
        passed, wrong_way = gates.check_progress(old_pos, brain.pos, brain.next_cp)

        if wrong_way:
            # Ha attraversato un gate fuori sequenza: strada sbagliata a un incrocio
            brain.alive = False
            brain.score -= WRONG_GATE_PENALTY
        elif passed:
            sector_idx = brain.next_cp
            base_reward = 8000
            weighted_reward = base_reward * brain.sector_weights[sector_idx]
            brain.score += weighted_reward

            # Tracciamento per pesi adattivi
            brain.cp_crossed_flags[sector_idx] = 1
            brain.next_cp += 1
            brain.frames_since_cp = 0

            if brain.next_cp >= len(gates):
                brain.completed = True
                just_completed = True
    else:
        # Caso di sicurezza: se per qualche motivo l'indice è già fuori, completa
        brain.completed = True

    # Chi non raggiunge il checkpoint successivo in tempo (es. gira in tondo) viene eliminato,
    # altrimenti la generazione non terminerebbe mai
    brain.frames_since_cp += 1
    if not brain.completed and brain.frames_since_cp > MAX_FRAMES_WITHOUT_CP:
        brain.alive = False

    # --- COLLISIONE ---
    try:
        pixel = track.get_at((int(brain.pos.x), int(brain.pos.y)))
        # Un muro è nero: Rosso < 30, Verde < 30, Blu < 30
        if brain.alive and pixel.r < 30 and pixel.g < 30 and pixel.b < 30:
            brain.alive = False

            # penalità base
            brain.score -= 50

            # penalità proporzionale alla velocità
            brain.score -= brain.velocity * 20
    except IndexError:
        brain.alive = False

    return just_completed


def run_simulation(population, track, background, scale, screen, clock, font, generation, spawn_pos, base_angle, gates, speed):
    finish_count = 0 # Conta quanti hanno finito il giro
    frame_count = 0  # Timer della simulazione (in passi, indipendente dalla velocità di visualizzazione)

    for brain in population:
        brain.reset(spawn_pos, base_angle)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()  # La finestra viene chiusa (salvandone la posizione) nel blocco finally
            speed.handle_event(event)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_s:  # premi S per skippare
                # uccide tutti quelli ancora vivi (non verranno considerati bene nello score)
                for brain in population:
                    if brain.alive and not brain.completed:
                        brain.alive = False
                        brain.score -= 1000  # penalità forte per evitare che vengano scelti
                return

        # Più passi di simulazione per frame quando la velocità è sopra x1
        for _ in range(speed.steps_per_frame):
            active = [b for b in population if b.alive and not b.completed]
            if not active:
                break
            frame_count += 1
            for brain in active:
                if step_brain(brain, track, gates):
                    finish_count += 1
                    premio_posizione = max(1000, 10000 - (finish_count - 1) * 1000)
                    # Tempo riportato a una pista di lunghezza standard, così il premio vale anche sulle piste lunghe
                    premio_velocita = max(500, 5000 - frame_count * REFERENCE_GATES / len(gates))
                    brain.score += (premio_posizione + premio_velocita) * brain.sector_weights[-1]
                    print(f"PILOTA {population.index(brain)} ARRIVATO! Posizione: {finish_count}")

        screen.blit(background, (0, 0))
        alive_count = 0
        for brain in population:
            if brain.alive and not brain.completed:
                alive_count += 1
                pygame.draw.circle(screen, (255, 0, 0), (int(brain.pos.x * scale), int(brain.pos.y * scale)), 4)

        # Disegno info
        txt = font.render(f"Gen: {generation} | Vivi: {alive_count} | Arrivati: {finish_count} | "
                          f"{speed.label()} | S = salta generazione", True, (0, 255, 0))
        screen.blit(txt, (10, 10))
        pygame.display.flip()
        speed.tick(clock)

        # Se tutti sono morti o hanno finito, chiudiamo la generazione
        if alive_count == 0:
            return

# --- EVOLUZIONE ---
def evolve(population, spawn_pos, base_angle, sector_weights):
    population.sort(key=lambda b: b.score, reverse=True)
    print(f"Best score: {int(population[0].score)}")
    
    # 1. Ampliamo l'Elite
    ELITE_SIZE = 10
    elite = population[:ELITE_SIZE]
    new_pop = elite.copy()
    
    # 2. Inseriamo individui casuali
    num_random = int(POP_SIZE * 0.10)
    for _ in range(num_random):
        # Passa i sector_weights anche ai nuovi random
        new_pop.append(Brain(spawn_pos, base_angle, sector_weights=sector_weights))
        
    # 3. Generiamo il resto con Crossover e Mutazione
    while len(new_pop) < POP_SIZE:
        p1, p2 = random.sample(elite, 2)
        mask = np.random.rand(*p1.weights.shape) > 0.5
        child_weights = np.where(mask, p1.weights, p2.weights)
        
        mutation_strength = random.uniform(0.05, 0.6)
        child_weights += np.random.normal(0, mutation_strength, child_weights.shape)
        
        # Passa i pesi e i sector_weights al figlio
        new_pop.append(Brain(spawn_pos, base_angle, weights=child_weights, sector_weights=sector_weights))
        
    return new_pop

# --- MAIN ---
def main():
    pygame.init()
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 20)
    speed = SpeedControl(base_fps=240)  # Rimane impostata tra una generazione e l'altra

    # Caricamento pista: immagine, spawn e gate in ordine di percorrenza
    try:
        track_temp, spawn_pos, base_angle, gates = load_track(TRACK_NAME)
    except (pygame.error, FileNotFoundError, KeyError) as e:
        print(f"Errore: impossibile caricare la pista '{TRACK_NAME}': {e}")
        return
    print(f"Pista '{TRACK_NAME}' caricata: {len(gates)} gate")
    restore_window_position(WINDOW_KEY)
    scale = view_scale(track_temp.get_size())  # Le piste grandi vengono mostrate ridotte
    screen = pygame.display.set_mode((round(track_temp.get_width() * scale), round(track_temp.get_height() * scale)))
    pygame.display.set_caption(f"Training IA - {TRACK_NAME}")
    track = track_temp.convert()

    # Sfondo con i gate disegnati (solo grafica: i sensori leggono l'immagine pulita)
    background = track.copy()
    for a, b in gates.gates:
        pygame.draw.line(background, (120, 160, 255), a, b, 2)
    background = scaled_view(background, scale)

    # --- NUOVO: Calcola i pesi dei settori ---
    sector_weights = compute_sector_weights(gates.gates, spawn_pos)
    print(f"Pesi settori iniziali: {[round(float(w), 2) for w in sector_weights]}")

    generation = 0 
    saved_weights = load_model()
    
    # Crea la popolazione iniziale passando i sector_weights
    if saved_weights is not None:
        population = [
            Brain(spawn_pos, base_angle, 
                  weights=saved_weights + np.random.normal(0, 0.05, saved_weights.shape), 
                  sector_weights=sector_weights) # Passa i pesi
            for _ in range(POP_SIZE)
        ]
    else:
        population = [
            Brain(spawn_pos, base_angle, sector_weights=sector_weights) 
            for _ in range(POP_SIZE)
        ]

    while True:
        run_simulation(population, track, background, scale, screen, clock, font, generation, spawn_pos, base_angle, gates, speed)
        
        # Salva il migliore (la popolazione va ordinata per punteggio prima di scegliere)
        population.sort(key=lambda b: b.score, reverse=True)
        save_model(population[0])
        
        # --- NUOVO: Aggiorna i pesi in modo adattivo (opzionale, vedi punto avanzato) ---
        # sector_weights = update_adaptive_weights(sector_weights, population, POP_SIZE)
        
        # Passa sector_weights a evolve
        population = evolve(population, spawn_pos, base_angle, sector_weights)
        generation += 1

def compute_sector_weights(checkpoints, spawn_pos):
    """
    Calcola il peso di ogni settore (tra checkpoint i-1 e i).
    Combina:
      - Fattore progressivo (settori finali valgono fino a 1.5x)
      - Fattore distanza (settori lunghi valgono di più)
    """
    n = len(checkpoints)
    weights = np.ones(n)
    
    # Centro di ogni gate (per stimare la lunghezza del settore)
    gate_centers = []
    for cp in checkpoints:
        p1, p2 = cp
        mid = pygame.Vector2((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        gate_centers.append(mid)
    
    prev_point = spawn_pos
    for i in range(n):
        # Distanza del settore i
        dist = prev_point.distance_to(gate_centers[i])
        dist_factor = max(0.5, min(2.0, dist / 100.0))   # 100px = peso 1.0
        
        # Progressivo: da 1.0 (settore 0) a 1.5 (settore finale)
        prog_factor = 1.0 + (i / max(1, n - 1)) * 0.5
        
        weights[i] = dist_factor * prog_factor
        prev_point = gate_centers[i]
    
    # Normalizza così che la media resti ~1 (scala simile al 8000 originale)
    weights = weights * (n / weights.sum())
    return weights

if __name__ == "__main__":
    try:
        main()
    finally:
        close_window(WINDOW_KEY)
