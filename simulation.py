import pygame
import numpy as np
import math
import pickle
import os

# --- CONFIGURAZIONE GARA ---
# WIDTH, HEIGHT = 800, 600
NUM_RACERS = 10  # Numero di partecipanti alla gara
SENSOR_COUNT = 5

# --- CLASSE PILOTA GARA ---
class Racer:
    def __init__(self, weights, color, racer_id, spawn_pos, base_angle):
        self.weights = weights
        self.color = color
        self.id = racer_id
        self.pos = pygame.Vector2(spawn_pos)
        self.angle = base_angle
        self.alive = True
        self.completed = False
        self.score = 0
        self.time = 0
        
        self.laps = 0
        self.can_score_lap = False # Impedisce di contare giri infiniti stando fermi
        self.finish_time = 0

    def predict(self, sensors):
        output = np.dot(sensors, self.weights)
        return np.tanh(output)

    def update(self, track_image, spawn_pos, total_laps):
        if not self.alive or self.completed:
            return

        self.time += 1
        # Lettura sensori (stessa logica del training)
        sensors = self.get_sensors(track_image)
        action = self.predict(sensors)

        # Movimento
        steer = action[0] * 5
        speed = max(2, (action[1] + 1) * 4)
        
        self.angle += steer
        rad = math.radians(self.angle)
        self.pos += pygame.Vector2(math.cos(rad), math.sin(rad)) * speed
        self.score += speed
        
        dist_to_spawn = self.pos.distance_to(spawn_pos)
        
        if dist_to_spawn > 150:
            self.can_score_lap = True
            
        # 2. Se torna vicino allo spawn ed è "abilitato", conta il giro
        if self.can_score_lap and dist_to_spawn < 40:
            self.laps += 1
            self.can_score_lap = False
            print(f"Pilota {self.id} ha completato il giro {self.laps}!")
            
            if self.laps >= total_laps:
                self.completed = True
                self.finish_time = self.time # Salva il tempo totale

        # Controllo Collisione (Nero = Muro)
        try:
            pixel = track_image.get_at((int(self.pos.x), int(self.pos.y)))
            if pixel.r < 30 and pixel.g < 30 and pixel.b < 30:
                self.alive = False
        except IndexError:
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
                    if track.get_at((x, y)).r < 30: break
                except: break
            sensors.append(dist / max_dist)
        return np.array(sensors)

# --- FUNZIONI DI SUPPORTO ---
def load_best_weights():
    if os.path.exists("best_brain.pkl"):
        with open("best_brain.pkl", "rb") as f:
            return pickle.load(f)
    return None

def find_spawn(track):
    for x in range(track.get_width()):
        for y in range(track.get_height()):
            pixel = track.get_at((x, y))
            if pixel.g == 255 and pixel.r == 0: return pygame.Vector2(x, y)
    return pygame.Vector2(400, 300)

# --- MAIN GARA ---
def main():
    LAPS_TO_WIN = 3
    
    pygame.init()
    # track = pygame.image.load("circuit.png").convert()
    track_temp = pygame.image.load("pista_gara.png")
    WIDTH, HEIGHT = track_temp.get_size()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    track = track_temp.convert()
    pygame.display.set_caption("GRAN PREMIO IA - Simulazione Finale")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)

    spawn_pos = find_spawn(track)
    base_angle = -135 # Regola in base alla direzione del tuo circuito

    # Caricamento intelligenza appresa
    trained_weights = load_best_weights()
    if trained_weights is None:
        print("Errore: Nessun addestramento trovato! Esegui prima training.py")
        return

    # Creazione Griglia di Partenza con colori diversi
    racers = []

    BASE_COLORS = [
        (255, 0, 0),   # Rosso
        (0, 0, 255),   # Blu
        (255, 255, 0), # Giallo
        (255, 165, 0), # Arancione
        (0, 255, 255), # Ciano
        (255, 0, 255), # Magenta
        (128, 0, 128), # Viola
        (0, 128, 128), # Ottanio
        (255, 255, 255), # Bianco
        (200, 200, 200)  # Grigio chiaro
    ]

    for i in range(NUM_RACERS):
        # Ogni pilota ha il DNA del migliore + una piccola variazione casuale (0.05)
        dna = trained_weights + np.random.normal(0, 0.05, trained_weights.shape)
        color = BASE_COLORS[i % len(BASE_COLORS)]
        # Griglia di partenza: spostiamo leggermente ogni pilota per non sovrapporli
        offset_pos = spawn_pos + pygame.Vector2(random.randint(-10, 10), random.randint(-10, 10))
        racers.append(Racer(dna, color, i+1, offset_pos, base_angle))

    running = True
    while running:
        screen.blit(track, (0, 0))
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False

        # Aggiornamento e Disegno
        active_racers = [r for r in racers if r.alive and not r.completed]
        for r in racers:
            r.update(track, spawn_pos, LAPS_TO_WIN)
            color = r.color if r.alive else (50, 50, 50)
            pygame.draw.circle(screen, color, (int(r.pos.x), int(r.pos.y)), 6)
            
            # Mostra Giri/Totale sopra ogni pilota
            lap_txt = font.render(f"{r.laps}/{LAPS_TO_WIN}", True, (255, 255, 255))
            screen.blit(lap_txt, (r.pos.x - 10, r.pos.y + 10))

        # Classifica aggiornata con i giri
        racers.sort(key=lambda x: (x.laps, x.score), reverse=True)
        for i, r in enumerate(racers[:5]):
            status = "VINTO!" if r.completed else ("OUT" if not r.alive else f"Giro {r.laps}")
            entry = font.render(f"{i+1}. Pilota {r.id}: {status}", True, r.color)
            screen.blit(entry, (WIDTH - 180, 20 + i * 25))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

import random # Necessario per i colori
if __name__ == "__main__":
    main()
