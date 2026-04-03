import pygame
import numpy as np
import math
import random
import pickle
import os

# --- CONFIG ---
# WIDTH, HEIGHT = 800, 600
POP_SIZE = 30
SENSOR_COUNT = 5

TRACK_NAME = "pista_gara"

# LOGICA DI SALVATAGGIO ---
def save_model(brain):
    with open(f"tracks/{TRACK_NAME}.pkl", "wb") as f:
        pickle.dump(brain.weights, f)
    print(">>> Progresso salvato in checkpoints.pkl")

def load_model():
    if os.path.exists(f"tracks/{TRACK_NAME}.pkl"):
        with open(f"tracks/{TRACK_NAME}.pkl", "rb") as f:
            print(">>> Modello precedente caricato con successo!")
            return pickle.load(f)
    return None

# --- BRAIN ---
class Brain:
    def __init__(self, spawn_pos, base_angle, weights=None):
        self.spawn_pos = spawn_pos
        self.base_angle = base_angle
        
        self.next_cp = 0
        
        # Inizializza i pesi: 5 sensori in ingresso, 2 decisioni in uscita (sterzo, velocità)
        if weights is None:
            self.weights = np.random.uniform(-1, 1, (SENSOR_COUNT, 2))
        else:
            self.weights = weights
        
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

    def predict(self, sensors):
        output = np.dot(sensors, self.weights)
        output = np.tanh(output)

        steer = output[0]
        speed = output[1]

        # CONFIDENCE = quanto il cervello “vede chiaro”
        # (più i sensori sono forti/contrastati → più è sicuro)
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
def run_simulation(population, track, screen, clock, font, generation, spawn_pos, base_angle, checkpoints):
    running = True
    finish_count = 0 # Conta quanti hanno finito il giro
    frame_count = 0  # Timer della simulazione
    skip_generation = False

    for brain in population:
        brain.reset(spawn_pos, base_angle)

    while running:
        frame_count += 1
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:  # premi S per skippare
                    skip_generation = True
            
            if skip_generation:
                # uccide tutti quelli ancora vivi (non verranno considerati bene nello score)
                for brain in population:
                    if brain.alive and not brain.completed:
                        brain.alive = False
                        brain.score -= 1000  # penalità forte per evitare che vengano scelti

                running = False
                        
            if event.type == pygame.QUIT:
                pygame.quit(); exit()

        screen.blit(track, (0, 0))
        alive_count = 0

        for brain in population:
            if not brain.alive or brain.completed:
                continue

            alive_count += 1
            sensors = get_sensors(brain.pos, brain.angle, track)
            action = brain.predict(sensors)

            # Decisioni
            steer_raw = action[0]
            speed_raw = action[1]

            # --- CONFIDENCE GATE ---
            # finché non è sicuro, va piano e sterza più forte
            safe_zone = 0.35  # soglia regolabile

            if brain.confidence < safe_zone:
                speed = 1.5 + (speed_raw + 1) * 0.5
                steer = steer_raw * 7
            else:
                speed = 2 + (speed_raw + 1) * 5
                steer = steer_raw * (5 + speed * 0.3)

            # applica sterzo dipendente dalla velocità
            brain.angle += steer
            
            brain.angle += (speed * 0.15) * steer_raw
            
            rad = math.radians(brain.angle)
            old_pos = brain.pos.copy()
            brain.pos += pygame.Vector2(math.cos(rad), math.sin(rad)) * speed
            
            if old_pos.distance_to(brain.pos) < 0.5:
                brain.stuck_timer += 1
            else:
                brain.stuck_timer = 0

            # punizione se si incastra
            if brain.stuck_timer > 20:
                brain.score -= 20
                brain.angle += random.uniform(-30, 30)
                brain.stuck_timer = 0
            
            brain.score += old_pos.distance_to(brain.pos)
            brain.score -= 0.1 
            
            # --- UNICA LOGICA DI ARRIVO: I CHECKPOINT ---
            if brain.next_cp < len(checkpoints):
                target_cp = checkpoints[brain.next_cp]
                dist_to_cp = brain.pos.distance_to(pygame.Vector2(target_cp))
                
                if dist_to_cp < 70: 
                    brain.score += 5000 
                    brain.next_cp += 1 

                    # IL TRAGUARDO
                    if brain.next_cp >= len(checkpoints):
                        finish_count += 1
                        brain.completed = True
                        
                        # PREMIO POSIZIONE: Il 1° prende più del 2°, ecc.
                        # Esempio: 1° = 10.000, 2° = 9.000, 3° = 8.000...
                        premio_posizione = max(1000, 10000 - (finish_count - 1) * 1000)
                        
                        # PREMIO VELOCITÀ: Bonus basato sui frame totali (chi corre forte vince di più)
                        premio_velocita = max(500, 5000 - frame_count)
                        
                        brain.score += (premio_posizione + premio_velocita)
                        
                        print(f"PILOTA {population.index(brain)} ARRIVATO! Posizione: {finish_count} | Tempo: {frame_count}")
                        # Se vuoi che la generazione finisca appena il PRIMO arriva:
                        
            else:
                # Caso di sicurezza: se per qualche motivo l'indice è già fuori, completa
                brain.completed = True

            # --- COLLISIONE ---
            try:
                if track.get_at((int(brain.pos.x), int(brain.pos.y))).r < 30:
                    brain.alive = False
                    brain.score -= 50 # Penalità pesante per chi sbatte
            except:
                brain.alive = False

            color = (255, 0, 0) if brain.alive else (100, 100, 100)
            pygame.draw.circle(screen, color, (int(brain.pos.x), int(brain.pos.y)), 4)

        # Se tutti sono morti o hanno finito, chiudiamo la generazione
        if alive_count == 0:
            running = False

        # Disegno info
        txt = font.render(f"Gen: {generation} | Vivi: {alive_count} | Arrivati: {finish_count}", True, (0,255,0))
        screen.blit(txt, (10, 10))
        pygame.display.flip()
        clock.tick(240) # Aumentato a 120 per velocizzare l'allenamento visivo

# --- EVOLUZIONE ---
def evolve(population, spawn_pos, base_angle):
    population.sort(key=lambda b: b.score, reverse=True)

    print(f"Best score: {int(population[0].score)}")

    elite = population[:5]
    new_pop = elite.copy()

    while len(new_pop) < POP_SIZE:
        parent = random.choice(elite)
        # Passa spawn_pos e base_angle, poi i nuovi pesi
        new_weights = parent.weights + np.random.normal(0, 0.1, parent.weights.shape)
        new_pop.append(Brain(spawn_pos, base_angle, weights=new_weights))

    return new_pop


# --- MAIN ---
def main():
    pygame.init()
    # track = pygame.image.load("circuit.png").convert()
    track_temp = pygame.image.load("pista_gara.png")
    WIDTH, HEIGHT = track_temp.get_size()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    track = track_temp.convert()
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 20)

    track = pygame.transform.scale(track, (WIDTH, HEIGHT))
    
    # 1. Caricamento Immagine
    try:
        track_temp = pygame.image.load(f"{TRACK_NAME}.png")
        WIDTH, HEIGHT = track_temp.get_size()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        track = track_temp.convert()
    except pygame.error:
        print("Errore: Immagine pista_gara.png non trovata!")
        return

    # 2. Caricamento Configurazione (Unico file necessario)
    try:
        with open(f"tracks_config/{TRACK_NAME}.pkl", "rb") as f:
            config = pickle.load(f)
        
        # Estraiamo tutto dal dizionario
        checkpoints = config["checkpoints"]
        spawn_pos = pygame.Vector2(config["spawn_pos"])
        base_angle = config["base_angle"]
        
        print(f"Track caricato correttamente!")
        print(f"Checkpoints: {len(checkpoints)} | Start Angle: {base_angle:.1f}°")
        
    except (FileNotFoundError, KeyError):
        print("Errore: Il file track_config.pkl è assente o corrotto. Rigenera la pista!")
        return

    
    spawn_pos = find_spawn(track)
    base_angle = -90
    
    generation = 0 
    
    saved_weights = load_model()

    population = [Brain(spawn_pos, base_angle) for _ in range(POP_SIZE)]
    if saved_weights is not None:
        population[0].weights = saved_weights

    while True:
        run_simulation(population, track, screen, clock, font, generation, spawn_pos, base_angle, checkpoints)
        # Salva il migliore di ogni generazione automaticamente
        save_model(population[0]) 
        population = evolve(population, spawn_pos, base_angle)
        generation += 1

def find_spawn(track):
    width, height = track.get_size()

    for x in range(width):
        for y in range(height):
            color = track.get_at((x, y))
            if color[0] == 0 and color[1] == 255 and color[2] == 0:
                return pygame.Vector2(x, y)

    return pygame.Vector2(400, 300)  # fallback

if __name__ == "__main__":
    main()