import pygame
import numpy as np
import math
import pickle
import os
import json

# --- CONFIGURAZIONE GARA ---
# WIDTH, HEIGHT = 800, 600
NUM_RACERS = 20  # Numero di partecipanti alla gara
SENSOR_COUNT = 5
WIDTH, HEIGHT = 800, 600

TRACK_NAME = "pista_gara"

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
        self.can_score_lap = False # Impedisce di contare giri infiniti stando fermi
        self.finish_time = 0
        
        self.learning_rate = 0.01  # Quanto velocemente cambia idea
        self.prev_dist_to_walls = 0
        
        self.velocity = 0
        self.prev_steer = 0

    def predict(self, sensors):
        output = np.dot(sensors, self.weights)
        return np.tanh(output)

    def update(self, track_image, spawn_pos, total_laps):
        if not self.alive or self.completed:
            return

        self.time += 1
        # Lettura sensori (stessa logica del training)
        sensors = self.get_sensors(track_image)
        
        self.learn_on_the_fly(sensors)
        action = self.predict(sensors)

        # Movimento
        # --- CONTROLLO ---
        steer_raw = action[0]
        speed_raw = action[1]

        # sterzo base
        steer = steer_raw * 5

        # velocità base (come training)
        base_speed = 2 + (speed_raw + 1) * 5

        # --- PENALITÀ CURVA ---
        turn_intensity = abs(steer_raw)
        curve_penalty = 1.0 - (turn_intensity * 0.8)

        # bonus rettilineo
        straight_bonus = 1.0 + ((1.0 - turn_intensity) * 0.5)

        speed = base_speed * curve_penalty * straight_bonus

        # soft cap velocità
        speed = speed * (1.0 - (speed / 10.0) * 0.5)
        speed = max(0.5, min(speed, 8))

        # --- GRIP (meno sterzo ad alta velocità) ---
        grip = max(0.2, 1.0 - self.velocity * 0.08)
        self.angle += steer * grip

        # --- INERZIA ---
        acceleration = 0.03
        brake_force = 0.08

        if speed > self.velocity:
            self.velocity += (speed - self.velocity) * acceleration
        else:
            self.velocity += (speed - self.velocity) * brake_force

        # movimento reale
        rad = math.radians(self.angle)
        old_pos = self.pos.copy()
        self.pos += pygame.Vector2(math.cos(rad), math.sin(rad)) * self.velocity

        # --- SCORE MOVIMENTO ---
        self.score += old_pos.distance_to(self.pos) * 0.5

        # penalità zig-zag
        steer_change = abs(steer_raw - self.prev_steer)
        self.score -= steer_change * 2.0

        # penalità sterzate continue
        self.score -= abs(steer_raw) * 0.2

        self.prev_steer = steer_raw
        
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
                
                # penalità forte se si schianta veloce
                self.score -= 50
                self.score -= self.velocity * 200
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
                    pixel = track.get_at((x, y))
                    if pixel.r < 50 and pixel.g < 50 and pixel.b < 50:
                        break
                except: break
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
def load_best_weights():
    path = f"tracks/{TRACK_NAME}.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
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
    track_temp = pygame.image.load(f"{TRACK_NAME}.png")
    WIDTH, HEIGHT = track_temp.get_size()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    track = track_temp.convert()
    pygame.display.set_caption("GRAN PREMIO IA - Simulazione Finale")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)

    with open(f"tracks_config/{TRACK_NAME}.pkl", "rb") as f:
        config = pickle.load(f)

    spawn_pos = pygame.Vector2(config["spawn_pos"])
    base_angle = config["base_angle"]
    
    base_angle = -90

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

    for i in range(NUM_RACERS):
        r_id = i + 1
        # Ogni pilota ha il DNA del migliore + una piccola variazione casuale (0.05)
        individual_dna = get_driver_weights(r_id, trained_weights)
        if not os.path.exists(f"single/{TRACK_NAME}/driver_{r_id}.pkl"):
            individual_dna += np.random.normal(0, 0.05, individual_dna.shape)
        color = BASE_COLORS[i % len(BASE_COLORS)]
        # Griglia di partenza: spostiamo leggermente ogni pilota per non sovrapporli
        offset_pos = spawn_pos + pygame.Vector2(random.randint(-10, 10), random.randint(-10, 10))
        racers.append(Racer(individual_dna, color, r_id, offset_pos, base_angle))

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
            # lap_txt = font.render(f"{r.laps}/{LAPS_TO_WIN}", True, (255, 255, 255))
            # screen.blit(lap_txt, (r.pos.x - 10, r.pos.y + 10))

        # Classifica aggiornata con i giri
        racers.sort(key=lambda x: (x.laps, x.score), reverse=True)
        for i, r in enumerate(racers[:5]):
            status = "VINTO!" if r.completed else ("OUT" if not r.alive else f"Giro {r.laps}")
            entry = font.render(f"{i+1}. Pilota {r.id}: {status}", True, r.color)
            screen.blit(entry, (WIDTH - 180, 20 + i * 25))

        pygame.display.flip()
        clock.tick(120)

        active = [r for r in racers if r.alive and not r.completed]
        if not active:
            final_standings = finalize_race(racers)
            show_post_race_screen(screen, font, final_standings)
            running = False
    
    pygame.quit()
    
def get_driver_weights(racer_id, base_weights):
    filename = f"single/driver_{racer_id}.pkl"
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return base_weights.copy() # Se è nuovo, parte dal DNA del training

def save_driver_weights(racer_id, weights):
    filename = f"single/driver_{racer_id}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(weights, f)
    
def finalize_race(racers):
    """Classifica DEFINITIVA - una volta sola per tutto!"""
    # ORDINA UNA SOLA VOLTA
    final_standings = sorted(racers, key=lambda x: (x.laps, x.score), reverse=True)
    
    # 1. Evoluzione DNA
    winner = final_standings[0]
    print(f"\n🏁 Vince Pilota {winner.id}!")
    
    for i, r in enumerate(final_standings):
        if i == 0:
            save_driver_weights(r.id, r.weights)
            print(f"1° Pilota {r.id}: DNA conservato")
        elif i < 3:
            r.weights += np.random.normal(0, 0.01, r.weights.shape)
            save_driver_weights(r.id, r.weights)
            print(f"{i+1}° Pilota {r.id}: DNA + mutazione")
        else:
            r.weights = (winner.weights * 0.8) + (r.weights * 0.2)
            r.weights += np.random.normal(0, 0.03, r.weights.shape)
            save_driver_weights(r.id, r.weights)
            print(f"{i+1}° Pilota {r.id}: apprende dal vincitore")
    
    # 2. Assegnazione punti
    points_system = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1] + [0] * 10
    filename = "leaderboard.json"
    
    if os.path.exists(filename):
        with open(filename, "r") as f:
            leaderboard = json.load(f)
    else:
        leaderboard = {}
    
    print("\n--- 🎯 PUNTI ASSEGNATI ---")
    for i, r in enumerate(final_standings):
        punti = points_system[i]
        r_name = f"Pilota {r.id}"
        
        entry = leaderboard.get(r_name, {"score": 0, "color": list(r.color)})
        entry["score"] += punti
        entry["color"] = list(r.color)
        leaderboard[r_name] = entry
        
        print(f"{i+1}° {r_name} ({r.color}): +{punti} pts (tot: {entry['score']})")
    
    safe_save_json(filename, leaderboard)
    
    return final_standings  # ← Ritorna la classifica per usarla ovunque!

def show_post_race_screen(screen, font, racers):
    # CARICA classifica già calcolata
    final_standings = finalize_race(racers)  # ✅ Usa quella corretta!
    
    # Carica generale
    filename = "leaderboard.json"
    if os.path.exists(filename):
        with open(filename, "r") as f:
            leaderboard = json.load(f)
    generale_classifica = sorted(leaderboard.items(), key=lambda x: x[1]['score'], reverse=True)
        
def draw_final_standings(screen, font):
    filename = "leaderboard.json"
    if not os.path.exists(filename): return

    with open(filename, "r") as f:
        data = json.load(f)
    
    # Ordina la classifica storica per punteggio decrescente
    sorted_standings = sorted(data.items(), key=lambda x: x[1]['score'], reverse=True)

    # Crea un overlay scuro
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 200)) 
    screen.blit(overlay, (0,0))

    title = font.render("CLASSIFICA GENERALE CAMPIONATO", True, (255, 255, 0))
    screen.blit(title, (WIDTH//2 - 150, 100))

    for i, (name, info) in enumerate(sorted_standings[:20]):
        # Usiamo il colore salvato nel JSON per il testo!
        color = tuple(info["color"])
        txt = font.render(f"{i+1}. {name}: {info['score']} pts", True, color)
        screen.blit(txt, (WIDTH//2 - 120, 150 + i * 35))
        
        # Disegniamo anche un quadratino colorato accanto
        pygame.draw.rect(screen, color, (WIDTH//2 - 150, 155 + i * 35, 15, 15))

    pygame.display.flip()
    # Attendi 5 secondi prima di chiudere o riavviare
    pygame.time.delay(5000)

def show_post_race_screen(screen, font, racers):
    # Calcola classifica gara
    final_standings = finalize_race(racers)
    
    # Carica generale
    leaderboard = safe_load_json("leaderboard.json")
    generale = sorted(leaderboard.items(), key=lambda x: x[1]['score'], reverse=True)
    
    clock = pygame.time.Clock()
    waiting = True
    
    while waiting:
        screen.fill((15, 15, 25))  # Sfondo scuro elegante
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                waiting = False
        
        # ========== HEADER ==========
        title_font = pygame.font.SysFont("Arial", 32, bold=True)
        title1 = title_font.render("🏁 CLASSIFICA GARA", True, (255, 215, 0))
        title2 = title_font.render("🏆 CAMPIONATO", True, (0, 255, 150))
        screen.blit(title1, (50, 30))
        screen.blit(title2, (450, 30))
        
        # Linee decorative
        pygame.draw.line(screen, (255, 215, 0), (50, 70), (350, 70), 3)
        pygame.draw.line(screen, (0, 255, 150), (450, 70), (750, 70), 3)
        
        # ========== CLASSIFICA GARA (con punti gara + totale) ==========
        pos_font = pygame.font.SysFont("Arial", 24, bold=True)
        data_font = pygame.font.SysFont("Arial", 20)
        
        points_system = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1] + [0] * 10
        
        for i, r in enumerate(final_standings[:10]):
            pos_color = (255, 215, 0) if i == 0 else (255, 255, 255) if i < 3 else (200, 200, 200)
            
            # Posizione grande
            pos_text = pos_font.render(f"{i+1}°", True, pos_color)
            screen.blit(pos_text, (60, 100 + i * 35))
            
            # Dati pilota
            pts_gara = points_system[i]
            pts_totali = r.score  # Score della gara corrente (distanza percorsa)
            
            line1 = data_font.render(f"Pilota {r.id}", True, r.color)
            line2 = data_font.render(f"📏 {pts_totali:.0f} | +{pts_gara}pts", True, (220, 220, 220))
            
            screen.blit(line1, (120, 95 + i * 35))
            screen.blit(line2, (120, 115 + i * 35))
            
            # Cerchio colorato
            pygame.draw.circle(screen, r.color, (105, 110 + i * 35), 8)
        
        # ========== CAMPIONATO ==========
        for i, (name, info) in enumerate(generale[:10]):
            color = tuple(info["color"])
            
            pos_text = pos_font.render(f"{i+1}°", True, (0, 255, 150) if i == 0 else (255, 255, 255))
            screen.blit(pos_text, (460, 100 + i * 35))
            
            name_text = data_font.render(name, True, color)
            score_text = data_font.render(f"{info['score']} pts", True, (0, 255, 200))
            
            screen.blit(name_text, (520, 95 + i * 35))
            screen.blit(score_text, (520, 115 + i * 35))
            
            # Quadrato colorato
            pygame.draw.rect(screen, color, (505, 105 + i * 35, 12, 12))
        
        # ========== ISTRUZIONI ==========
        hint_font = pygame.font.SysFont("Arial", 22)
        hint = hint_font.render("⏸️ SPACE = Nuova Gara | ESC = Esci", True, (150, 200, 255))
        screen.blit(hint, (WIDTH//2 - 160, HEIGHT - 60))
        
        # Statistiche in basso
        stats_font = pygame.font.SysFont("Arial", 18)
        survived = len([r for r in racers if r.alive])
        winners = len([r for r in racers if r.completed])
        stats = stats_font.render(f"💀 Sopravvissuti: {survived}/20 | 🏆 Completati: {winners}/20", True, (255, 150, 150))
        screen.blit(stats, (50, HEIGHT - 90))
        
        pygame.display.flip()
        clock.tick(60)

def safe_load_json(filename):
    """Carica JSON con gestione errori"""
    if not os.path.exists(filename):
        return {}
    
    try:
        with open(filename, "r") as f:
            content = f.read().strip()
            if not content:  # File vuoto
                return {}
            return json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError):
        print(f"❌ File {filename} corrotto. Reset!")
        return {}  # Reset se corrotto

def safe_save_json(filename, data):
    temp_file = filename + ".backup"
    with open(temp_file, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    os.replace(temp_file, filename)

import random # Necessario per i colori
if __name__ == "__main__":
    main()
