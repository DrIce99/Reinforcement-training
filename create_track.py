import pygame
import math
import random
import pickle
import numpy as np

# --- CONFIGURAZIONE MASSIMA ---
WIDTH, HEIGHT = 2500, 1800 
TRACK_WIDTH = 80
COLLISION_THRESHOLD = TRACK_WIDTH * 1.5

def point_to_segment_distance(p, a, b):
    ap = p - a
    ab = b - a
    ab_len = ab.length_squared()
    if ab_len == 0:
        return ap.length()
    t = max(0, min(1, ap.dot(ab) / ab_len))
    closest = a + ab * t
    return (p - closest).length()

class TrackGenerator:
    def __init__(self):
        self.points = []
        # Partiamo in un quadrante specifico per iniziare il giro largo
        self.current_pos = pygame.Vector2(400, 300)
        self.spawn_pos = self.current_pos.copy()
        self.current_angle = 0
        
        # Waypoints obbligatori per forzare l'uso di tutta la mappa
        self.waypoints = [
            pygame.Vector2(WIDTH - 400, 300),    # Angolo Top-Right
            pygame.Vector2(WIDTH - 400, HEIGHT - 300), # Angolo Bottom-Right
            pygame.Vector2(400, HEIGHT - 300),   # Angolo Bottom-Left
            self.spawn_pos                       # Ritorno
        ]

    def add_rettilineo(self, length):
        step = 25
        for _ in range(0, int(length), step):
            next_p = self.current_pos + pygame.Vector2(math.cos(math.radians(self.current_angle)), 
                                                       math.sin(math.radians(self.current_angle))) * step
            
            # Limiti fisici (margine di sicurezza 100px)
            if not (100 < next_p.x < WIDTH-100 and 100 < next_p.y < HEIGHT-100): return False
            
            # Auto-collisione (escludiamo i punti recenti)
            if len(self.points) > 10:
                for i in range(len(self.points) - 10):
                    a = pygame.Vector2(self.points[i])
                    b = pygame.Vector2(self.points[i + 1])
                    if point_to_segment_distance(next_p, a, b) < COLLISION_THRESHOLD:
                        return False
                    
            self.current_pos = next_p
            self.points.append((self.current_pos.x, self.current_pos.y))
        return True

    def steer_towards(self, target):
        """Sterza gradualmente verso un punto target"""
        vec_to_target = target - self.current_pos
        target_angle = math.degrees(math.atan2(vec_to_target.y, vec_to_target.x))
        
        # Calcola la differenza d'angolo minima
        diff = (target_angle - self.current_angle + 180) % 360 - 180
        
        # Applica una curva morbida o un rettilineo
        steer_limit = random.randint(15, 45)
        
        angle_to_apply = max(-steer_limit, min(steer_limit, diff))
        
        # Definisci lo step (5 o -5)
        angle_step = 5 if angle_to_apply > 0 else -5
        
        # Applica l'angolo e aggiungi il tratto
        self.current_angle += angle_step
        return self.add_rettilineo(30)

    def build_track(self):
        base_angle = 0
        self.add_rettilineo(400) # Rettilineo Box
        
        # Per ogni waypoint obbligatorio, navighiamo aggiungendo tratti casuali
        for wp in self.waypoints:
            dist_to_wp = self.current_pos.distance_to(wp)
            attempts = 0
            max_attempts = 500

            while dist_to_wp > 300 and attempts < max_attempts:
                attempts += 1
                # 70% delle volte punta al waypoint, 30% fa una manovra casuale
                if random.random() < 0.7:
                    self.steer_towards(wp)
                else:
                    move_type = random.choice(["chicane", "curva", "rettilineo"])

                    if move_type == "chicane":
                        orig = self.current_angle
                        self.current_angle += random.choice([30, -30])
                        if self.add_rettilineo(80):
                            self.current_angle -= random.choice([30, -30])
                            self.add_rettilineo(80)
                        else:
                            self.current_angle = orig

                    elif move_type == "curva":
                        self.current_angle += random.randint(-20, 20)
                        self.add_rettilineo(random.randint(60, 140))

                    else:  # rettilineo
                        self.add_rettilineo(random.randint(100, 250))
                        
                dist_to_wp = self.current_pos.distance_to(wp)
                if len(self.points) > 5000: break # Sicurezza

        if attempts >= max_attempts:
            print("⚠️ Bloccato, forzo direzione verso waypoint")
            direction = (wp - self.current_pos).normalize()
            self.current_angle = math.degrees(math.atan2(direction.y, direction.x))
            self.add_rettilineo(200)
        
        # Chiusura finale
        self.points.append((self.spawn_pos.x, self.spawn_pos.y))
        return self.spawn_pos, base_angle, len(self.points) // 100

def main():
    pygame.init()
    gen = TrackGenerator()
    spawn, angle, complessita = gen.build_track()
    
    surface = pygame.Surface((WIDTH, HEIGHT))
    surface.fill((0, 0, 0))

    # Disegno Alta Risoluzione
    for i in range(len(gen.points)-1):
        p1, p2 = gen.points[i], gen.points[i+1]
        pygame.draw.line(surface, (255, 255, 255), p1, p2, TRACK_WIDTH)
        pygame.draw.circle(surface, (255, 255, 255), (int(p1[0]), int(p1[1])), TRACK_WIDTH // 2)

    # Checkpoints densi
    checkpoints = [gen.points[i] for i in range(0, len(gen.points), 12)]
    pygame.draw.circle(surface, (0, 255, 0), (int(spawn.x), int(spawn.y)), 20)

    with open("track_config.pkl", "wb") as f:
        pickle.dump({"checkpoints": checkpoints, "spawn_pos": (spawn.x, spawn.y), "base_angle": angle}, f)
    
    pygame.image.save(surface, "pista_gara.png")
    print(f"CIRCUITO TOTAL-MAP GENERATO: {WIDTH}x{HEIGHT}")
    pygame.quit()

if __name__ == "__main__":
    main()
