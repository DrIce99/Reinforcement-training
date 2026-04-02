import pygame
import math
import random
import pickle
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import ConvexHull
import os

# --- CONFIGURAZIONE ---
WIDTH, HEIGHT = 1800, 1000 
TRACK_WIDTH = 80

class TrackGenerator:
    def __init__(self):
        self.points = []

    def build_track(self):
        margin = 150
        num_initial_points = 18 # Aumentato leggermente per varietà
        
        # 1. Genera punti casuali nell'area
        random_points = [
            [random.randint(margin, WIDTH - margin), random.randint(margin, HEIGHT - margin)]
            for _ in range(num_initial_points)
        ]
        
        # 2. Crea l'anello base usando il Convex Hull (evita incroci)
        hull = ConvexHull(random_points)
        base_ring = [pygame.Vector2(random_points[i][0], random_points[i][1]) for i in hull.vertices]
        
        # 3. Trasforma rettilinei in curve (Distorsione Singola)
        distorted_points = []
        for i in range(len(base_ring)):
            p1 = base_ring[i]
            p2 = base_ring[(i + 1) % len(base_ring)]
            
            distorted_points.append((p1.x, p1.y))
            
            # Se il segmento è lungo, aggiungi un "gomito"
            if p1.distance_to(p2) > 80:
                mid = (p1 + p2) / 2
                direction = (p2 - p1).normalize()
                normal = pygame.Vector2(-direction.y, direction.x)
                
                # Spostamento casuale (interno o esterno)
                offset = random.uniform(-120, 120)
                curve_point = mid + normal * offset
                distorted_points.append((curve_point.x, curve_point.y))

        # 4. Smoothing Finale con Spline
        self.points = self.apply_smoothing(distorted_points)
        
        # Calcolo angolo di spawn (direzione tra primo e secondo punto)
        p1 = pygame.Vector2(self.points[0])
        p2 = pygame.Vector2(self.points[1])
        angle = math.degrees(math.atan2(p2.y - p1.y, p2.x - p1.x))
        
        return self.points, angle, len(self.points)

    def apply_smoothing(self, points):
        pts = np.array(points)
        # s=80-120 offre un ottimo bilanciamento tra fedeltà e morbidezza
        tck, u = splprep([pts[:,0], pts[:,1]], s=100, per=True)
        u_new = np.linspace(0, 1, 1500)
        new_x, new_y = splev(u_new, tck)
        return list(zip(new_x, new_y))

def main():
    pygame.init()
    gen = TrackGenerator()
    punti_smooth, angle, _ = gen.build_track()
    
    # Primo punto per lo spawn
    spawn_pos = punti_smooth[0]
    
    surface = pygame.Surface((WIDTH, HEIGHT))
    surface.fill((0, 0, 0)) # Verde scuro erba

    # 1. Disegno Pista (Asfalto e Bordi)
    for i in range(len(punti_smooth)):
        p1 = (int(punti_smooth[i][0]), int(punti_smooth[i][1]))
        p2 = (int(punti_smooth[(i+1)%len(punti_smooth)][0]), int(punti_smooth[(i+1)%len(punti_smooth)][1]))
        
        pygame.draw.circle(surface, (255, 255, 255), p1, TRACK_WIDTH // 2)

    # 2. Linea di Partenza (Traguardo)
    p1_start = pygame.Vector2(punti_smooth[0])
    p2_start = pygame.Vector2(punti_smooth[1])
    direction = (p2_start - p1_start).normalize()
    normal = pygame.Vector2(-direction.y, direction.x)
    line_start = p1_start + normal * (TRACK_WIDTH // 2)
    line_end = p1_start - normal * (TRACK_WIDTH // 2)
    pygame.draw.line(surface, (255, 255, 255), line_start, line_end, 10)

    # 3. Pallino VERDE dello SPAWN
    spawn_coords = (int(spawn_pos[0]), int(spawn_pos[1]))
    pygame.draw.circle(surface, (0, 255, 0), spawn_coords, 15)

    # 4. Salvataggio Dati e Immagine
    if not os.path.exists("tracks_config"): os.makedirs("tracks_config")
    
    # Checkpoints campionati ogni 30 punti per il sistema di gara
    checkpoints = [punti_smooth[i] for i in range(0, len(punti_smooth), 30)]
    
    with open("tracks_config/pista_gara1.pkl", "wb") as f:
        pickle.dump({
            "checkpoints": checkpoints, 
            "spawn_pos": spawn_pos, 
            "base_angle": angle
        }, f)
    
    pygame.image.save(surface, "pista_gara1.png")
    print(f"Circuito generato correttamente. Spawn pos: {spawn_coords}")
    pygame.quit()

if __name__ == "__main__":
    main()
