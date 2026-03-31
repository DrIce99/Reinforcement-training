import pygame
import random
import math
import numpy as np
import pickle

def get_bezier_point(p0, p1, p2, p3, t):
    px = (1-t)**3 * p0[0] + 3*(1-t)**2 * t * p1[0] + 3*(1-t) * t**2 * p2[0] + t**3 * p3[0]
    py = (1-t)**3 * p0[1] + 3*(1-t)**2 * t * p1[1] + 3*(1-t) * t**2 * p2[1] + t**3 * p3[1]
    return (px, py)

def generate_track():
    pygame.init()
    width, height = 800, 600
    surface = pygame.Surface((width, height))
    BLACK, WHITE, GREEN = (0,0,0), (255,255,255), (0,255,0)
    surface.fill(BLACK)

    # 1. DEFINIAMO IL RETTILINEO DI PARTENZA/ARRIVO
    # Lo spawn sarà a metà di questo rettilineo
    start_line_y = 100
    p_start = (200, start_line_y)  # Inizio rettilineo
    p_end = (600, start_line_y)    # Fine rettilineo (direzione marcia: verso destra)
    spawn_pos = (400, start_line_y)

    # 2. PUNTI DI CONTROLLO CASUALI (per il resto del circuito)
    # Creiamo un "arco" di punti che partono da p_end e tornano a p_start
    control_points = [p_end]
    control_points.append((750, 200))
    control_points.append((700, 500))
    control_points.append((400, 550))
    control_points.append((100, 500))
    control_points.append((50, 200))
    control_points.append(p_start)

    # 3. GENERAZIONE PUNTI PISTA (Bézier + Rettilineo)
    track_points = []
    
    # Aggiungiamo il rettilineo iniziale manualmente per essere sicuri della direzione
    for x in np.linspace(p_start[0], p_end[0], 10):
        track_points.append((x, start_line_y))

    # Curve di Bézier per il resto
    for i in range(len(control_points)-1):
        p0 = control_points[i]
        p3 = control_points[i+1]
        # Punti di controllo intermedi "morbidi"
        p1 = (p0[0] + (p3[0]-p0[0])*0.5, p0[1] + random.randint(-100,100))
        p2 = (p3[0] - (p3[0]-p0[0])*0.5, p3[1] + random.randint(-100,100))
        for t in np.linspace(0, 1, 15):
            track_points.append(get_bezier_point(p0, p1, p2, p3, t))

    # 4. DISEGNA PISTA
    track_width = 50
    for i in range(len(track_points)-1):
        pygame.draw.line(surface, WHITE, track_points[i], track_points[i+1], track_width)
        pygame.draw.circle(surface, WHITE, (int(track_points[i][0]), int(track_points[i][1])), track_width//2)

    # 5. CHECKPOINTS AUTOMATICI
    checkpoints = []
    for i in range(0, len(track_points), 12):
        checkpoints.append((int(track_points[i][0]), int(track_points[i][1])))
    
    with open("checkpoints.pkl", "wb") as f:
        pickle.dump(checkpoints, f)

    # 6. SPAWN VERDE (esattamente sul rettilineo)
    pygame.draw.circle(surface, GREEN, spawn_pos, 8)

    pygame.image.save(surface, "circuit.png")
    print(f"Circuito con rettilineo creato! Direzione: DESTRA (0 gradi)")
    pygame.quit()

if __name__ == "__main__":
    generate_track()
