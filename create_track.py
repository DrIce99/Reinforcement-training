import pygame
import random
import math

def generate_track():
    pygame.init()
    width, height = 800, 600
    # Creiamo la superficie principale
    surface = pygame.Surface((width, height))

    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)

    surface.fill(BLACK)

    # --- PARAMETRI DEL CIRCUITO ---
    center = (width // 2, height // 2)
    points = []
    num_points = 15 # Numero di "curve" principali
    base_radius = 200
    track_width = 50 # Larghezza della carreggiata

    # --- 1. GENERA PUNTI CASUALI PER UN CIRCUITO MORBIDO ---
    for i in range(num_points):
        angle = (i / num_points) * 2 * math.pi
        # Rumore casuale per rendere la pista varia ma non spezzata
        radius = base_radius + random.randint(-70, 70)
        
        x = int(center[0] + math.cos(angle) * radius)
        y = int(center[1] + math.sin(angle) * radius)
        points.append((x, y))

    # Chiudiamo il loop riportando all'inizio
    points.append(points[0])

    # --- 2. DISEGNA LA PISTA BIANCA ---
    # Usiamo un trucco: disegniamo cerchi su ogni punto e linee spesse tra loro
    # per evitare che rimangano "buchi" neri nelle curve strette.
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i+1]
        # Disegna il segmento
        pygame.draw.line(surface, WHITE, p1, p2, track_width)
        # Disegna il "giunto" circolare per smussare l'angolo
        pygame.draw.circle(surface, WHITE, p1, track_width // 2)

    # --- 3. POSIZIONA LO SPAWN VERDE ---
    # Lo mettiamo sul primo punto generato (che è sicuramente sulla pista)
    spawn_pos = points[0]
    pygame.draw.circle(surface, GREEN, spawn_pos, 8)

    # --- 4. SALVA L'IMMAGINE ---
    pygame.image.save(surface, "circuit.png")
    print(f"Circuito creato con successo! Spawn trovato a: {spawn_pos}")

    pygame.quit()

if __name__ == "__main__":
    generate_track()
