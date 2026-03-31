import pygame
import numpy as np
import pickle
import openvino as ov # Importante per la NPU

# Inizializzazione OpenVINO (NPU)
core = ov.Core()
# Nota: Qui caricheremmo un modello IR/ONNX. Per brevità usiamo la CPU 
# se non hai ancora convertito il file .pkl in .xml/.bin (OpenVINO)
# device = "NPU" if "NPU" in core.available_devices else "CPU"

class Car:
    def __init__(self, weights, id):
        self.pos = [100, 300]
        self.angle = 0
        self.weights = weights
        self.id = id
        self.alive = True
        self.distance = 0

    def get_sensors(self, track_surface):
        # Legge i 5 pixel davanti/lato e ritorna distanze 0-1
        return np.random.rand(5) # Placeholder per la funzione cast_sensors vista prima

    def update(self):
        if not self.alive: return
        sensors = self.get_sensors(track)
        # Qui la NPU calcola la mossa
        action = np.dot(sensors, self.weights) 
        self.angle += action[0] * 5
        self.pos[0] += np.cos(np.radians(self.angle)) * 5
        self.pos[1] += np.sin(np.radians(self.angle)) * 5
        self.distance += 1

# --- MAIN LOOP ---
pygame.init()
screen = pygame.display.set_mode((800, 600))
track = pygame.image.load("circuit.png") # Carica il tuo disegno (Nero=Muro, Bianco=Pista)

with open("best_pilot.pkl", "rb") as f:
    best_weights = pickle.load(f)

# Creiamo 10 piloti basati sul migliore, ma con piccole differenze
racers = [Car(best_weights + np.random.rand(5,2)*0.05, i) for i in range(10)]

running = True
while running:
    screen.blit(track, (0,0))
    for r in racers:
        r.update()
        pygame.draw.circle(screen, (255, 0, 0), (int(r.pos[0]), int(r.pos[1])), 5)
        # Controllo collisione
        if track.get_at((int(r.pos[0]), int(r.pos[1]))) == (0,0,0): r.alive = False
    
    pygame.display.flip()
    # Logica Classifica basata su r.distance...
