import pygame
import math
import numpy as np

# Configurazione Sensori
NUM_SENSORS = 5
SENSOR_ANGLE_RANGE = 120 # Ventaglio di 120 gradi (60 a dx e 60 a sx)
MAX_SENSOR_LEN = 200     # Lunghezza massima della "vista"

class Pilot:
    def __init__(self, x, y):
        self.pos = pygame.Vector2(x, y)
        self.angle = 0
        self.speed = 0
        self.alive = True
        self.sensors = [0.0] * NUM_SENSORS # Output dei sensori (0.0 a 1.0)

    def cast_sensors(self, screen_map):
        start_angle = self.angle - (SENSOR_ANGLE_RANGE / 2)
        step = SENSOR_ANGLE_RANGE / (NUM_SENSORS - 1)

        for i in range(NUM_SENSORS):
            angle_rad = math.radians(start_angle + (i * step))
            dist = 0
            
            while dist < MAX_SENSOR_LEN:
                dist += 4
                test_x = int(self.pos.x + math.cos(angle_rad) * dist)
                test_y = int(self.pos.y + math.sin(angle_rad) * dist)

                try:
                    pixel = screen_map.get_at((test_x, test_y))
                    if pixel.r < 50 and pixel.g < 50 and pixel.b < 50:
                        break
                except IndexError:
                    break
            
            self.sensors[i] = dist / MAX_SENSOR_LEN

    def update(self, action):
        """Applica la decisione della NPU (accelerazione e sterzata)"""
        if not self.alive: return

        # Esempio: action[0] = sterzata, action[1] = accelerazione
        self.angle += action[0] * 5
        self.speed = max(2, action[1] * 10)
        
        # Movimento
        velocity = pygame.Vector2(math.cos(math.radians(self.angle)), 
                                  math.sin(math.radians(self.angle))) * self.speed
        self.pos += velocity

# --- LOGICA NPU (Integrazione OpenVINO) ---
# Durante la gara, passerai self.sensors alla NPU così:
# input_data = np.array(pilot.sensors).reshape(1, 5)
# output = compiled_model(input_data)
# pilot.update(output)
