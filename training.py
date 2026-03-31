import pygame
import numpy as np
import math
import random

# --- CONFIG ---
WIDTH, HEIGHT = 800, 600
POP_SIZE = 30
SENSOR_COUNT = 5

# --- BRAIN ---
class Brain:
    def __init__(self, spawn_pos, base_angle, weights=None):
        self.spawn_pos = spawn_pos
        self.base_angle = base_angle
        
        # Inizializza i pesi: 5 sensori in ingresso, 2 decisioni in uscita (sterzo, velocità)
        if weights is None:
            self.weights = np.random.uniform(-1, 1, (SENSOR_COUNT, 2))
        else:
            self.weights = weights

        self.reset(spawn_pos, base_angle)

    def reset(self, spawn_pos, base_angle):
        self.pos = spawn_pos.copy()
        self.angle = base_angle + random.uniform(-10, 10)
        self.alive = True
        self.score = 0
        self.completed = False

    def predict(self, sensors):
        # Il "cervello" moltiplica i sensori per i pesi
        # Ritorna un array con due valori: [sterzo, velocità]
        output = np.dot(sensors, self.weights)
        return np.tanh(output) # Normalizza l'output tra -1 e 1

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
def run_simulation(population, track, screen, clock, font, generation, spawn_pos, base_angle):
    running = True

    for brain in population:
        brain.reset(spawn_pos, base_angle)
        brain.completed = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        screen.blit(track, (0, 0))

        alive_count = 0

        for brain in population:
            if not brain.alive:
                continue

            alive_count += 1

            sensors = get_sensors(brain.pos, brain.angle, track)
            action = brain.predict(sensors)

            steer = action[0] * 5
            speed = max(2, (action[1] + 1) * 2)

            brain.angle += steer
            rad = math.radians(brain.angle)

            old_pos = brain.pos.copy()
            brain.pos += pygame.Vector2(math.cos(rad), math.sin(rad)) * speed
            
            dist_moved = old_pos.distance_to(brain.pos)
            brain.score += dist_moved

            dist_to_start = brain.pos.distance_to(spawn_pos)
            if brain.score > 500 and dist_to_start < 30:
                brain.score += 5000 # Bonus enorme
                brain.completed = True
                print(f"PALLINO HA COMPLETATO IL GIRO! Passaggio alla Gen {generation + 1}")
                return

            # collisione
            try:
                color = track.get_at((int(brain.pos.x), int(brain.pos.y)))
                if color == (0, 0, 0, 255):
                    brain.alive = False
                    brain.score -= 10
            except:
                brain.alive = False

            pygame.draw.circle(screen, (255, 0, 0),
                               (int(brain.pos.x), int(brain.pos.y)), 4)

        text = font.render(f"Gen: {generation} | Alive: {alive_count}", True, (0,255,0))
        screen.blit(text, (10, 10))

        pygame.display.flip()
        clock.tick(60)

        if alive_count == 0:
            running = False


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
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 20)

    track = pygame.image.load("circuit.png").convert()
    track = pygame.transform.scale(track, (WIDTH, HEIGHT))
    
    spawn_pos = find_spawn(track)
    base_angle = -135

    population = [Brain(spawn_pos, base_angle) for _ in range(POP_SIZE)]

    generation = 0

    while True:
        run_simulation(population, track, screen, clock, font, generation, spawn_pos, base_angle)
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