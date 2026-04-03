import pygame
import math
import pickle
import numpy as np
import os

# --- CONFIGURAZIONE ---
WIDTH, HEIGHT = 1600, 900
TRACK_WIDTH = 75
WHITE, GRAY, BLACK, RED, GREEN, BLUE = (255,255,255), (55,55,55), (0,0,0), (255,50,50), (50,255,50), (50,100,255)

class ControlPoint:
    def __init__(self, x, y):
        self.pos = pygame.Vector2(x, y)
        self.h_in = pygame.Vector2(x - 60, y)
        self.h_out = pygame.Vector2(x + 60, y)
        self.dragging = None 
        self.locked_handles = False 

class Editor:
    def __init__(self):
        self.points = []
        self.closed = False
        self.spawn_pos = pygame.Vector2(WIDTH//2, HEIGHT//2)
        self.spawn_angle = 0
        self.setting_spawn = False

    def get_bezier_points(self, steps=30):
        if len(self.points) < 2: return []
        res = []
        pts = list(self.points)
        if self.closed: pts.append(self.points[0])
        
        for i in range(len(pts) - 1):
            p1, p2 = pts[i], pts[i+1]
            for t in np.linspace(0, 1, steps):
                t = float(t)
                p0, p1_c, p2_c, p3 = p1.pos, p1.h_out, p2.h_in, p2.pos
                x = (1-t)**3*p0.x + 3*(1-t)**2*t*p1_c.x + 3*(1-t)*t**2*p2_c.x + t**3*p3.x
                y = (1-t)**3*p0.y + 3*(1-t)**2*t*p1_c.y + 3*(1-t)*t**2*p2_c.y + t**3*p3.y
                res.append((x, y))
        return res

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Track Editor Pro")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("Arial", 16, bold=True)

        while True:
            m_pos = pygame.Vector2(pygame.mouse.get_pos())
            for event in pygame.event.get():
                if event.type == pygame.QUIT: return
                if event.type == pygame.MOUSEBUTTONDOWN:
                    for p in self.points:
                        if m_pos.distance_to(p.pos) < 15: p.dragging = "pos"; break
                        if m_pos.distance_to(p.h_in) < 10: p.dragging = "in"; break
                        if m_pos.distance_to(p.h_out) < 10: p.dragging = "out"; break
                    else:
                        if not self.setting_spawn: self.points.append(ControlPoint(m_pos.x, m_pos.y))
                if event.type == pygame.MOUSEBUTTONUP:
                    for p in self.points: p.dragging = None
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_c: self.closed = not self.closed
                    if event.key == pygame.K_l:
                        for p in self.points: p.locked_handles = not p.locked_handles
                    if event.key == pygame.K_s: self.setting_spawn = True
                    if event.key == pygame.K_z and self.points: self.points.pop()
                    if event.key == pygame.K_RETURN: self.save_all()

            for p in self.points:
                if p.dragging == "pos":
                    d = m_pos - p.pos
                    p.pos += d; p.h_in += d; p.h_out += d
                elif p.dragging == "in":
                    p.h_in = pygame.Vector2(m_pos)
                    if p.locked_handles: p.h_out = p.pos + (p.pos - p.h_in)
                elif p.dragging == "out":
                    p.h_out = pygame.Vector2(m_pos)
                    if p.locked_handles: p.h_in = p.pos + (p.pos - p.h_out)

            if self.setting_spawn:
                if pygame.mouse.get_pressed()[0]: self.spawn_pos = pygame.Vector2(m_pos)
                diff = m_pos - self.spawn_pos
                if diff.length() > 5: self.spawn_angle = math.degrees(math.atan2(-diff.y, diff.x))
                if event.type == pygame.MOUSEBUTTONUP: self.setting_spawn = False

            screen.fill((0, 0, 0))
            bezier_pts = self.get_bezier_points(steps=30)

            if len(bezier_pts) > 1:
                # pygame.draw.lines(screen, (200, 200, 200), False, bezier_pts, TRACK_WIDTH + 10)
                pygame.draw.lines(screen, WHITE, False, bezier_pts, TRACK_WIDTH)

            for p in self.points:
                pygame.draw.line(screen, (255,255,255), p.h_in, p.h_out, 1)
                pygame.draw.circle(screen, WHITE, (int(p.pos.x), int(p.pos.y)), 12)
                pygame.draw.circle(screen, RED, (int(p.h_in.x), int(p.h_in.y)), 8)
                pygame.draw.circle(screen, BLUE, (int(p.h_out.x), int(p.h_out.y)), 8)

            rad = math.radians(-self.spawn_angle)
            end = self.spawn_pos + pygame.Vector2(math.cos(rad), math.sin(rad)) * 60
            pygame.draw.line(screen, GREEN, self.spawn_pos, end, 6)
            pygame.draw.circle(screen, GREEN, (int(self.spawn_pos.x), int(self.spawn_pos.y)), 15)

            pygame.display.flip()
            clock.tick(60)

    def save_all(self):
        if not os.path.exists("tracks_config"): os.makedirs("tracks_config")
        pts = self.get_bezier_points(steps=80)
        
        surf = pygame.Surface((WIDTH, HEIGHT))
        surf.fill((0, 0, 0))
        
        if len(pts) > 1:
            for p in pts: pygame.draw.circle(surf, (255, 255, 255), (int(p[0]), int(p[1])), TRACK_WIDTH // 2)
            # Traguardo Bianco
            self.draw_finish_line(surf, self.spawn_pos, self.spawn_angle, color=(0, 255, 0), width=12)

        pygame.image.save(surf, "pista_gara.png")
        with open("tracks_config/pista_gara.pkl", "wb") as f:
            pickle.dump({
                "checkpoints": pts[::20], 
                "spawn_pos": (self.spawn_pos.x, self.spawn_pos.y), 
                "base_angle": self.spawn_angle
            }, f)
        print("Salvataggio completato: Immagine con traguardo linea creata.")

    def draw_finish_line(self, surface, pos, angle, color=(0, 255, 0), width=10):
        """Disegna la linea di traguardo perpendicolare alla direzione"""
        rad = math.radians(-angle)
        direction = pygame.Vector2(math.cos(rad), math.sin(rad))
        normal = pygame.Vector2(-direction.y, direction.x)
        l_start = pos + normal * (TRACK_WIDTH // 2)
        l_end = pos - normal * (TRACK_WIDTH // 2)
        pygame.draw.line(surface, color, l_start, l_end, width)
        return l_start, l_end # Utile per calcolare hitbox se necessario

if __name__ == "__main__":
    Editor().run()
