# speed_control.py
# Velocità di simulazione regolabile con le frecce (su/destra = più veloce, giù/sinistra = più lenta).
import pygame

SPEED_LEVELS = [0.25, 0.5, 1, 2, 4, 8, 16, 32]
FASTER_KEYS = (pygame.K_UP, pygame.K_RIGHT)
SLOWER_KEYS = (pygame.K_DOWN, pygame.K_LEFT)


class SpeedControl:
    """
    Sotto x1 rallenta il frame rate; sopra x1 esegue più passi di simulazione per ogni
    frame disegnato, così la fisica resta identica a qualsiasi velocità.
    """

    def __init__(self, base_fps, level=1):
        self.base_fps = base_fps
        self.index = SPEED_LEVELS.index(level)

    @property
    def multiplier(self):
        return SPEED_LEVELS[self.index]

    @property
    def steps_per_frame(self):
        return max(1, int(self.multiplier))

    def handle_event(self, event):
        """Ritorna True se l'evento ha cambiato la velocità."""
        if event.type != pygame.KEYDOWN:
            return False
        if event.key in FASTER_KEYS and self.index < len(SPEED_LEVELS) - 1:
            self.index += 1
            return True
        if event.key in SLOWER_KEYS and self.index > 0:
            self.index -= 1
            return True
        return False

    def tick(self, clock):
        clock.tick(self.base_fps * min(1, self.multiplier))

    def label(self):
        m = self.multiplier
        return f"Velocità x{m:g} (frecce)"
