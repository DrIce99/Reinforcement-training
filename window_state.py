# window_state.py
# Salva la posizione della finestra pygame alla chiusura e la ripristina all'avvio successivo.
import json
import os
import sys
import warnings

import pygame

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "window_state.json")


def _load_all():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _is_on_screen(x, y):
    """Evita di aprire la finestra fuori dallo schermo (es. monitor scollegato)."""
    if sys.platform != "win32":
        return True
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        user32.MonitorFromPoint.argtypes = [wintypes.POINT, wintypes.DWORD]
        user32.MonitorFromPoint.restype = wintypes.HMONITOR
        MONITOR_DEFAULTTONULL = 0
        # Punto vicino all'angolo in alto a sinistra: basta che sia visibile per poter spostare la finestra
        return bool(user32.MonitorFromPoint(wintypes.POINT(x + 40, y + 10), MONITOR_DEFAULTTONULL))
    except Exception:
        return True


def _get_window_position():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            if hasattr(pygame, "Window"):  # pygame-ce
                window = pygame.Window.from_display_module()
            else:  # pygame classico
                from pygame._sdl2.video import Window
                window = Window.from_display_module()
            x, y = window.position
        return int(x), int(y)
    except Exception:
        return None


def restore_window_position(key):
    """Da chiamare PRIMA di pygame.display.set_mode()."""
    pos = _load_all().get(key)
    if (isinstance(pos, list) and len(pos) == 2
            and all(isinstance(v, int) for v in pos) and _is_on_screen(*pos)):
        os.environ["SDL_VIDEO_WINDOW_POS"] = f"{pos[0]},{pos[1]}"


def save_window_position(key):
    """Da chiamare PRIMA di pygame.quit(), finché la finestra esiste ancora."""
    if not pygame.display.get_init() or pygame.display.get_surface() is None:
        return
    if not pygame.display.get_active():  # finestra ridotta a icona: posizione non significativa
        return
    pos = _get_window_position()
    if pos is None:
        return

    data = _load_all()
    data[key] = list(pos)
    temp_file = STATE_FILE + ".tmp"
    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        os.replace(temp_file, STATE_FILE)
    except OSError as e:
        print(f"Impossibile salvare la posizione della finestra: {e}")


def close_window(key):
    """Salva la posizione della finestra e chiude pygame."""
    save_window_position(key)
    pygame.quit()
