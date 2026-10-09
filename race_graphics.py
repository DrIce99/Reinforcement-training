# race_graphics.py
"""
Grafica della gara in stile MotoGP: pista decorata, piloti, intestazione, avvisi in sovrimpressione
e torre dei tempi (classifica live in una finestra separata).
Tutto è solo visivo: i sensori delle auto leggono sempre l'immagine originale della pista.
"""
import math
from collections import deque

import numpy as np
import pygame

from race_timing import TREND_SHOWN

# --- TEMA ---
FONT_NAMES = "bahnschrift,segoeui,arial"   # Bahnschrift: condensato e sportivo, simile al font MotoGP
MOTOGP_RED = (226, 20, 40)
PANEL = (16, 16, 22)
ROW_A = (24, 24, 31)
ROW_B = (31, 31, 40)
WHITE = (245, 245, 245)
GREY = (140, 140, 150)
DIM = (90, 90, 100)
GREEN = (40, 215, 95)
RED = (240, 65, 65)
ORANGE = (255, 160, 40)
GOLD = (255, 200, 40)
SECTOR_YELLOW = (250, 205, 30)
BLACK = (10, 10, 12)

_fonts = {}
_texts = {}


def font(size, bold=True):
    key = (size, bold)
    if key not in _fonts:
        _fonts[key] = pygame.font.SysFont(FONT_NAMES, size, bold=bold)
    return _fonts[key]


def text(string, size, color, bold=True):
    """Testo renderizzato con cache (la classifica ridisegna le stesse scritte a ogni frame)."""
    key = (string, size, color, bold)
    surf = _texts.get(key)
    if surf is None:
        if len(_texts) > 4000:
            _texts.clear()
        surf = font(size, bold).render(string, True, color)
        _texts[key] = surf
    return surf


def readable(color, min_brightness=110):
    """Schiarisce i colori troppo scuri per essere letti sullo sfondo scuro."""
    brightness = max(color)
    if brightness >= min_brightness:
        return tuple(color)
    return tuple(int(c + (min_brightness - brightness) + 40) for c in color)


def contrast(color):
    """Testo nero sui colori chiari, bianco su quelli scuri."""
    luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    return BLACK if luminance > 150 else WHITE


def rider_name(racer_id):
    return f"PILOTA {racer_id}"


def format_gap(gap):
    if gap is None:
        return "--"
    kind, value = gap
    if kind == "giri":
        return f"+{value} GIRO" if value == 1 else f"+{value} GIRI"
    return f"+{value:.3f}"


def triangle(surface, center, size, up, color):
    x, y = center
    if up:
        points = [(x, y - size), (x - size, y + size * 0.7), (x + size, y + size * 0.7)]
    else:
        points = [(x, y + size), (x - size, y - size * 0.7), (x + size, y - size * 0.7)]
    pygame.draw.polygon(surface, color, points)


def ease_out(t):
    t = min(1.0, max(0.0, t))
    return 1 - (1 - t) ** 3


def draw_row(surface, x, y, w, h, position, color, name, right=None, right_color=WHITE,
             dim=False, bg=ROW_A, highlight_pos=False):
    """Riga di classifica stile MotoGP: casella posizione, colore squadra, nome, dato a destra."""
    pygame.draw.rect(surface, bg, (x, y, w, h))
    box_color = GOLD if highlight_pos else (WHITE if not dim else GREY)
    pygame.draw.rect(surface, box_color, (x, y, 38, h))
    pos_txt = text(str(position), 18, BLACK)
    surface.blit(pos_txt, (x + 19 - pos_txt.get_width() // 2, y + h // 2 - pos_txt.get_height() // 2))
    pygame.draw.rect(surface, color if not dim else DIM, (x + 38, y, 6, h))
    name_txt = text(name, 17, WHITE if not dim else DIM)
    surface.blit(name_txt, (x + 54, y + h // 2 - name_txt.get_height() // 2))
    if right:
        right_txt = text(right, 17, right_color if not dim else DIM)
        surface.blit(right_txt, (x + w - 12 - right_txt.get_width(), y + h // 2 - right_txt.get_height() // 2))


# --- PISTA ---
def _box_any(mask, r):
    """True dove c'è almeno un True entro un quadrato di lato 2r+1 (dilatazione veloce con l'immagine integrale)."""
    w, h = mask.shape
    integral = np.zeros((w + 1, h + 1), dtype=np.int32)
    integral[1:, 1:] = mask.astype(np.int32).cumsum(0).cumsum(1)
    x0 = np.clip(np.arange(w) - r, 0, w)
    x1 = np.clip(np.arange(w) + r + 1, 0, w)
    y0 = np.clip(np.arange(h) - r, 0, h)
    y1 = np.clip(np.arange(h) + r + 1, 0, h)
    total = (integral[x1][:, y1] - integral[x0][:, y1] - integral[x1][:, y0] + integral[x0][:, y0])
    return total > 0


def build_track_view(track_surface):
    """
    Versione decorata della pista: erba a strisce, vie di fuga in ghiaia, asfalto,
    cordoli rossi e bianchi e traguardo a scacchi.
    """
    arr = pygame.surfarray.array3d(track_surface).astype(np.int16)
    road = arr.max(axis=2) >= 50                    # Stessa soglia dei sensori: sotto è muro
    finish = (arr[..., 1] > 200) & (arr[..., 0] < 80) & (arr[..., 2] < 80)
    w, h = road.shape
    xs, ys = np.meshgrid(np.arange(w), np.arange(h), indexing="ij")
    noise = np.random.default_rng(0).normal(0, 1, (w, h))[..., None]

    stripes = (((xs + ys) // 70) % 2)[..., None]
    out = np.where(stripes == 0, np.array([50, 120, 52]), np.array([58, 132, 58])).astype(float)
    out += noise * 4

    gravel = _box_any(road, 22) & ~road
    out[gravel] = np.array([206, 190, 150]) + noise[gravel] * 7

    out[road] = np.array([70, 71, 77]) + noise[road] * 5

    kerb = road & _box_any(~road, 7)
    kerb_stripe = ((xs - ys) // 16) % 2 == 0
    out[kerb & kerb_stripe] = (222, 32, 38)
    out[kerb & ~kerb_stripe] = (242, 242, 242)

    check = ((xs // 6) + (ys // 6)) % 2 == 0
    out[finish & check] = (250, 250, 250)
    out[finish & ~check] = (20, 20, 20)
    return pygame.surfarray.make_surface(np.clip(out, 0, 255).astype(np.uint8))


def draw_sector_markers(view, track_surface, gates, sector_ends, scale, track_width=75):
    """
    Linee gialle di fine settore ed etichette S1-S4 (all'inizio di ogni settore) sulla vista ridotta.
    L'etichetta va dal lato della linea dove non c'è asfalto, per non coprire la pista.
    """
    w, h = track_surface.get_size()
    starts = [sector_ends[-1]] + sector_ends[:-1]   # Il settore 1 parte dal traguardo
    for number, gate_index in enumerate(starts, start=1):
        a, b = (pygame.Vector2(p) for p in gates[gate_index])
        center = (a + b) / 2
        across = (b - a).normalize()
        half = across * (track_width / 2)
        if number > 1:   # Il traguardo è già disegnato a scacchi
            pygame.draw.line(view, SECTOR_YELLOW, (center - half) * scale, (center + half) * scale, 3)

        side = across
        for candidate in (across, -across):
            probe = center + candidate * (track_width / 2 + 26)
            x, y = int(probe.x), int(probe.y)
            if 0 <= x < w and 0 <= y < h and max(track_surface.get_at((x, y))[:3]) < 50:
                side = candidate
                break
        label = text(f"S{number}", 15, BLACK)
        pos = (center + side * (track_width / 2 + 26)) * scale
        box = pygame.Rect(0, 0, label.get_width() + 12, label.get_height() + 4)
        box.center = (int(pos.x), int(pos.y))
        pygame.draw.rect(view, SECTOR_YELLOW, box)
        view.blit(label, (box.x + 6, box.y + 2))


def draw_rider(surface, pos, color, number, alive=True, leader=False):
    x, y = pos
    if not alive:
        pygame.draw.circle(surface, (40, 40, 40), (x, y), 5)
        pygame.draw.circle(surface, DIM, (x, y), 5, 1)
        return
    pygame.draw.circle(surface, (0, 0, 0), (x + 2, y + 2), 11)       # ombra
    pygame.draw.circle(surface, color, (x, y), 11)
    pygame.draw.circle(surface, GOLD if leader else WHITE, (x, y), 11, 2)
    label = text(str(number), 12, contrast(color))
    surface.blit(label, (x - label.get_width() // 2, y - label.get_height() // 2))


def draw_header(surface, lap_text, race_info, speed_label, sector=None):
    """
    Intestazione in alto a sinistra: giro in corso su fondo rosso, settore del leader in giallo,
    gara e velocità su fondo scuro.
    """
    lap = text(lap_text, 28, WHITE)
    info = text(race_info.upper(), 17, WHITE)
    speed = text(speed_label, 14, GREY, bold=False)
    lap_w = lap.get_width() + 28
    info_w = max(info.get_width(), speed.get_width()) + 28
    pygame.draw.rect(surface, MOTOGP_RED, (14, 14, lap_w, 58))
    surface.blit(lap, (28, 43 - lap.get_height() // 2))
    x = 14 + lap_w
    if sector is not None:
        sec = text(f"S{sector}", 24, BLACK)
        sec_w = sec.get_width() + 22
        pygame.draw.rect(surface, SECTOR_YELLOW, (x, 14, sec_w, 58))
        surface.blit(sec, (x + 11, 43 - sec.get_height() // 2))
        x += sec_w
    panel = pygame.Surface((info_w, 58), pygame.SRCALPHA)
    panel.fill((*PANEL, 225))
    surface.blit(panel, (x, 14))
    surface.blit(info, (x + 14, 20))
    surface.blit(speed, (x + 14, 46))


def draw_hint(surface, string):
    hint = text(string, 14, GREY, bold=False)
    panel = pygame.Surface((hint.get_width() + 16, hint.get_height() + 8), pygame.SRCALPHA)
    panel.fill((*PANEL, 190))
    y = surface.get_height() - panel.get_height() - 10
    surface.blit(panel, (12, y))
    surface.blit(hint, (20, y + 4))


# --- AVVISI IN SOVRIMPRESSIONE ---
class BannerQueue:
    """Avvisi in basso stile grafica TV: sorpassi e variazioni dei distacchi, uno alla volta."""
    DURATION = 3.2   # secondi reali
    SLIDE = 0.3
    MAX_QUEUE = 3

    def __init__(self):
        self.queue = deque()
        self.current = None
        self.started = 0.0

    def push(self, event):
        self.queue.append(event)
        if len(self.queue) > self.MAX_QUEUE:
            # Gli avvisi arrivano più in fretta di quanto si leggano: si scarta quello più in fondo al gruppo
            least = max(self.queue, key=lambda ev: ev.behind_pos)
            self.queue.remove(least)

    def draw(self, surface, now):
        if self.current is not None and now - self.started > self.DURATION:
            self.current = None
        if self.current is None:
            if not self.queue:
                return
            self.current = self.queue.popleft()
            self.started = now
        t = now - self.started
        appear = min(ease_out(t / self.SLIDE), ease_out((self.DURATION - t) / self.SLIDE))
        self._draw_event(surface, self.current, offset=int((1 - appear) * 140))

    def _draw_event(self, surface, ev, offset):
        W, H = 820, 96
        x = (surface.get_width() - W) // 2
        y = surface.get_height() - H - 26 + offset
        chip_colors = {"sorpasso": MOTOGP_RED, "avvicina": (20, 150, 70), "allunga": (210, 120, 20)}
        titles = {"sorpasso": "SORPASSO", "avvicina": "SI AVVICINA", "allunga": "ALLUNGA"}

        panel = pygame.Surface((W, H), pygame.SRCALPHA)
        panel.fill((*PANEL, 235))
        surface.blit(panel, (x, y))
        pygame.draw.rect(surface, chip_colors[ev.kind], (x, y, 190, H))
        title = text(titles[ev.kind], 26, WHITE)

        if ev.kind == "sorpasso":
            surface.blit(title, (x + 95 - title.get_width() // 2, y + 22))
            caption = f"{rider_name(ev.ahead.id).title()} passa {rider_name(ev.behind.id).title()}"
            sub = text(caption, 13, WHITE, bold=False)
            surface.blit(sub, (x + 95 - sub.get_width() // 2, y + 60))
            self._rider_block(surface, x + 210, y + 14, ev.ahead, ev.ahead_pos)
            arrow_x = x + 210 + 290
            triangle(surface, (arrow_x, y + 48), 12, True, GREEN)
            self._rider_block(surface, x + 530, y + 14, ev.behind, ev.behind_pos, dim=True)
            return

        if ev.kind == "avvicina":
            caption = f"{rider_name(ev.behind.id).title()} recupera"
            change_color, up = GREEN, False
        else:
            caption = f"{rider_name(ev.ahead.id).title()} aumenta il vantaggio"
            change_color, up = ORANGE, True
        surface.blit(title, (x + 95 - title.get_width() // 2, y + 10))
        sector = text(f"SETTORE {ev.sector}", 15, WHITE)
        surface.blit(sector, (x + 95 - sector.get_width() // 2, y + 46))
        sub = text(caption, 13, WHITE, bold=False)
        surface.blit(sub, (x + 95 - sub.get_width() // 2, y + 68))

        self._rider_block(surface, x + 210, y + 14, ev.ahead, ev.ahead_pos)
        center_x = x + 210 + 300 + 5
        gap = text(f"{ev.interval:.3f}s", 30, WHITE)
        surface.blit(gap, (center_x - gap.get_width() // 2, y + 10))
        change = text(f"{ev.change:+.3f}", 18, change_color)
        triangle(surface, (center_x - change.get_width() // 2 - 12, y + 66), 7, up, change_color)
        surface.blit(change, (center_x - change.get_width() // 2 + 2, y + 56))
        self._rider_block(surface, x + 600, y + 14, ev.behind, ev.behind_pos)

    def _rider_block(self, surface, x, y, racer, position, dim=False):
        draw_row(surface, x, y + 17, 200, 34, position, racer.color, rider_name(racer.id), dim=dim, bg=ROW_B)


# --- TORRE DEI TEMPI (FINESTRA SEPARATA) ---
class StandingsTower:
    """Classifica live con tutte le posizioni, in una finestra a parte."""
    WIDTH = 420
    ROW_H = 34
    HEADER_H = 92
    FOOTER_H = 30
    FLASH = 1.6      # secondi di evidenziazione dopo un cambio di posizione
    ANIM_SPEED = 10  # velocità dell'animazione di scambio delle righe

    def __init__(self, count, position=None):
        self.count = count
        self.size = (self.WIDTH, self.HEADER_H + self.ROW_H * count + self.FOOTER_H)
        self.last_position = position
        self.window = None
        self.mode = "intervallo"         # oppure "distacco" (dal leader)
        self.row_y = {}
        self.flash = {}                  # id -> (istante, +1 guadagnata / -1 persa)
        self.last_time = None
        self.open()

    # gestione finestra
    def open(self):
        if self.window is not None:
            return
        pos = self.last_position if self.last_position is not None else pygame.WINDOWPOS_CENTERED
        self.window = pygame.Window("Classifica live", self.size, position=pos)
        self.row_y = {}

    def close(self):
        if self.window is not None:
            self.last_position = tuple(self.window.position)
            self.window.destroy()
            self.window = None

    def toggle(self):
        if self.window is None:
            self.open()
        else:
            self.close()

    def owns(self, window):
        return self.window is not None and window == self.window

    def toggle_mode(self):
        self.mode = "distacco" if self.mode == "intervallo" else "intervallo"

    def note_overtake(self, event, now):
        self.flash[event.ahead.id] = (now, 1)
        self.flash[event.behind.id] = (now, -1)

    # disegno
    def draw(self, timing, title, now):
        if self.window is None:
            return
        dt = 0.0 if self.last_time is None else min(0.1, now - self.last_time)
        self.last_time = now
        surf = self.window.get_surface()
        surf.fill(PANEL)

        # intestazione
        pygame.draw.rect(surf, MOTOGP_RED, (0, 0, self.WIDTH, 50))
        lap = text(timing.leader_lap_text(), 26, WHITE)
        surf.blit(lap, (14, 25 - lap.get_height() // 2))
        ttl = text(title.upper(), 13, WHITE, bold=False)
        surf.blit(ttl, (self.WIDTH - 12 - ttl.get_width(), 25 - ttl.get_height() // 2))
        mode = "INTERVALLO" if self.mode == "intervallo" else "DISTACCO DAL LEADER"
        surf.blit(text(mode, 15, GREY), (14, 62))
        leader = timing.order[0]
        if not leader.completed:
            sec = text(f"SETTORE {timing.current_sector(leader)}", 14, BLACK)
            box = pygame.Rect(0, 0, sec.get_width() + 14, sec.get_height() + 4)
            box.topright = (self.WIDTH - 12, 60)
            pygame.draw.rect(surf, SECTOR_YELLOW, box)
            surf.blit(sec, (box.x + 7, box.y + 2))

        # righe: ognuna scorre verso la sua nuova posizione (animazione di scambio)
        blend = 1 - math.exp(-dt * self.ANIM_SPEED)
        rows = []
        for idx, r in enumerate(timing.order):
            target = idx * self.ROW_H
            y = self.row_y.get(r.id, target)
            y += (target - y) * blend
            if abs(target - y) < 0.5:
                y = target
            self.row_y[r.id] = y
            rows.append((idx, r, y))
        # chi ha appena guadagnato posizioni viene disegnato sopra gli altri durante lo scambio
        rows.sort(key=lambda row: self._flash_dir(row[1].id, now) > 0)
        for idx, r, y in rows:
            self._draw_racer_row(surf, timing, idx, r, self.HEADER_H + int(y), now)

        footer = text("G intervallo/distacco  ·  T nascondi  ·  frecce velocità", 12, DIM, bold=False)
        surf.blit(footer, (14, self.size[1] - self.FOOTER_H + 8))
        self.window.flip()

    def _flash_dir(self, racer_id, now):
        info = self.flash.get(racer_id)
        if info is None or now - info[0] > self.FLASH:
            return 0
        return info[1]

    def _draw_racer_row(self, surf, timing, idx, r, y, now):
        dim = not r.alive
        bg = ROW_A if idx % 2 == 0 else ROW_B
        if r.completed:
            right, right_color = ("VINCITORE", GOLD) if idx == 0 else (format_gap(timing.gap_to_leader(r)), WHITE)
        elif not r.alive:
            right, right_color = None, RED   # "OUT" in rosso, disegnato sotto
        elif idx == 0:
            right, right_color = "LEADER", GOLD
        else:
            gap = timing.interval(r) if self.mode == "intervallo" else timing.gap_to_leader(r)
            right, right_color = format_gap(gap), WHITE
        draw_row(surf, 0, y, self.WIDTH, self.ROW_H - 2, idx + 1, r.color, rider_name(r.id),
                 right=right, right_color=right_color, dim=dim, bg=bg, highlight_pos=idx == 0)
        if not r.alive:
            out = text("OUT", 17, RED)
            surf.blit(out, (self.WIDTH - 12 - out.get_width(), y + (self.ROW_H - 2) // 2 - out.get_height() // 2))

        # evidenziazione dopo un sorpasso: verde chi guadagna, rosso chi perde
        direction = self._flash_dir(r.id, now)
        if direction:
            fade = 1 - (now - self.flash[r.id][0]) / self.FLASH
            overlay = pygame.Surface((self.WIDTH - 44, self.ROW_H - 2), pygame.SRCALPHA)
            overlay.fill((*(GREEN if direction > 0 else RED), int(70 * fade)))
            surf.blit(overlay, (44, y))
            triangle(surf, (178, y + 16), 6, direction > 0, GREEN if direction > 0 else RED)

        # usura gomme
        if r.alive and not r.completed:
            health = r.tyre_health / 100
            bar_color = (int(255 * (1 - health)), int(220 * health), 40)
            pygame.draw.rect(surf, (50, 50, 60), (196, y + 13, 50, 6))
            pygame.draw.rect(surf, bar_color, (196, y + 13, int(50 * health), 6))

        # tendenza dell'intervallo: verde = recupera su chi precede, arancione = perde terreno
        trend = timing.trend.get(r.id)
        if self.mode == "intervallo" and r.alive and not r.completed and idx > 0 and trend is not None:
            if abs(trend) >= TREND_SHOWN:
                closing = trend < 0
                triangle(surf, (272, y + 16), 6, not closing, GREEN if closing else ORANGE)
