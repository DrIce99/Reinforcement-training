import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import pickle
import pygame
import math

def genera_e_salva_circuito(filename="pista_gara.png", n_punti=12):
    # 1. Generazione punti base
    angoli = np.sort(np.random.uniform(0, 2*np.pi, n_punti))
    raggi = 50 + np.random.uniform(-20, 20, n_punti)
    
    x = raggi * np.cos(angoli)
    y = raggi * np.sin(angoli)
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    
    # 2. Interpolazione Spline
    tck, u = splprep([x, y], s=0, per=True)
    u_fine = np.linspace(0, 1, 1000)
    smooth_x, smooth_y = splev(u_fine, tck)
    
    # 3. Creazione figura con dimensioni fisse (800x600)
    dpi = 100
    fig = plt.figure(figsize=(8, 6), dpi=dpi, facecolor='black')
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_facecolor('black')
    
    # Disegna solo la PISTA BIANCA (niente checkpoint colorati qui!)
    ax.plot(smooth_x, smooth_y, color='white', linewidth=40, zorder=1)
    
    # Calcola le posizioni dei Checkpoint e dello Spawn
    n_checkpoints = 20
    u_cp = np.linspace(0, 1, n_checkpoints, endpoint=False)
    cp_x, cp_y = splev(u_cp, tck)
    
    # Trasforma le coordinate di Matplotlib in coordinate Pixel (0-800, 0-600)
    # Calcoliamo il fattore di scala basandoci sui limiti degli assi
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    def to_pixel(x_val, y_val):
        px = ((x_val - x_min) / (x_max - x_min)) * 800
        py = (1.0 - (y_val - y_min) / (y_max - y_min)) * 600 # Inverti Y per Pygame
        return (int(px), int(py))

    # Salva Checkpoint e Spawn per il Training
    pixel_checkpoints = [to_pixel(cx, cy) for cx, cy in zip(cp_x, cp_y)]
    spawn_pixel = pixel_checkpoints[0]
    
    # Disegna lo spawn point VERDE (0, 255, 0) puro sull'immagine
    ax.scatter(cp_x[0], cp_y[0], color='#00FF00', s=100, marker='o', zorder=5)

    # Salvataggio immagine
    plt.savefig(filename, facecolor='black')
    plt.close(fig)
    
    p1 = pixel_checkpoints[0]
    p2 = pixel_checkpoints[1]
    # atan2 restituisce l'angolo in radianti, lo convertiamo in gradi per Pygame
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    base_angle = math.degrees(math.atan2(dy, dx))

    # Salva tutto in un dizionario per comodità
    track_data = {
        "checkpoints": pixel_checkpoints,
        "spawn_pos": pixel_checkpoints[0],
        "base_angle": base_angle
    }

    with open("track_config.pkl", "wb") as f:
        pickle.dump(track_data, f)

    print(f"Configurazione salvata! Angolo di partenza: {base_angle:.2f}°")

if __name__ == "__main__":
    genera_e_salva_circuito()
