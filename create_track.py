import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

def genera_e_salva_circuito(filename="circuito_generato.png", n_punti=20):
    # 1. Generazione punti polari con jitter (garantisce no intersezioni)
    angoli = np.linspace(0, 2*np.pi, n_punti, endpoint=False)
    angoli += np.random.uniform(-0.05, 0.05, n_punti) # Leggera asimmetria
    angoli = np.sort(angoli)
    
    raggio_medio = 50
    # Jitter radiale per creare curve e rientranze complesse
    raggi = raggio_medio + np.random.uniform(-25, 25, n_punti)
    
    x = raggi * np.cos(angoli)
    y = raggi * np.sin(angoli)
    
    # Chiusura del set di punti
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    
    # 2. Interpolazione Spline Cubica Periodica
    tck, u = splprep([x, y], s=0, per=True)
    u_fine = np.linspace(0, 1, 1000)
    smooth_x, smooth_y = splev(u_fine, tck)
    
    # 3. Checkpoints (15 punti distribuiti sulla spline)
    n_checkpoints = 15
    u_cp = np.linspace(0, 1, n_checkpoints, endpoint=False)
    cp_x, cp_y = splev(u_cp, tck)
    
    # 4. Rendering e salvataggio PNG
    fig = plt.figure(figsize=(10, 10), facecolor='black')
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect='equal')
    ax.set_facecolor('black')
    
    # Disegna tracciato bianco
    ax.plot(smooth_x, smooth_y, color='white', linewidth=35, zorder=1)
    
    # Disegna checkpoints azzurri
    ax.scatter(cp_x[1:], cp_y[1:], color='cyan', s=60, edgecolors='white', zorder=3)
    
    # Disegna spawn point verde (quadrato)
    ax.scatter(cp_x[0], cp_y[0], color='#00FF00', s=250, marker='s', edgecolors='white', zorder=5)
    
    plt.xticks([]); plt.yticks([])
    plt.savefig(filename, facecolor='black', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Circuito salvato con successo in: {filename}")

# Generazione
genera_e_salva_circuito("pista_gara.png")
