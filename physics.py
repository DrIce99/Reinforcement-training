# physics.py
import numpy as np

def compute_step(angle, velocity, steer_raw, speed_raw, confidence,
                 precision=1.0, aggressiveness=1.0, risk_taking=1.0, tyre_health=100.0):
    safe_zone = 0.35
    if confidence < safe_zone:
        base_speed = (1.5 + (speed_raw + 1) * 0.5) * aggressiveness
        steer = steer_raw * 7 * precision
    else:
        base_speed = (2 + (speed_raw + 1) * 5) * aggressiveness
        steer = steer_raw * (5 + base_speed * 0.3) * precision

    turn_intensity = abs(steer_raw)
    turn_penalty = 1.0 - (turn_intensity * 0.7 * risk_taking)
    straight_bonus = 1.0 + ((1.0 - turn_intensity) * 0.5)
    
    speed = base_speed * turn_penalty * straight_bonus
    speed = speed * (1.0 - (speed / 10.0) * 0.5)
    speed = max(0.5, min(speed, 8))

    # --- EFFETTO GOMME USCITE SULLA ADERENZA ---
    # grip_factor va da 1.0 (gomme perfette) a 0.5 (gomme distrutte)
    grip_factor = 0.5 + 0.5 * (tyre_health / 100.0)
    
    # Riduce l'efficacia dello sterzo: la macchina "va dritta" (understeer)
    steer *= grip_factor

    grip = max(0.2, 1.0 - speed * 1.1)
    new_angle = angle + steer * grip

    # --- EFFETTO GOMME USCITE SULLA FRENATA ---
    acceleration = 0.01
    # Le gomme morbide fermano bene (0.08), le gomme lisce fermano male (0.04)
    brake_force = 0.08 * grip_factor 
    
    if speed > velocity:
        velocity += (speed - velocity) * acceleration
    else:
        # Frenata meno efficace se le gomme sono consumate
        velocity += (speed - velocity) * brake_force

    return new_angle, velocity

def check_line_intersection(p1, p2, q1, q2):
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))