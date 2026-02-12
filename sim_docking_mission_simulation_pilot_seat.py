from vpython import *
import numpy as np
import requests
import time

# --- CONFIGURATION ---
URL = "http://127.0.0.1:8000/predict_thrust"
DT = 0.1             
MAX_THRUST = 0.15
GRAVITY = 0.02
DRAG = 0.01
TARGET_POS = vector(10, 0, 0)

# --- 3D SCENE SETUP ---
scene = canvas(title='Pilot Seat: Final Approach', width=1000, height=700)
scene.background = color.black
scene.forward = vector(1, 0, 0) 
scene.autoscale = False # Prevents the camera from jumping around

# Starfield
for i in range(150):
    sphere(pos=vector.random()*60 - vector(30,30,30), radius=0.04, color=color.white, emissive=True)

# Target Station
station = box(pos=TARGET_POS, size=vector(2, 2, 2), color=vector(0.7, 0.7, 0.7))
docking_ring = ring(pos=TARGET_POS - vector(1,0,0), axis=vector(1,0,0), 
                    radius=0.4, thickness=0.05, color=color.yellow)

# High-Intensity Guidance Light
local_light(pos=TARGET_POS - vector(0.9,0,0), color=color.green)

# Agent CubeSat (Camera Anchor)
satellite = box(pos=vector(0,0,0), size=vector(0.4, 0.4, 0.4), opacity=0) 
scene.camera.follow(satellite)

# HUD Elements
ch_v = label(pos=satellite.pos, text='|', opacity=0, box=False, height=30, color=color.green)
ch_h = label(pos=satellite.pos, text='—', opacity=0, box=False, height=30, color=color.green)
hud_dist = label(pos=satellite.pos, text='', xoffset=0, yoffset=180, height=14, font='monospace', box=False)
hud_vel = label(pos=satellite.pos, text='', xoffset=0, yoffset=-180, height=14, font='monospace', box=False)

# Initial State
pos = np.array([0.0, 0.0, 0.0])
vel = np.array([0.0, 0.0, 0.0])
fuel = 100.0
mission_active = True

print("👨‍🚀 System Check: Green. HUD: Active. Initiating Docking...")

while mission_active:
    rate(30) # Lock to 30 Steps Per Second

    # 1. Telemetry
    rel_pos = pos - np.array([10.0, 0.0, 0.0])
    dist_val = np.linalg.norm(rel_pos)
    
    telemetry = {
        "rel_pos": rel_pos.tolist(),
        "velocity": vel.tolist(),
        "fuel": float(fuel),
        "dist": float(dist_val)
    }

    # 2. API Call
    try:
        resp = requests.post(URL, json=telemetry, timeout=1).json()
        thrust = np.array([resp['thrust_x'], resp['thrust_y'], resp['thrust_z']])
    except Exception as e:
        print(f"Connection lost: {e}")
        break

    # 3. Physics
    r_dist = np.linalg.norm(pos)
    gravity_vec = -GRAVITY * pos / r_dist if r_dist > 0.1 else np.zeros(3)
    accel = (thrust * MAX_THRUST) + gravity_vec + (-DRAG * vel)
    vel += accel * DT
    pos += vel * DT

    # 4. HUD & Camera Sync
    satellite.pos = vector(pos[0], pos[1], pos[2])
    
    # Offset HUD slightly in front of the "camera"
    hud_anchor = satellite.pos + vector(1, 0, 0)
    ch_v.pos = hud_anchor
    ch_h.pos = hud_anchor
    hud_dist.pos = hud_anchor
    hud_vel.pos = hud_anchor
    
    hud_dist.text = f"RANGE: {dist_val:.2f} M"
    hud_vel.text = f"CLOSURE: {np.linalg.norm(vel):.2f} M/S"

    # 5. Success/Fail Conditions
    if dist_val < 0.8:
        hud_dist.text = "CONTACT SUCCESSFUL - DOCKED"
        hud_dist.color = color.cyan
        mission_active = False # Stop physics, but window stays open
    
    if np.any(np.abs(pos) > 50.0):
        hud_dist.text = "MISSION ABORTED: OUT OF BOUNDS"
        hud_dist.color = color.red
        mission_active = False

# Keep the window alive
print("🏁 Mission ended. Close the 3D window manually.")
while True:
    rate(1) # Idle loop to keep the graphics window from closing