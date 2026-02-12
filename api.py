from fastapi import FastAPI
from pydantic import BaseModel
from stable_baselines3 import PPO
import numpy as np

app = FastAPI(title="CubeSat GNC API")

# Load the trained brain once when the server starts
model = PPO.load("models/cubesat_3d_final")

class Telemetry(BaseModel):
    rel_pos: list[float]  # [dx, dy, dz]
    velocity: list[float] # [vx, vy, vz]
    fuel: float
    dist: float

@app.post("/predict_thrust")
async def predict_thrust(data: Telemetry):
    # 1. Format the data into the 8-dimension observation space
    obs = np.array(data.rel_pos + data.velocity + [data.fuel] + [data.dist], 
                   dtype=np.float32)
    
    # 2. Get the model's "decision"
    action, _states = model.predict(obs, deterministic=True)
    
    # 3. Return the thrust vector to the satellite
    return {
        "thrust_x": float(action[0]),
        "thrust_y": float(action[1]),
        "thrust_z": float(action[2]),
        "status": "active"
    }