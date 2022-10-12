import asyncio

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
import numpy as np
from prometheus_client import generate_latest

from monitoring_with_prometheus.metrics import REQUEST_COUNT


app = FastAPI()


@app.get("/")
def root():
    """Root of the prometheus monitoring app"""
    return {"message": "Prometheus monitoring app root"}


@app.get('/metrics', response_class=PlainTextResponse)
def metrics():
    return generate_latest()


@app.get("/predict/random")
async def get_random_prediction(time_delay_scale: float = 0.1, error_rate: float = 0.01) -> float:
    """Get a random uniform prediction with an exponential time delay"""
    REQUEST_COUNT.inc()

    time_delay = np.random.exponential(time_delay_scale)
    await asyncio.sleep(time_delay)

    if np.random.uniform() < error_rate:
        raise ValueError("Could not make a prediction")

    prediction = np.random.uniform()

    return prediction
