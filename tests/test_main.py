import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "machine_vision"}

@pytest.mark.asyncio
async def test_calibrate_camera():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/api/calibrate", json={"reference_points": [{"x":0, "y":0}]})
    assert response.status_code == 200
    assert response.json()["status"] == "calibrated"
    assert "matrix" in response.json()

@pytest.mark.asyncio
async def test_calculate_transformation():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/api/transform", json={"workpiece_id": "test_wp", "target_pos": {"x": 10}})
    assert response.status_code == 200
    assert "scale" in response.json()
