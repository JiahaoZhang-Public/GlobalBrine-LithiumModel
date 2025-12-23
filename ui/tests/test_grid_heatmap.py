def test_grid_heatmap_constant_light(client):
    payload = {
        "metric": "Selectivity",
        "bbox": {"lon_min": -68.0, "lat_min": -20.0, "lon_max": -67.0, "lat_max": -19.0},
        "step_deg": 1.0,
        "light_source": "constant",
        "light_value": 1.2,
        "scenario_features": {"TDS_gL": 300.0, "MLR": 1000.0},
        "recompute": True,
    }
    r = client.post("/grid_heatmap", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_points"] > 0
    p0 = body["points"][0]
    assert "lon" in p0 and "lat" in p0 and "value" in p0

