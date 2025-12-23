import os

import pandas as pd


def _write_predictions_csv(path: str, n: int = 20) -> None:
    rows = []
    for i in range(n):
        rows.append(
            {
                "Location": f"Loc{i%3}",
                "Latitude": -20.0 + 0.1 * i,
                "Longitude": -68.0 + 0.1 * i,
                "TDS_gL": 300.0 + i,
                "MLR": 1000.0 + 2 * i,
                "Light_kW_m2": 1.2 + 0.01 * i,
                "Pred_Selectivity": 0.5 + 0.01 * i,
                "Pred_Li_Crystallization_mg_m2_h": 10.0 + 0.1 * i,
                "Pred_Evap_kg_m2_h": 5.0 + 0.05 * i,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_map_points_schema_and_bbox(client, tmp_path):
    pred_path = tmp_path / "pred.csv"
    _write_predictions_csv(str(pred_path), n=25)
    os.environ["PREDICTIONS_CSV"] = str(pred_path)

    r = client.get("/map_points", params={"metric": "Selectivity", "limit": 5})
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["n_points"] == 5
    assert isinstance(payload["points"], list) and len(payload["points"]) == 5
    p0 = payload["points"][0]
    for key in ("sample_id", "lat", "lon", "raw", "prediction", "metric_value"):
        assert key in p0

    # BBox filter should reduce points.
    bbox = "-68.0,-20.0,-67.8,-19.8"  # includes first ~3 points
    r2 = client.get("/map_points", params={"metric": "Selectivity", "limit": 100, "bbox": bbox})
    assert r2.status_code == 200, r2.text
    payload2 = r2.json()
    assert payload2["n_points"] < 25


def test_aggregate_region_polygon(client, tmp_path):
    pred_path = tmp_path / "pred.csv"
    _write_predictions_csv(str(pred_path), n=30)
    os.environ["PREDICTIONS_CSV"] = str(pred_path)

    poly = [
        [-68.0, -20.0],
        [-67.0, -20.0],
        [-67.0, -19.0],
        [-68.0, -19.0],
        [-68.0, -20.0],
    ]
    r = client.post("/aggregate_region", json={"polygon": poly, "metric": "Selectivity"})
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["n_points"] > 0
    assert payload["aggregates"]["metric"] == "Selectivity"

