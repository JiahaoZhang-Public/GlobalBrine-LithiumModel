def test_batch_predict_json(client):
    payload = {
        "rows": [
            {"TDS_gL": 331.0, "MLR": 1400.0, "Light_kW_m2": 1.5, "sample_id": "s1"},
            {"TDS_gL": 200.0, "MLR": 900.0, "Light_kW_m2": 0.8, "sample_id": "s2"},
        ],
        "model_version": "downstream_head_latest",
        "impute_strategy": "model",
    }
    r = client.post("/batch_predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "results" in body
    assert len(body["results"]) == 2

