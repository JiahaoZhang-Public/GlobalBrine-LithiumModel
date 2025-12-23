def test_predict_contract(client):
    payload = {
        "features": {"TDS_gL": 331.0, "MLR": 1400.0, "Light_kW_m2": 1.5, "sample_id": "s1"},
        "model_version": "downstream_head_latest",
        "impute_strategy": "model",
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "request_id" in body
    assert "input_features" in body
    assert "imputed_features" in body
    assert "predictions" in body
    assert "attributions" in body
    assert "meta" in body

