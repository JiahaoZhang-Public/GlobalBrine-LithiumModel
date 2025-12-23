import os

import pandas as pd


def _write_brines_csv(path: str, n: int = 20) -> None:
    rows = []
    for i in range(n):
        rows.append(
            {
                "Location": f"Loc{i%3}",
                "Latitude": -20.0 + 0.1 * i,
                "Longitude": -68.0 + 0.1 * i,
                "Li_gL": 0.1 + 0.01 * i,
                "Mg_gL": 1.0 + 0.02 * i,
                "Na_gL": 10.0 + 0.1 * i,
                "K_gL": 0.5 + 0.01 * i,
                "Ca_gL": 0.3 + 0.01 * i,
                "SO4_gL": 2.0 + 0.03 * i,
                "Cl_gL": 15.0 + 0.2 * i,
                "MLR": 1000.0 + 2 * i,
                "TDS_gL": 300.0 + i,
                "Light_kW_m2": 1.2 + 0.01 * i,
                "Country": "X",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_predictions_csv(path: str, n: int = 20) -> None:
    rows = []
    for i in range(n):
        rows.append(
            {
                "Location": f"Loc{i%3}",
                "Latitude": -20.0 + 0.1 * i,
                "Longitude": -68.0 + 0.1 * i,
                "Pred_Selectivity": 0.5 + 0.01 * i,
                "Pred_Li_Crystallization_mg_m2_h": 10.0 + 0.1 * i,
                "Pred_Evap_kg_m2_h": 5.0 + 0.05 * i,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_latent_embeddings_pca_with_cache(client, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    brines_path = data_dir / "brines.csv"
    preds_path = tmp_path / "pred.csv"
    cache_dir = tmp_path / "cache"

    _write_brines_csv(str(brines_path), n=15)
    _write_predictions_csv(str(preds_path), n=15)

    os.environ["DATA_DIR"] = str(data_dir)
    os.environ["PREDICTIONS_CSV"] = str(preds_path)
    os.environ["UI_CACHE_DIR"] = str(cache_dir)

    r = client.get("/latent_embeddings", params={"method": "pca", "recompute": True})
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["n_samples"] == 15
    assert len(payload["sample_ids"]) == 15
    assert len(payload["embeddings"]) == 15
    assert len(payload["metadata"]) == 15
    assert len(payload["embeddings"][0]) == 2
    assert "neighbors" in payload["metadata"][0]
    assert len(payload["metadata"][0]["neighbors"]) == 10

    # Second call should hit cache (recompute=False default).
    r2 = client.get("/latent_embeddings", params={"method": "pca"})
    assert r2.status_code == 200, r2.text
    payload2 = r2.json()
    assert payload2["n_samples"] == 15

