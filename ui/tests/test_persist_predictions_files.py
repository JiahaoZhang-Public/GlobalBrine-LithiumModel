def test_persist_predictions_files(client, tmp_path):
    imputed_src = tmp_path / "imputed.csv"
    preds_src = tmp_path / "preds.csv"
    imputed_src.write_text("a,b\n1,2\n", encoding="utf-8")
    preds_src.write_text("a,b,Pred_Selectivity\n1,2,0.5\n", encoding="utf-8")

    imputed_out = tmp_path / "out_imputed.csv"
    preds_out = tmp_path / "out_preds.csv"

    with open(imputed_src, "rb") as f1, open(preds_src, "rb") as f2:
        r = client.post(
            "/persist_predictions_files",
            files={
                "imputed_file": ("imputed.csv", f1, "text/csv"),
                "predictions_file": ("preds.csv", f2, "text/csv"),
            },
            data={"imputed_path": str(imputed_out), "predictions_path": str(preds_out)},
        )

    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["paths"]["imputed_csv"] == str(imputed_out)
    assert payload["paths"]["predictions_csv"] == str(preds_out)
    assert imputed_out.read_text(encoding="utf-8") == "a,b\n1,2\n"
    assert preds_out.read_text(encoding="utf-8") == "a,b,Pred_Selectivity\n1,2,0.5\n"
