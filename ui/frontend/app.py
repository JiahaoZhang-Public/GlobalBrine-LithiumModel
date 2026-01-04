from __future__ import annotations

import os
import tempfile
from typing import Any

import gradio as gr
import pandas as pd
import requests

from ui.frontend.ui_helpers import (
    attributions_barplot,
    build_latent_scatter,
    build_plotly_geo_map,
    build_plotly_geo_raster_map,
    colorize_values,
    safe_get,
    sanitize_for_json,
    warnings_markdown,
    write_temp_csv,
)


def backend_url() -> str:
    return os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")


def _auth_headers(token: str | None) -> dict[str, str]:
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def api_models(token: str | None):
    r = requests.get(f"{backend_url()}/models", headers=_auth_headers(token), timeout=10)
    r.raise_for_status()
    return r.json()


def api_infer_light(lat: float, lon: float, token: str | None):
    r = requests.get(
        f"{backend_url()}/infer_light",
        params={"lat": lat, "lon": lon},
        headers=_auth_headers(token),
        timeout=20,
    )
    if r.status_code != 200:
        return None, r.text
    return r.json(), ""


def api_predict(payload: dict[str, Any], token: str | None):
    r = requests.post(
        f"{backend_url()}/predict",
        json=sanitize_for_json(payload),
        headers=_auth_headers(token),
        timeout=60,
    )
    if r.status_code != 200:
        return None, r.text
    return r.json(), ""


def api_impute(payload: dict[str, Any], token: str | None):
    r = requests.post(
        f"{backend_url()}/impute",
        json=sanitize_for_json(payload),
        headers=_auth_headers(token),
        timeout=60,
    )
    if r.status_code != 200:
        return None, r.text
    return r.json(), ""


def api_batch_predict_csv(
    file_obj: Any,
    *,
    model_version: str,
    impute_strategy: str,
    token: str | None,
):
    if file_obj is None:
        return None, "Upload a CSV first."
    with open(file_obj.name, "rb") as handle:
        files = {"file": (os.path.basename(file_obj.name), handle, "text/csv")}
        data = {"model_version": model_version, "impute_strategy": impute_strategy}
        r = requests.post(
            f"{backend_url()}/batch_predict",
            files=files,
            data=data,
            headers=_auth_headers(token),
            timeout=300,
        )
    if r.status_code != 200:
        return None, r.text
    return r.content, ""

def api_batch_predict_csv_path(
    path: str,
    *,
    model_version: str,
    impute_strategy: str,
    token: str | None,
):
    with open(path, "rb") as handle:
        files = {"file": (os.path.basename(path), handle, "text/csv")}
        data = {"model_version": model_version, "impute_strategy": impute_strategy}
        r = requests.post(
            f"{backend_url()}/batch_predict",
            files=files,
            data=data,
            headers=_auth_headers(token),
            timeout=300,
        )
    if r.status_code != 200:
        return None, r.text
    return r.content, ""


def api_persist_predictions_files(
    *,
    imputed_csv_path: str,
    predictions_csv_path: str,
    token: str | None,
    imputed_path: str | None = None,
    predictions_path: str | None = None,
):
    with open(imputed_csv_path, "rb") as f1, open(predictions_csv_path, "rb") as f2:
        files = {
            "imputed_file": (os.path.basename(imputed_csv_path), f1, "text/csv"),
            "predictions_file": (os.path.basename(predictions_csv_path), f2, "text/csv"),
        }
        data = {}
        if imputed_path:
            data["imputed_path"] = imputed_path
        if predictions_path:
            data["predictions_path"] = predictions_path
        r = requests.post(
            f"{backend_url()}/persist_predictions_files",
            files=files,
            data=data,
            headers=_auth_headers(token),
            timeout=60,
        )
    if r.status_code != 200:
        return None, r.text
    return r.json(), ""


def _load_default_brines() -> pd.DataFrame:
    path = os.environ.get("BRINES_CSV", "data/processed/brines.csv")
    return pd.read_csv(path)


def _highlight_missing_html(df: pd.DataFrame, max_rows: int = 50) -> str:
    view = df.head(max_rows).copy()
    styled = (
        view.style.map(lambda v: "background-color: #ffebee" if pd.isna(v) else "")
        .set_table_attributes('style="width: 100%; font-size: 12px;"')
    )
    return styled.to_html()


def build_demo() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown("# Nature-Water Gradio UI (Demo)")

        state_df = gr.State(value=None)

        with gr.Tab("Data Explorer"):
            gr.Markdown("Browse brine datasets and apply simple imputation.")
            with gr.Row():
                file_in = gr.File(label="Upload CSV (optional)", file_types=[".csv"])
                load_default = gr.Button("Load default data/processed/brines.csv")
                explorer_token = gr.Textbox(label="Auth token (optional)", type="password")
            with gr.Row():
                impute_strategy = gr.Dropdown(
                    choices=["none", "mean", "knn", "model"],
                    value="mean",
                    label="Impute strategy",
                )
                apply_impute = gr.Button("Apply Imputation (backend)")
            with gr.Row():
                pred_model = gr.Dropdown(
                    choices=["downstream_head_latest"],
                    value="downstream_head_latest",
                    label="Model (batch predict)",
                )
                run_predict = gr.Button("Predict (batch) + Save for Map/Latent")
                persist_outputs = gr.Checkbox(
                    value=True,
                    label="Persist to data/predictions (backend cache)",
                )
            missing_html = gr.HTML()
            preview = gr.DataFrame(interactive=False, label="Preview (first 50 rows)")
            download_out = gr.File(label="Download imputed CSV")
            pred_preview = gr.DataFrame(interactive=False, label="Predictions preview (first 50 rows)")
            pred_download = gr.File(label="Download predictions CSV")
            explorer_warn = gr.Markdown()

            def _load_df(file_obj):
                df = pd.read_csv(file_obj.name)
                return df, _highlight_missing_html(df), df.head(50)

            def _load_default():
                df = _load_default_brines()
                return df, _highlight_missing_html(df), df.head(50)

            def _apply_impute(df: pd.DataFrame, strategy: str, token: str | None):
                if df is None:
                    return None, None, None, "Load a dataset first.", None, None, None
                payload = {"rows": df.to_dict(orient="records"), "impute_strategy": strategy}
                res, err = api_impute(payload, token)
                if err:
                    return df, _highlight_missing_html(df), df.head(50), f"### Error\n{err}", None, None, None
                imputed = pd.DataFrame(res["imputed_rows"])
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                tmp.close()
                imputed.to_csv(tmp.name, index=False)
                return imputed, _highlight_missing_html(imputed), imputed.head(50), "", tmp.name, None, None

            file_in.change(_load_df, inputs=[file_in], outputs=[state_df, missing_html, preview])
            load_default.click(_load_default, outputs=[state_df, missing_html, preview])
            apply_impute.click(
                _apply_impute,
                inputs=[state_df, impute_strategy, explorer_token],
                outputs=[state_df, missing_html, preview, explorer_warn, download_out, pred_preview, pred_download],
            )

            def _predict_and_persist(
                df: pd.DataFrame,
                model_version: str,
                do_persist: bool,
                token: str | None,
            ):
                if df is None:
                    return None, None, "Load a dataset first."

                # Require that core numeric feature columns are fully imputed (no NaNs).
                feature_cols = [
                    "Li_gL",
                    "Mg_gL",
                    "Na_gL",
                    "K_gL",
                    "Ca_gL",
                    "SO4_gL",
                    "Cl_gL",
                    "MLR",
                    "TDS_gL",
                    "Light_kW_m2",
                ]
                missing_any = False
                for c in feature_cols:
                    if c in df.columns and df[c].isna().any():
                        missing_any = True
                        break
                if missing_any:
                    return None, None, "### Error\nPlease apply imputation first (NaNs still present in brine features)."

                tmp_imp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                tmp_imp.close()
                df.to_csv(tmp_imp.name, index=False)

                content, err = api_batch_predict_csv_path(
                    tmp_imp.name,
                    model_version=str(model_version),
                    impute_strategy="none",
                    token=token,
                )
                if err:
                    return None, None, f"### Error\n{err}"

                tmp_pred = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                tmp_pred.close()
                with open(tmp_pred.name, "wb") as handle:
                    handle.write(content)

                try:
                    pred_df = pd.read_csv(tmp_pred.name).head(50)
                except Exception:
                    pred_df = pd.DataFrame()

                if do_persist:
                    saved, serr = api_persist_predictions_files(
                        imputed_csv_path=tmp_imp.name,
                        predictions_csv_path=tmp_pred.name,
                        token=token,
                    )
                    if serr:
                        return pred_df, tmp_pred.name, f"### Warning\nSaved predictions file locally, but backend persist failed:\n\n{serr}"
                    paths = (saved or {}).get("paths", {})
                    return (
                        pred_df,
                        tmp_pred.name,
                        f"### Saved\n- imputed: `{paths.get('imputed_csv')}`\n- predictions: `{paths.get('predictions_csv')}`",
                    )

                return pred_df, tmp_pred.name, ""

            run_predict.click(
                _predict_and_persist,
                inputs=[state_df, pred_model, persist_outputs, explorer_token],
                outputs=[pred_preview, pred_download, explorer_warn],
            )

        with gr.Tab("Predict & Explain (Single)"):
            gr.Markdown("Enter features (partial allowed), infer Light from lat/lon, run prediction and see attributions.")
            with gr.Row():
                with gr.Column(scale=1):
                    token_in = gr.Textbox(label="Auth token (optional)", type="password")
                    model_version = gr.Dropdown(
                        choices=["downstream_head_latest"], value="downstream_head_latest", label="Model"
                    )
                    impute_single = gr.Dropdown(
                        choices=["none", "mean", "knn", "model"],
                        value="model",
                        label="Impute strategy",
                    )
                    sample_id = gr.Textbox(label="sample_id", value="sample-1")
                    lat = gr.Number(label="lat")
                    lon = gr.Number(label="lon")
                    light = gr.Number(label="Light_kW_m2")
                    infer_light_btn = gr.Button("Infer Light (GeoTIFF)")

                    # Brine chemistry features.
                    TDS = gr.Number(label="TDS_gL")
                    MLR = gr.Number(label="MLR")
                    Li = gr.Number(label="Li_gL")
                    Mg = gr.Number(label="Mg_gL")
                    Na = gr.Number(label="Na_gL")
                    K = gr.Number(label="K_gL")
                    Ca = gr.Number(label="Ca_gL")
                    SO4 = gr.Number(label="SO4_gL")
                    Cl = gr.Number(label="Cl_gL")
                    predict_btn = gr.Button("Predict")
                with gr.Column(scale=1):
                    warn_md = gr.Markdown()
                    out_json = gr.JSON(label="Response")
                    target_pick = gr.Dropdown(
                        choices=[
                            "Selectivity",
                            "Li_Crystallization_mg_m2_h",
                            "Evap_kg_m2_h",
                        ],
                        value="Selectivity",
                        label="Attribution target",
                    )
                    attr_plot = gr.Plot(label="Attributions (top)")

            def _infer(lat, lon, token):
                if lat is None or lon is None:
                    return None, "Provide lat and lon."
                res, err = api_infer_light(float(lat), float(lon), token)
                if err:
                    return None, f"### Error\n{err}"
                return float(res["Light_kW_m2"]), ""

            def _predict(
                token,
                model_version,
                impute_strategy,
                sample_id,
                lat,
                lon,
                light,
                TDS,
                MLR,
                Li,
                Mg,
                Na,
                K,
                Ca,
                SO4,
                Cl,
                target,
            ):
                features = {
                    "sample_id": sample_id,
                    "lat": lat,
                    "lon": lon,
                    "Light_kW_m2": light,
                    "TDS_gL": TDS,
                    "MLR": MLR,
                    "Li_gL": Li,
                    "Mg_gL": Mg,
                    "Na_gL": Na,
                    "K_gL": K,
                    "Ca_gL": Ca,
                    "SO4_gL": SO4,
                    "Cl_gL": Cl,
                }
                payload = {
                    "features": features,
                    "model_version": model_version,
                    "impute_strategy": impute_strategy,
                }
                res, err = api_predict(payload, token)
                if err:
                    return f"### Error\n{err}", None, None
                warnings = safe_get(res, "meta", "warnings", default=[])
                warn = warnings_markdown(warnings)
                attrs = safe_get(res, "attributions", target, default=None)
                fig = attributions_barplot(attrs, title=f"Attributions for {target}", top_n=10)
                return warn, res, fig

            infer_light_btn.click(_infer, inputs=[lat, lon, token_in], outputs=[light, warn_md])
            predict_btn.click(
                _predict,
                inputs=[
                    token_in,
                    model_version,
                    impute_single,
                    sample_id,
                    lat,
                    lon,
                    light,
                    TDS,
                    MLR,
                    Li,
                    Mg,
                    Na,
                    K,
                    Ca,
                    SO4,
                    Cl,
                    target_pick,
                ],
                outputs=[warn_md, out_json, attr_plot],
            )

        with gr.Tab("Batch Upload"):
            gr.Markdown("Upload a CSV and download an annotated CSV with predictions.")
            batch_token = gr.Textbox(label="Auth token (optional)", type="password")
            batch_model = gr.Dropdown(
                choices=["downstream_head_latest"], value="downstream_head_latest", label="Model"
            )
            batch_strategy = gr.Dropdown(
                choices=["none", "mean", "knn", "model"],
                value="model",
                label="Impute strategy",
            )
            batch_file = gr.File(label="CSV file", file_types=[".csv"])
            batch_btn = gr.Button("Run batch predict")
            batch_err = gr.Markdown()
            batch_out = gr.File(label="Annotated CSV")

            def _run_batch(file_obj, model_version, strategy, token):
                content, err = api_batch_predict_csv(
                    file_obj, model_version=model_version, impute_strategy=strategy, token=token
                )
                if err:
                    return f"### Error\n{err}", None
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                with open(tmp.name, "wb") as handle:
                    handle.write(content)
                return "", tmp.name

            batch_btn.click(
                _run_batch,
                inputs=[batch_file, batch_model, batch_strategy, batch_token],
                outputs=[batch_err, batch_out],
            )

        def api_map_points(
            *,
            metric: str,
            limit: int,
            bbox: str | None,
            include_attributions: bool,
            token: str | None,
        ):
            params = {"metric": metric, "limit": int(limit), "include_attributions": bool(include_attributions)}
            if bbox:
                params["bbox"] = bbox
            r = requests.get(
                f"{backend_url()}/map_points",
                params=params,
                headers=_auth_headers(token),
                timeout=60,
            )
            if r.status_code != 200:
                return None, r.text
            return r.json(), ""

        def api_grid_heatmap(
            payload: dict[str, Any],
            token: str | None,
        ):
            r = requests.post(
                f"{backend_url()}/grid_heatmap",
                json=sanitize_for_json(payload),
                headers=_auth_headers(token),
                timeout=300,
            )
            if r.status_code != 200:
                return None, r.text
            return r.json(), ""

        def api_interpolate_heatmap(
            payload: dict[str, Any],
            token: str | None,
        ):
            r = requests.post(
                f"{backend_url()}/interpolate_heatmap",
                json=sanitize_for_json(payload),
                headers=_auth_headers(token),
                timeout=300,
            )
            if r.status_code != 200:
                return None, r.text
            return r.json(), ""

        def api_aggregate_region(payload: dict[str, Any], token: str | None):
            r = requests.post(
                f"{backend_url()}/aggregate_region",
                json=sanitize_for_json(payload),
                headers=_auth_headers(token),
                timeout=60,
            )
            if r.status_code != 200:
                return None, r.text
            return r.json(), ""

        def api_latent_embeddings(
            *,
            method: str,
            n_neighbors: int,
            perplexity: float,
            model_version: str,
            recompute: bool,
            token: str | None,
        ):
            r = requests.get(
                f"{backend_url()}/latent_embeddings",
                params={
                    "method": method,
                    "n_neighbors": int(n_neighbors),
                    "perplexity": float(perplexity),
                    "model_version": model_version,
                    "recompute": bool(recompute),
                },
                headers=_auth_headers(token),
                timeout=300,
            )
            if r.status_code != 200:
                return None, r.text
            return r.json(), ""

        with gr.Tab("Map"):
            gr.Markdown("Map: sample points (known sites) + optional global heatmap overlay.")
            map_state = gr.State(value=None)  # stores `points` list
            with gr.Row():
                with gr.Column(scale=1):
                    map_token = gr.Textbox(label="Auth token (optional)", type="password")
                    map_metric = gr.Dropdown(
                        choices=[
                            "missing_rate",
                            "Selectivity",
                            "Li_Crystallization_mg_m2_h",
                            "Evap_kg_m2_h",
                            "TDS_gL",
                            "MLR",
                        ],
                        value="missing_rate",
                        label="Metric",
                    )
                    map_limit = gr.Number(label="Max points", value=5000, precision=0)
                    color_scale = gr.Dropdown(
                        choices=["linear", "quantile", "log"],
                        value="linear",
                        label="Color scale",
                    )
                    bbox_filter = gr.Textbox(
                        label="BBox filter (min_lon,min_lat,max_lon,max_lat) (optional)",
                        value="",
                    )
                    gr.Markdown("### Global heatmap (interpolated from samples)")
                    heat_style = gr.Dropdown(
                        choices=[
                            "none",
                            "point_cloud (fast)",
                            "raster (paper-style)",
                        ],
                        value="none",
                        label="Heatmap style",
                    )
                    heat_step = gr.Slider(minimum=0.5, maximum=5.0, step=0.5, value=2.0, label="Grid step (deg)")
                    heat_method = gr.Dropdown(choices=["idw", "knn"], value="idw", label="Interpolation method")
                    heat_k = gr.Slider(minimum=1, maximum=32, step=1, value=8, label="k (nearest neighbors)")
                    heat_power = gr.Slider(minimum=0.5, maximum=4.0, step=0.5, value=2.0, label="IDW power")
                    heat_max_km = gr.Number(label="Max extrapolation distance (km; blank = no mask)", value=1500.0)
                    heat_opacity = gr.Slider(minimum=0.1, maximum=0.9, step=0.1, value=0.6, label="Heatmap opacity")
                    refresh_map = gr.Button("Refresh / Recompute")
                    map_err = gr.Markdown()
                with gr.Column(scale=3):
                    map_plot = gr.Plot()

            map_points_table = gr.DataFrame(interactive=False, label="Points preview (first 200)")

            gr.Markdown("### Region aggregation (bbox)")
            with gr.Row():
                lat_min = gr.Number(label="lat_min", value=-90)
                lat_max = gr.Number(label="lat_max", value=90)
                lon_min = gr.Number(label="lon_min", value=-180)
                lon_max = gr.Number(label="lon_max", value=180)
            aggregate_btn = gr.Button("Aggregate selection")
            agg_out = gr.JSON(label="Aggregates")
            agg_samples_csv = gr.File(label="Download selected samples (CSV)")

            def _refresh_map(
                metric: str,
                limit: float,
                scale: str,
                bbox_text: str,
                heat_style_v: str,
                heat_step_v: float,
                heat_method_v: str,
                heat_k_v: float,
                heat_power_v: float,
                heat_max_km_v,
                heat_opacity_v: float,
                token: str | None,
            ):
                bbox_text = (bbox_text or "").strip()
                resp, err = api_map_points(
                    metric=metric,
                    limit=int(limit),
                    bbox=bbox_text or None,
                    include_attributions=False,
                    token=token,
                )
                if err:
                    return None, None, None, f"### Error\n{err}"
                points = resp.get("points", [])
                colors = colorize_values([p.get("metric_value") for p in points], scale=scale)
                for p, c in zip(points, colors, strict=False):
                    p["color"] = c
                heat_points = None
                heat_grid = None
                heat_style_v = (heat_style_v or "none").lower().strip()
                if heat_style_v != "none":
                    bbox_obj = None
                    if bbox_text:
                        try:
                            parts = [float(x.strip()) for x in bbox_text.split(",")]
                            if len(parts) == 4:
                                bbox_obj = {
                                    "lon_min": parts[0],
                                    "lat_min": parts[1],
                                    "lon_max": parts[2],
                                    "lat_max": parts[3],
                                }
                        except Exception:
                            bbox_obj = None

                    payload = {
                        "metric": metric,
                        "step_deg": float(heat_step_v),
                        "method": str(heat_method_v),
                        "k": int(heat_k_v),
                        "power": float(heat_power_v),
                        "max_distance_km": None
                        if heat_max_km_v in (None, "")
                        else float(heat_max_km_v),
                        "return_grid": heat_style_v.startswith("raster"),
                    }
                    if bbox_obj is not None:
                        payload["bbox"] = bbox_obj
                    grid, gerr = api_interpolate_heatmap(payload, token)
                    if gerr:
                        preview = pd.DataFrame(points).head(200)
                        fig = build_plotly_geo_map(points, metric=metric)
                        return points, fig, preview, f"### Error\n{gerr}"
                    heat_points = grid.get("points", [])
                    heat_grid = grid.get("grid", None)

                preview = pd.DataFrame(points).head(200)
                if heat_style_v.startswith("raster") and heat_grid is not None:
                    fig = build_plotly_geo_raster_map(
                        points,
                        metric=metric,
                        grid=heat_grid,
                        heat_opacity=float(heat_opacity_v),
                    )
                else:
                    fig = build_plotly_geo_map(
                        points,
                        metric=metric,
                        heatmap_points=heat_points,
                        heat_opacity=float(heat_opacity_v),
                    )
                return points, fig, preview, ""

            refresh_map.click(
                _refresh_map,
                inputs=[
                    map_metric,
                    map_limit,
                    color_scale,
                    bbox_filter,
                    heat_style,
                    heat_step,
                    heat_method,
                    heat_k,
                    heat_power,
                    heat_max_km,
                    heat_opacity,
                    map_token,
                ],
                outputs=[map_state, map_plot, map_points_table, map_err],
            )

            def _aggregate_bbox(
                points: list[dict[str, Any]] | None,
                metric: str,
                lat_min_v,
                lat_max_v,
                lon_min_v,
                lon_max_v,
                token: str | None,
            ):
                if points is None:
                    return {"error": "Load map points first."}, None
                poly = [
                    [float(lon_min_v), float(lat_min_v)],
                    [float(lon_max_v), float(lat_min_v)],
                    [float(lon_max_v), float(lat_max_v)],
                    [float(lon_min_v), float(lat_max_v)],
                    [float(lon_min_v), float(lat_min_v)],
                ]
                resp, err = api_aggregate_region({"polygon": poly, "metric": metric}, token)
                if err:
                    return {"error": err}, None
                samples = resp.get("samples", [])
                csv_path = write_temp_csv(samples)
                return resp, csv_path

            aggregate_btn.click(
                _aggregate_bbox,
                inputs=[map_state, map_metric, lat_min, lat_max, lon_min, lon_max, map_token],
                outputs=[agg_out, agg_samples_csv],
            )

        with gr.Tab("Latent Space"):
            gr.Markdown("2D projection of MAE latents (UMAP/t-SNE/PCA), with coloring and neighbor inspection.")
            latent_state = gr.State(value=None)
            with gr.Row():
                with gr.Column(scale=1):
                    latent_token = gr.Textbox(label="Auth token (optional)", type="password")
                    latent_method = gr.Dropdown(choices=["umap", "tsne", "pca"], value="pca", label="Method")
                    latent_n_neighbors = gr.Number(label="n_neighbors (UMAP)", value=15, precision=0)
                    latent_perplexity = gr.Number(label="perplexity (t-SNE)", value=30.0)
                    latent_color_by = gr.Dropdown(
                        choices=["none", "Brine", "Type_of_water", "Location"],
                        value="Brine",
                        label="Color by",
                    )
                    latent_point_size = gr.Slider(minimum=2, maximum=12, step=1, value=6, label="Point size")
                    latent_recompute = gr.Checkbox(label="Force recompute (ignore cache)", value=False)
                    latent_model = gr.Dropdown(
                        choices=["downstream_head_latest"], value="downstream_head_latest", label="Model version"
                    )
                    latent_btn = gr.Button("Compute / Refresh")
                    latent_err = gr.Markdown()
                with gr.Column(scale=3):
                    latent_plot = gr.Plot()

            latent_table = gr.DataFrame(interactive=False, label="Metadata preview (first 200)")

            with gr.Row():
                inspect_sample_id = gr.Textbox(label="Inspect sample_id")
                inspect_btn = gr.Button("Inspect")
            inspect_json = gr.JSON(label="Sample metadata")
            neighbors_table = gr.DataFrame(interactive=False, label="Nearest neighbors (top-10)")

            gr.Markdown("### Brush selection (x/y bounds)")
            with gr.Row():
                x_min = gr.Number(label="x_min")
                x_max = gr.Number(label="x_max")
                y_min = gr.Number(label="y_min")
                y_max = gr.Number(label="y_max")
            brush_btn = gr.Button("Select points in bounds")
            brush_table = gr.DataFrame(interactive=False, label="Selected points")
            brush_csv = gr.File(label="Download selected points (CSV)")

            def _compute_latents(
                method: str,
                n_neighbors_v,
                perplexity_v,
                color_by: str,
                point_size: int,
                recompute: bool,
                model_version: str,
                token: str | None,
            ):
                resp, err = api_latent_embeddings(
                    method=method,
                    n_neighbors=int(n_neighbors_v),
                    perplexity=float(perplexity_v),
                    model_version=model_version,
                    recompute=bool(recompute),
                    token=token,
                )
                if err:
                    return None, None, None, f"### Error\n{err}"
                fig = build_latent_scatter(
                    resp.get("embeddings", []),
                    resp.get("metadata", []),
                    color_by=color_by,
                    point_size=int(point_size),
                )
                preview = pd.DataFrame(resp.get("metadata", [])).head(200)
                return resp, fig, preview, ""

            latent_btn.click(
                _compute_latents,
                inputs=[
                    latent_method,
                    latent_n_neighbors,
                    latent_perplexity,
                    latent_color_by,
                    latent_point_size,
                    latent_recompute,
                    latent_model,
                    latent_token,
                ],
                outputs=[latent_state, latent_plot, latent_table, latent_err],
            )

            def _inspect(state: dict[str, Any] | None, sample_id: str):
                if not state:
                    return None, None
                sid = (sample_id or "").strip()
                if not sid:
                    return None, None
                md = state.get("metadata", [])
                for row in md:
                    if row.get("sample_id") == sid:
                        neigh = [{"neighbor": x} for x in (row.get("neighbors") or [])]
                        return row, pd.DataFrame(neigh)
                return {"error": f"sample_id not found: {sid}"}, None

            inspect_btn.click(_inspect, inputs=[latent_state, inspect_sample_id], outputs=[inspect_json, neighbors_table])

            def _brush_select(state: dict[str, Any] | None, x0, x1, y0, y1):
                if not state:
                    return None, None
                emb = state.get("embeddings", [])
                md = state.get("metadata", [])
                if not emb or not md:
                    return None, None
                x0 = float(x0) if x0 is not None else -float("inf")
                x1 = float(x1) if x1 is not None else float("inf")
                y0 = float(y0) if y0 is not None else -float("inf")
                y1 = float(y1) if y1 is not None else float("inf")
                rows = []
                for (x, y), meta in zip(emb, md, strict=False):
                    if x0 <= float(x) <= x1 and y0 <= float(y) <= y1:
                        rows.append({"sample_id": meta.get("sample_id"), "x": float(x), "y": float(y), **meta})
                df = pd.DataFrame(rows)
                return df, (write_temp_csv(df) if len(df) else None)

            brush_btn.click(
                _brush_select,
                inputs=[latent_state, x_min, x_max, y_min, y_max],
                outputs=[brush_table, brush_csv],
            )

        with gr.Tab("Settings / Model"):
            gr.Markdown("Backend connection and available model versions.")
            settings_token = gr.Textbox(label="Auth token (optional)", type="password")
            refresh_btn = gr.Button("Refresh models")
            models_json = gr.JSON()

            def _refresh(token):
                try:
                    return api_models(token)
                except Exception as exc:
                    return {"error": str(exc), "backend_url": backend_url()}

            refresh_btn.click(_refresh, inputs=[settings_token], outputs=[models_json])

    return demo


if __name__ == "__main__":
    build_demo().launch(server_name="0.0.0.0", server_port=7860)
