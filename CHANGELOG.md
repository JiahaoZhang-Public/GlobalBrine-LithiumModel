# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-04-06

### Added

- FiLM (Feature-wise Linear Modulation) conditioning layer (`src/models/film.py`): Light_kW_m2 now modulates the latent representation via learned affine transforms (z_film = gamma * z + beta) instead of being concatenated or fed into the MAE encoder.
- Identity initialization for FiLM layer ensures stable training from the start (gamma=1, beta=0 at init).
- `mae_impute_brine_features()` in inference pipeline for MAE-based chemistry imputation with TDS physical clamp.

### Changed

- **Breaking**: MAE encoder now receives 9 brine chemistry features only (Light_kW_m2 removed from `BRINE_FEATURE_COLUMNS`). Light enters via FiLM conditioning in the regression head.
- Regression head upgraded from `RegressionHead` to `FiLMRegressionHead`; checkpoint format includes `use_film` flag for backward compatibility with legacy models.
- Experimental scaler: Light_kW_m2 statistics always computed from experimental data (not brine data) to fix distribution mismatch between lab irradiance and geographic GHI.
- Updated model checkpoints (`mae_pretrained.pth`, `downstream_head.pth`) for v0.3.0 architecture.

### Removed

- Dead code: `sample_feature_mask()` from `mae.py`, empty `src/visualization/visualize.py`, obsolete `tox.ini`.
- `concat_light` logic from finetune and inference pipelines (replaced by FiLM).

## [0.2.2] - 2026-02-02

### Added

- Bilingual UI (EN/中文) with persistent language toggle across all key pages.
- Dataset cards with download-only flow and on-demand preview (first 10 rows).

### Changed

- Default `GLB_MODEL_VERSION` bumped to `0.2.2` across backend config, Dockerfile, and web docs.
- Landing, map, prediction, batch, model, and team pages polished for consistent copy and units.

### Fixed

- Dataset preview containers now stay within their cards (min-w-0 + overflow handling).
- Map legend, filters, and hover copy clarified; prediction helper text refined.

## [0.2.1] - 2026-02-02

### Added

- Dataset preview on the landing page with inline stats and one-click CSV download.
- Team page crediting MBZUAI (ML Department) and HIT Zongyao Zhou Group.
- Public code repo link in header.
- Homepage screenshot (`web/app/public/homepage.png`) and pipeline overview SVG in README.

### Changed

- Refreshed typography/gradients and streamlined navigation for scientist-focused UI.
- Map explorer filter UX upgraded with range sliders, counts, and clearer units/legend.
- Single prediction form now has example autofill; batch jobs include example rows and quick recipe.
- Default `GLB_MODEL_VERSION` bumped to `0.2.1` across Dockerfile, config, and DO app spec.

## [0.2.0] - 2026-02-01

### Added

- DigitalOcean App Platform deployment for the API and static web UI, with same-origin `/api` routing preserved.
- `/v1/*` alias endpoints on the backend to support DO path prefix stripping without redirects.

### Changed

- Default model version bumped to `0.2.0` across backend, Docker image, DO spec, and documentation.
- Frontend API base remains relative for clean static hosting behind DO routing.

## [0.1.0] - 2026-01-30

### Added

- Initial tagged release of the GlobalBrine-LithiumModel codebase.
