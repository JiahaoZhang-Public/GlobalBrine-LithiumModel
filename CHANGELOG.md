# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
