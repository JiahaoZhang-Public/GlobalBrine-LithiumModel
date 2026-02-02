# Release Notes — v0.2.2 (February 2, 2026)

## Highlights
- Added bilingual UI (EN/中文) with persistent language toggle; all key pages translated for natural phrasing.
- Dataset cards now offer download-only flow with on-demand previews (first 10 rows), improved layout that keeps previews within their cards.
- Landing, map, prediction, batch, model, and team pages polished for consistent copy and units.

## Backend & metadata
- Default `GLB_MODEL_VERSION` bumped to `0.2.2` and propagated to backend config, Dockerfile, and web docs.

## UX/layout fixes
- Dataset preview containers now shrink inside the card (min-w-0 + overflow handling) to avoid grid collisions.
- Map legend, filters, and hover copy clarified; Single/Batch prediction helper text refined.

## Verification
- Frontend build: `cd web/app && npm run build` (pass; known chunk-size warning).

## Notes
- Public site: `https://globalbrine-web-gwvzz.ondigitalocean.app/`.
