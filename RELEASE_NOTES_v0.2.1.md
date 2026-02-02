# Release Notes — v0.2.1 (February 2, 2026)

## Highlights
- Scientist-focused UI polish: refreshed typography/gradients, streamlined navigation, and methods/limits copy.
- Added dataset preview on the landing page with inline stats and one-click CSV download.
- Map explorer filter UX upgraded (range sliders + counts) with clearer units/legend.
- Single prediction form now has example autofill; batch jobs include example rows and quick recipe.
- New Team page credits MBZUAI (ML Department) and HIT Zongyao Zhou Group; added public code repo link in header.
- Public site & homepage screenshot documented in both root and web READMEs.

## Backend & metadata
- Default `GLB_MODEL_VERSION` bumped to `0.2.1`; propagated to Dockerfile, config, and DO app spec.

## Assets
- Added homepage PNG (`web/app/public/homepage.png`) and pipeline overview SVG surfaced in README.

## Verification
- Frontend build: `cd web/app && npm run build` (pass; known chunk size warning).

## Notes
- Web repo is public at `https://github.com/JiahaoZhang-Public/GlobalBrine-LithiumModel`.
- Production URL: `https://globalbrine-web-gwvzz.ondigitalocean.app/`.
