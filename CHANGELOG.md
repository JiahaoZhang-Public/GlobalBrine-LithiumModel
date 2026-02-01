# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
