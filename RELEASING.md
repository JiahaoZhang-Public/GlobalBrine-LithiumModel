# Releasing

This repository uses lightweight, manual releases via Git tags.

## Checklist

1. Ensure CI is green on `main`.
2. Ensure the version is correct:
   - `setup.py` (`version="..."`)
   - `src/__init__.py` (`__version__ = "..."`)
3. Update `CHANGELOG.md` for the release.
4. Create and push an annotated tag:

```bash
git checkout main
git pull --ff-only

git tag -a v0.1.0 -m "v0.1.0"
git push origin v0.1.0
```

5. (GitHub) Create a Release from the tag and paste highlights from `CHANGELOG.md`.

## Next version

For the next release, bump the version in `setup.py` and `src/__init__.py`, add a new
section to `CHANGELOG.md`, and repeat the checklist.

