---
title: CV Detection Demo
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# CV Object Detection — Full CI/CD Demo

A small object detection service (YOLOv8 + FastAPI) built specifically to
demonstrate a complete, ML-aware CI/CD pipeline: linting, unit tests, model
sanity tests, Docker packaging, and automated deployment to Hugging Face
Spaces.

## Architecture

```
app/
  detector.py   - Core detection logic (model loading + inference)
  main.py       - FastAPI service exposing /detect over HTTP
tests/
  test_detector.py - Code-level tests + ML-specific model sanity tests
.github/workflows/
  ci.yml  - Lint + test on every push/PR
  cd.yml  - Build Docker image + deploy to HF Spaces on merge to main
Dockerfile - Containerizes the service for deployment
```

## Running locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
# then POST an image to http://localhost:8000/detect
```

## Running tests

```bash
pytest tests/ -v
```

## Building the Docker image

```bash
docker build -t cv-detection-api .
docker run -p 7860:7860 cv-detection-api
```

## The CI/CD pipeline

**CI (`ci.yml`)** runs on every push and PR:
1. Lint (`ruff`) — catches style and obvious bugs fast, before tests even run
2. Unit tests — verify code logic (error handling, data structures)
3. Model sanity tests — verify the *model* behaves correctly:
   - loads successfully
   - inference runs end-to-end without error
   - output schema (types, value ranges) is correct
   - confidence threshold is respected

**CD (`cd.yml`)** runs after CI passes on `main`:
1. Builds the Docker image (validates the container actually builds)
2. Pushes the repo to a Hugging Face Space, which triggers HF's own
   Docker-based build and redeploys the live demo automatically

## Why ML CI/CD is different from typical software CI/CD

Standard software CI mostly asks "does the code work as written?" ML
systems can pass that check and still be silently broken — wrong
preprocessing, a stale model file, an output format change that breaks a
downstream consumer. So ML CI needs an additional layer of **model-level
sanity tests**: does the model load, does inference run, are output
shapes/types correct, is performance within expected bounds. This project's
test suite is deliberately split into code-level and model-level tests for
exactly this reason — see the docstring at the top of `tests/test_detector.py`.

## Model versioning

Model weights are *not* committed to git (see `.gitignore`) — binary
weight files bloat repository size and git diffs are meaningless for them.
Instead:

- **In this minimal demo**: weights are pulled from the public Ultralytics
  release on each CI run / Docker build, pinned implicitly by the
  `ultralytics` package version in `requirements.txt`.
- **In a production setup**, you'd manage weights with a tool built for
  large binary versioning, most commonly **DVC (Data Version Control)** or
  cloud storage (S3/GCS) with a manifest file pinning a specific weights
  hash/version per release, so:
  - the exact model version is reproducible from any git commit
  - you can roll back a bad model deployment the same way you'd roll back
    a bad code deployment
  - weight changes go through the same review/CI process as code changes

This is worth saying explicitly in an interview: "code versioning and
model versioning are separate concerns, and conflating them by committing
weights directly to git doesn't scale."

## What I'd add next (good interview talking points)

- A **regression test** against a curated real-world image with known
  expected detections, to catch accuracy regressions
- **Latency benchmarking** in CI (fail if p95 inference time regresses)
- **DVC** for full model lineage tracking
- A **staging environment** before production deploy (canary or blue/green)
- **Monitoring** in production: track confidence distributions and
  detection counts over time to catch data drift
