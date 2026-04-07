# DAO Ethics Engine

DAO Ethics Engine is a governance and reasoning node from the Sampo AI OS ecosystem. It provides the analytical layer behind DAO Hub's ethics surfaces: multi-agent consensus simulation, proposal evaluation, benefit-distribution analysis, evidence-quality scoring, and cognitive-bias detection.

This repository should be read as a standalone node around DAO Hub, not as the hub itself. DAO Hub remains the orchestration center. This node provides one of the governance subsystems that the hub can call into for ethics review, audit summaries, and consensus history.

See `PUBLIC_EDITION_SCOPE.md` for repository scope and runtime notes.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-service-009688)
![Docker](https://img.shields.io/badge/Docker-ready-blue)

## What This Node Does

This repository exposes three core capability groups:

1. Multi-agent consensus simulation
   Run belief propagation, Raft-style leader replication, or simplified Byzantine fault-tolerant consensus across a synthetic agent network.

2. Ethics evaluation for governance proposals
   Score proposal benefit distribution, evidence quality, concentration risk, and structural bias signals before a proposal is approved, reviewed, or rejected.

3. Audit and transparency surfaces
   Return immutable agent values, recent proposal evaluations, consensus history, and aggregate ethics trends suitable for DAO Hub dashboards.

## Capability Summary

- immutable values manifest for public-benefit-first decision framing
- Gini-based concentration checks for elite-capture detection
- evidence-quality scoring with sample-size and confidence thresholds
- structural cognitive-bias detection
- authenticated agent, consensus, simulation, and evaluation endpoints
- SQLite-backed local runtime for easy portfolio review
- Docker and local dev workflow suitable for quick inspection

## Architecture Overview

The repository is intentionally compact.

- `main.py` contains the API, data models, consensus algorithms, and ethics evaluation pipeline
- `tests/test_ethics_engine.py` validates core decision logic
- `docker-compose.yml` and `Dockerfile` provide the public runtime path

## Quick Start

### Docker

This is the fastest way to run the node.

```bash
docker compose up --build
```

By default, the service is published on `http://localhost:18006`.

Swagger UI:

```text
http://localhost:18006/docs
```

Health check:

```text
http://localhost:18006/health
```

Stop the stack:

```bash
docker compose down
```

### Local Development

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn main:app --reload --port 18006
```

Run tests:

```bash
pytest -q
```

## Configuration

The service is configured through environment variables.

Copy values from `.env.example` or set them directly in your shell:

- `APP_PORT`
- `SECRET_KEY`
- `DATABASE_URL`
- `ADMIN_USERNAME`
- `ADMIN_PASSWORD`

The shipped defaults are for local demo use only. Change `SECRET_KEY` and `ADMIN_PASSWORD` before any shared deployment.

## Authentication

The API uses Bearer tokens.

1. Request a token from `POST /api/v1/auth/token`
2. Use the returned access token for protected endpoints

Default local development credentials:

- username: `admin`
- password: `change-me`

Example using `application/x-www-form-urlencoded`:

```text
username=admin&password=change-me
```

## Main Endpoints

Public endpoint:

- `GET /api/v1/ethics/values`

Protected endpoints:

- `GET /api/v1/agents`
- `POST /api/v1/agents`
- `POST /api/v1/consensus/run`
- `GET /api/v1/consensus/history`
- `POST /api/v1/simulation/run`
- `GET /api/v1/simulation/runs`
- `POST /api/v1/ethics/evaluate`
- `GET /api/v1/ethics/evaluations`
- `GET /api/v1/ethics/audit`
- `POST /api/v1/ethics/gini/calculate`

## Example Ethics Evaluation Payload

```json
{
  "proposal_name": "Public AI Procurement Oversight Policy",
  "description": "Governance policy for AI procurement and citizen-facing oversight.",
  "beneficiary_distribution": [0.36, 0.24, 0.20, 0.20],
  "beneficiary_labels": ["Citizens", "SMEs", "Municipal teams", "Regulators"],
  "evidence_claims": [
    {
      "claim": "Pilot reduced manual review backlog by 28%",
      "sample_size": 180,
      "confidence": 0.91,
      "source": "public-sector pilot",
      "peer_reviewed": false
    },
    {
      "claim": "Oversight workflow improved consistency",
      "sample_size": 96,
      "confidence": 0.88,
      "source": "audit report",
      "peer_reviewed": false
    }
  ]
}
```

## Relationship To DAO Hub

This repository is one Wave 1 satellite around DAO Hub.

Public narrative:

- DAO Hub is the ecosystem center
- DAO Ethics Engine is a specialized governance-and-reasoning node
- both were developed within Sampo AI OS

That structure lets the hub remain the flagship while this repo shows a concrete, reviewable subsystem behind its governance outputs.
