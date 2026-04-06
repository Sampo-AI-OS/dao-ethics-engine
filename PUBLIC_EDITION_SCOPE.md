# Public Edition Scope

This folder contains the curated public-edition version of the Multi-Agent AI Consensus + Ethics node that sits around DAO Hub inside the Sampo AI OS ecosystem.

It is published as a standalone node because it powers governance-facing ethics evaluations, consensus history, and reasoning surfaces that DAO Hub can expose without revealing the full private ecosystem implementation.

## Included

This public edition intentionally includes:

- a working FastAPI service for multi-agent consensus simulations
- the public ethics evaluation pipeline and immutable values manifest
- Docker runtime files and local development setup
- a lightweight regression test file for the core ethics logic
- documentation that explains how this node connects back to DAO Hub

## Intentionally Excluded

This repository does not include:

- private downstream integrations beyond this node boundary
- internal orchestration logic used only inside the broader DAO ecosystem
- unpublished governance workflows, strategy material, or commercial planning
- alternative stale router branches that are not part of the curated runtime
- any higher-value private implementations that would disclose disproportionate moat relative to portfolio value

## Narrative Role

In public ecosystem terms:

- DAO Hub remains the orchestration center
- this repository is one specialized governance-and-reasoning node around that center
- both were developed within Sampo AI OS

## Safety And Operational Notes

The public edition removes hard-coded production secrets and expects environment-based configuration for runtime credentials.

Default local credentials are intentionally development-only and should be changed before any non-local deployment.
