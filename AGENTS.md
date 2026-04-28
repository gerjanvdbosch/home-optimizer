# AGENTS.md

This repository follows a clean, layered architecture. Any agent working in this codebase should preserve that structure and prefer reuse over local convenience.

## Core principles

1. Do not duplicate code.
2. Keep responsibilities inside the correct layer.
3. Prefer shared domain abstractions over feature-local copies.
4. Add new logic in the narrowest correct place, not the nearest convenient file.
5. Keep names and behavior consistent across the system.

## No duplicate code

Duplicate code is not acceptable here, especially for:

- time-series alignment logic
- derived signal calculations
- constant sensor/series names
- mapping logic between layers
- validation or parsing rules

If logic already exists somewhere else, reuse it or move it to a shared location.

Bad patterns:

- copying helper functions like `latest_value_at` into multiple modules
- redefining string constants like `"shutter_living_room"` in multiple places
- rebuilding the same derived series logic separately for dashboarding and identification

Preferred patterns:

- shared names in `src/home_optimizer/domain/names.py`
- shared generic series types in `src/home_optimizer/domain/series.py`
- shared series transformations in `src/home_optimizer/domain/series_transforms.py`

## Architecture layers

The intended layers are:

- `domain`
- `features`
- `infrastructure`
- `app`
- `web`
- `entrypoints`

### Domain

`src/home_optimizer/domain/`

Put pure, reusable concepts here:

- canonical names/constants
- generic domain models
- generic time-series types
- pure transformations and calculations
- unit/domain parsing helpers

Domain code should not know about:

- FastAPI
- HTML
- database sessions
- Home Assistant HTTP details
- UI/chart response models

### Features

`src/home_optimizer/features/`

Put use-case logic here:

- telemetry collection
- forecast assembly
- history import
- identification/model fitting

Feature code may orchestrate domain logic, but should not redefine domain concepts locally if they can be shared.

If two features need the same calculation, it should usually move down into `domain`.

### Infrastructure

`src/home_optimizer/infrastructure/`

Put adapters here:

- database repositories
- external API gateways
- local JSON gateways

Infrastructure should implement access to the outside world, not hold business rules that belong in domain/features.

### App

`src/home_optimizer/app/`

Put composition and runtime wiring here:

- container setup
- scheduler setup
- settings loading
- job orchestration

This layer wires features and infrastructure together.

### Web

`src/home_optimizer/web/`

Put presentation/API concerns here:

- routers
- response schemas
- HTML rendering
- web-specific mapping

This layer should stay thin.

Important:

- web response models may be chart-oriented
- domain models should stay generic
- do not move chart/presentation semantics into domain just because the dashboard needs them
- do not expose infrastructure models or ORM entities directly to web responses

Map explicitly:

infrastructure -> domain -> web response model

### Entrypoints

`src/home_optimizer/entrypoints/`

Put startup code here for local/add-on execution.

## Dependency direction

Dependencies should point inward:

web -> app -> features -> domain

Infrastructure implements dependencies required by features,
but domain must never depend on infrastructure.

Never import upward across layers.

Examples:

- `web` may depend on `features`
- `features` may depend on `domain`
- `domain` must not depend on `web`
- `domain` must not depend on `infrastructure`
- `infrastructure` adapts external systems into domain concepts

## Rules for adding new code

When adding code, ask:

1. Is this a generic concept? Put it in `domain`.
2. Is this use-case orchestration? Put it in `features`.
3. Is this external I/O? Put it in `infrastructure`.
4. Is this app composition/runtime wiring? Put it in `app`.
5. Is this API or UI formatting? Put it in `web`.

Prefer composition over inheritance.

Do not introduce class hierarchies unless polymorphism is genuinely needed.
Prefer small explicit objects and pure functions.

## Rules for names and types

- Reuse canonical names from `domain/names.py`.
- Reuse generic series types from `domain/series.py`.
- Reuse shared transforms from `domain/series_transforms.py`.
- Do not introduce new string literals for known series names when a constant already exists.
- Do not create feature-local variants of shared types unless there is a strong boundary reason.

## Refactoring guidance

If you notice duplication:

1. Stop adding more copies.
2. Extract the shared logic.
3. Move it to the lowest sensible layer.
4. Update callers to use the shared implementation.

Prefer one clean shared abstraction over several similar helpers.

## What to avoid

- fat web services with domain calculations that are also needed elsewhere
- repositories returning presentation-specific concepts when a generic domain type is sufficient
- hardcoded sensor names repeated across modules
- local one-off helpers that duplicate existing transformations
- bypassing the layer boundaries because it feels faster

## Target shape

The target architecture is:

- `domain` contains reusable concepts and calculations
- `features` contain workflows/use cases
- `infrastructure` contains adapters
- `app` contains composition
- `web` contains presentation

If in doubt, choose the option that reduces duplication and strengthens the layer boundaries.
