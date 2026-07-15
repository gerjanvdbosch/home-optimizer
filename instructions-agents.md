# Agent Instructions

## Architecture Principles

This project follows clean, DRY (Don't Repeat Yourself), and professional architectural principles:

1. **Clean Architecture**: Separation of concerns with clear boundaries between domain logic, application logic, and infrastructure.
2. **DRY Principle**: No code duplication; reusable components and abstractions.
3. **Professional Standards**: 
   - Type hints throughout
   - Comprehensive documentation
   - Consistent naming conventions
   - Modular design

## Pydantic Usage Requirements

All data models and configuration classes must use Pydantic for:
- Data validation
- Type hinting
- Serialization/deserialization
- Configuration management

### Model Guidelines

1. Use Pydantic v2 with `BaseModel` as base class
2. Define all fields with appropriate types
3. Include validation where necessary
4. Use `@validator` for custom validations when needed
5. Ensure models are serializable and deserializable

## Directory Structure

```
src/
├── domain/              # Core business logic
├── features/           # Feature modules
├── infrastructure/     # External integrations
└── web/                # Web-related components
```

## Code Quality Standards

1. All code must be properly typed
2. Use meaningful variable and function names
3. Write docstrings for all public functions and classes
4. Follow PEP 8 style guide
5. Include unit tests for all logic
6. Maintain consistent logging practices
7. Handle errors gracefully with appropriate exception types

## Development Workflow

1. All changes must pass linting (ruff) and type checking (mypy)
2. Unit tests must cover 100% of logic
3. Integration tests for major flows
4. Follow Git commit message conventions
5. Keep PRs small and focused
