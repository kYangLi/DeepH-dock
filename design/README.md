# DeepH-dock Technical Design Documents

This directory contains technical design documents for developers and AI agents working on DeepH-dock. These documents explain key design decisions, architecture patterns, and development guidelines.

**Quick Links**:
- 📋 **[TODO.md](TODO.md)** - Planned features and work in progress
- 📖 **[development.md](development.md)** - How to extend DeepH-dock
- ❓ **[FAQ.md](FAQ.md)** - Common development questions
- 🏗️ **[architecture.md](architecture.md)** - System architecture overview

## Document Organization

```
design/
├── README.md              # This file - Quick navigation
├── architecture.md        # System architecture and core design patterns
├── development.md         # Development guide for extending DeepH-dock
├── converters.md          # DFT format converter design (SIESTA, OpenMX, etc.)
├── compute.md             # Electronic structure computation design
├── analysis.md            # Data analysis tools design
├── FAQ.md                 # Frequently asked questions for developers
└── TODO.md                # Planned features and work in progress
```

## Quick Navigation

### Core Architecture

- **[System Architecture](architecture.md)**
  - CLI auto-registration system
  - Unified data format design
  - Modular architecture pattern
  - Parallel processing strategy

### Development Guide

- **[Development Guide](development.md)**
  - Adding new DFT converters
  - Adding new computation functions
  - Adding new analysis tools
  - Testing and validation

### Module Designs

- **[DFT Converters](converters.md)**
  - Common conversion pipeline
  - Parser design patterns
  - Basis set handling
  - Spin-orbit coupling

- **[Electronic Structure Computation](compute.md)**
  - Eigenvalue calculation system
  - Band structure computation
  - DOS calculation
  - Ill-conditioned eigenvalue handling

- **[Data Analysis Tools](analysis.md)**
  - Error analysis system
  - Dataset analysis
  - Equivariance testing
  - Visualization tools

### Reference

- **[FAQ](FAQ.md)** - Common development questions
- **[TODO](TODO.md)** - Planned features and priorities

## Document Format

Each design document follows this structure:

1. **Problem Statement** - What problem does this design solve?
2. **Design Goals** - What are we trying to achieve?
3. **Solution Overview** - High-level approach
4. **Implementation Details** - Key code patterns and decisions
5. **Usage Guide** - How to use/extend this feature

## Target Audience

These documents are written for:
- **Developers** extending DeepH-dock with new features
- **AI agents** working on code generation and refactoring
- **Contributors** understanding the codebase architecture

## Key Design Principles

1. **Modularity** - Clear separation of concerns (convert/compute/analyze/design)
2. **Automation** - CLI auto-registration, minimal boilerplate
3. **Standardization** - Unified data format across all DFT codes
4. **Extensibility** - Easy to add new DFT converters and functions
5. **Performance** - Multi-level parallel processing (MPI/threads)

## Related Documentation

- `AGENTS.md` - Build/test/lint commands, code style guidelines
- `README.md` - Project overview and quick start
- `docs/` - User-facing documentation
- `examples/` - Usage examples and tutorials

## Contributing

When adding a new design document or updating existing ones:

1. Follow the document format above
2. Focus on design decisions and implementation patterns
3. Include code examples where helpful
4. Keep it concise and developer-focused
5. Update this README.md with links to new documents

## When to Update Design Docs

Update these documents when:
- Adding major new features or modules
- Changing core architecture patterns
- Introducing new design patterns
- Improving performance or scalability
- Fixing major design issues

---

**Last Updated**: 2025-03-08

**Maintainer**: DeepH Team <deeph-pack@outlook.com>
