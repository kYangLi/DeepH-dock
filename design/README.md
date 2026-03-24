# DeepH-dock Design Documents

Technical design documents for developers and AI agents.

## Quick Navigation

| Document | Description |
|----------|-------------|
| [TODO.md](TODO.md) | Planned features and progress |
| [architecture.md](architecture.md) | System architecture, CLI auto-registration |
| [development.md](development.md) | How to extend DeepH-dock |
| [basis_standardization.md](basis_standardization.md) | Basis set format & overlap calculation |
| [converters.md](converters.md) | DFT format converter design |
| [compute.md](compute.md) | Electronic structure computation |
| [analysis.md](analysis.md) | Data analysis tools |
| [FAQ.md](FAQ.md) | Common development questions |

## Key Design Documents

### Basis Standardization (v0.9.12)

**File**: [basis_standardization.md](basis_standardization.md)

Standardized basis.h5 format for all DFT codes:
- Flat HDF5 structure optimized for AI/ML
- OpenMX PAO → HDF5 conversion
- HPRO integration for overlap calculation

```
basis.h5
├── @element, @basis_name, @source, @normalized, @units_length
├── radial_grid: [Nr]
├── mul_list: [lmax+1]
└── radial_basis: [total_orbitals, Nr]
```

### Architecture

**File**: [architecture.md](architecture.md)

Four-layer modular architecture:
- CLI Layer (auto-registration)
- convert / compute / analyze / design modules
- Unified Data Format Layer

## Design Principles

1. **Modularity** - Clear separation of concerns
2. **Automation** - CLI auto-registration, minimal boilerplate
3. **Standardization** - Unified data format across all DFT codes
4. **Extensibility** - Easy to add new converters and functions
5. **Performance** - Multi-level parallel processing

## Related Documentation

| Location | Description |
|----------|-------------|
| `AGENTS.md` | Build/test/lint commands, code style |
| `README.md` | Project overview |
| `examples/` | Usage examples and tutorials |

---

**Last Updated**: 2025-03-24
**Maintainer**: DeepH Team <deeph-pack@outlook.com>
