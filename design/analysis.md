# Data Analysis Tools Design

**Status**: ✅ Implemented  
**Version**: 0.9.11  
**Last Updated**: 2025-03-08

---

## Problem Statement

How to comprehensively analyze DeepH prediction results and training datasets to understand model performance and data characteristics?

---

## Design Goals

1. **Multi-dimensional Analysis** - Analyze errors at multiple levels (entry, orbital, element, structure)
2. **Visualization** - Generate clear, informative plots
3. **Dataset Understanding** - Extract features and statistics for model training
4. **Validation** - Test equivariance and other physics properties

---

## Analysis Categories

### 1. Error Analysis (`analyze/error/`)

#### Multi-level Error Analysis

```python
class ErrorAnalyzer:
    """Multi-dimensional error analysis"""
    
    def analyze_entries(self, pred: np.ndarray, label: np.ndarray):
        """Entry-level error analysis"""
        errors = np.abs(pred - label)
        return {
            "mae": np.mean(errors),
            "rmse": np.sqrt(np.mean(errors**2)),
            "max": np.max(errors),
        }
    
    def analyze_orbital(self, pred: np.ndarray, label: np.ndarray, orbital_info: dict):
        """Orbital-resolved error analysis"""
        # Group by orbital types
        errors_by_orbital = {}
        for orb_type in orbital_info:
            mask = orbital_info[orb_type]
            errors_by_orbital[orb_type] = self.analyze_entries(
                pred[mask], label[mask]
            )
        return errors_by_orbital
    
    def analyze_element(self, pred: np.ndarray, label: np.ndarray, elements: list):
        """Element-resolved error analysis"""
        errors_by_element = {}
        for elem in set(elements):
            mask = np.array(elements) == elem
            errors_by_element[elem] = self.analyze_entries(pred[mask], label[mask])
        return errors_by_element
```

#### Visualization

```python
def plot_error_distribution(errors: np.ndarray, output_path: Path):
    """Plot error distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    axes[0].hist(errors, bins=50, alpha=0.7)
    axes[0].set_xlabel('Error (eV)')
    axes[0].set_ylabel('Count')
    
    # Scatter plot
    axes[1].scatter(label, pred, alpha=0.5, s=1)
    axes[1].plot([label.min(), label.max()], [label.min(), label.max()], 'r--')
    axes[1].set_xlabel('DFT (eV)')
    axes[1].set_ylabel('DeepH (eV)')
    
    plt.savefig(output_path, dpi=300)
```

### 2. Dataset Analysis (`analyze/dataset/`)

#### Feature Detection

```python
class DatasetAnalyzer:
    """Analyze dataset features and statistics"""
    
    def analysis_dft_features(self, common_orbital_types: str = None):
        """Detect dataset features"""
        features = {
            "elements": set(),
            "orbital_types": set(),
            "spinful": False,
            "structures": [],
        }
        
        for data_dir in self.data_dirs:
            info = load_json_file(data_dir / "info.json")
            
            # Collect elements
            features["elements"].update(info["elements_orbital_map"].keys())
            
            # Collect orbital types
            for elem, orbs in info["elements_orbital_map"].items():
                features["orbital_types"].update(orbs)
            
            # Check spin
            if info["spinful"]:
                features["spinful"] = True
        
        return features
```

#### Data Splitting

```python
def generate_data_split_json(
    data_dirs: list[Path],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    rng_seed: int = 137,
):
    """Generate train/val/test split"""
    np.random.seed(rng_seed)
    
    # Shuffle directories
    dirs = np.random.permutation(data_dirs)
    
    # Split
    n_train = int(len(dirs) * train_ratio)
    n_val = int(len(dirs) * val_ratio)
    
    split = {
        "train": [str(d) for d in dirs[:n_train]],
        "validate": [str(d) for d in dirs[n_train:n_train+n_val]],
        "test": [str(d) for d in dirs[n_train+n_val:]],
    }
    
    return split
```

#### Edge Statistics

```python
def statistic_edge_quantity(data_dirs: list[Path]):
    """Count edges in dataset"""
    edge_counts = []
    
    for data_dir in data_dirs:
        with h5py.File(data_dir / "hamiltonian.h5", 'r') as f:
            n_edges = len(f['atom_pairs'])
            edge_counts.append(n_edges)
    
    return {
        "min": min(edge_counts),
        "max": max(edge_counts),
        "mean": np.mean(edge_counts),
        "total": sum(edge_counts),
    }
```

### 3. Equivariance Testing (`analyze/dft_equiv/`)

#### Generate Test Structures

```python
def generate_equivariance_structures(
    structure: dict,
    transformations: list[str] = ["translate", "rotate"],
):
    """Generate transformed structures for equivariance test"""
    
    test_structures = {}
    
    if "translate" in transformations:
        # Translation test
        for shift in [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]:
            translated = translate_structure(structure, shift)
            test_structures[f"trans_{shift}"] = translated
    
    if "rotate" in transformations:
        # Rotation test
        for angle in [30, 60, 90]:
            rotated = rotate_structure(structure, angle)
            test_structures[f"rot_{angle}"] = rotated
    
    return test_structures
```

#### Test Equivariance

```python
def test_equivariance(
    original_H: np.ndarray,
    transformed_H: np.ndarray,
    transformation: str,
    tolerance: float = 1e-10,
) -> bool:
    """Test if Hamiltonian transforms correctly"""
    
    if transformation == "translate":
        # H should be invariant under translation
        error = np.max(np.abs(original_H - transformed_H))
        return error < tolerance
    
    elif transformation == "rotate":
        # H should transform as: H' = R * H * R†
        # where R is Wigner D matrix
        expected_H = apply_wigner_rotation(original_H, rotation_matrix)
        error = np.max(np.abs(expected_H - transformed_H))
        return error < tolerance
```

---

## Visualization Tools

### 1. Error Heatmaps

```python
def plot_error_heatmap(
    errors: np.ndarray,
    x_labels: list[str],
    y_labels: list[str],
    output_path: Path,
):
    """Plot error heatmap"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(errors, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(x_labels)))
    ax.set_yticks(range(len(y_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_yticklabels(y_labels)
    
    plt.colorbar(im, ax=ax, label='Error (eV)')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
```

### 2. Orbital Error Distribution

```python
def plot_orbital_errors(
    orbital_errors: dict[str, dict],
    output_path: Path,
):
    """Plot orbital-resolved errors"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    orb_types = list(orbital_errors.keys())
    maes = [orbital_errors[orb]["mae"] for orb in orb_types]
    
    ax.bar(orb_types, maes, alpha=0.7)
    ax.set_xlabel('Orbital Type')
    ax.set_ylabel('MAE (eV)')
    ax.set_title('Orbital-Resolved Error')
    
    plt.savefig(output_path, dpi=300)
```

### 3. Structure-wise Error Distribution

```python
def plot_structure_errors(
    structure_errors: list[float],
    output_path: Path,
):
    """Plot error distribution across structures"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(structure_errors, bins=30, alpha=0.7, edgecolor='black')
    ax.set_xlabel('MAE (eV)')
    ax.set_ylabel('Number of Structures')
    ax.set_title('Structure-wise Error Distribution')
    
    plt.savefig(output_path, dpi=300)
```

---

## Usage Examples

### Error Analysis

```bash
# Entry-level error analysis
dock analyze error entries data_dir/ pred_dir/ -o error_analysis

# Orbital-resolved error
dock analyze error orbital data_dir/ pred_dir/ -o orbital_error.png

# Element-pair error
dock analyze error element-pair data_dir/ pred_dir/ -o element_pair_error.png
```

### Dataset Analysis

```bash
# Analyze features
dock analyze dataset features data_dir/ -o features.json

# Generate data split
dock analyze dataset split data_dir/ --split-ratio 0.6 0.2 0.2 -o split.json

# Edge statistics
dock analyze dataset edge data_dir/ -o edge_stats.png
```

### Equivariance Testing

```bash
# Generate test structures
dock analyze dft-equiv gen input_dir/ -o test_structures/

# Test equivariance
dock analyze dft-equiv test test_structures/ pred_dir/ -o equiv_report.json
```

---

**Last Updated**: 2025-03-08
