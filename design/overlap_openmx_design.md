# OpenMX Overlap矩阵计算模块设计文档

> 版本: 1.0
> 日期: 2026-03-13
> 作者: DeepH Team

---

## 目录

1. [概述](#1-概述)
2. [架构设计](#2-架构设计)
3. [数据格式规范](#3-数据格式规范)
4. [Python模块设计](#4-python模块设计)
5. [C++核心库设计](#5-c核心库设计)
6. [编译系统](#6-编译系统)
7. [测试方案](#7-测试方案)
8. [实施计划](#8-实施计划)

---

## 1. 概述

### 1.1 目标

实现高性能、高精度的OpenMX风格overlap矩阵计算模块，替代当前依赖的外部HPRO库。

### 1.2 核心特性

- **统一的Basis数据格式**: HDF5格式存储，支持对数网格和线性网格
- **高性能C++核心**: 使用Eigen3进行矩阵运算，pybind11提供Python接口
- **k空间方法**: 采用OpenMX的k空间积分算法，避免实空间双中心积分
- **预计算优化**: k空间径向函数可预计算并缓存到H5文件

### 1.3 数学原理

Overlap矩阵元素：
$$S_{\alpha\beta} = \int \phi_\alpha^*(\mathbf{r}) \phi_\beta(\mathbf{r} - \mathbf{R}) \, d^3r$$

k空间计算方法：
1. 傅里叶变换: $\tilde{R}_\ell(k) = \int_0^\infty j_\ell(kr) R_\ell(r) r^2 \, dr$
2. k空间积分: $I_{\ell_1 \mu_1, \ell_2 \mu_2}^\ell(R) = \int_0^\infty \tilde{R}_{\ell_1\mu_1}(k) \tilde{R}_{\ell_2\mu_2}(k) j_\ell(kR) k^2 \, dk$
3. 角度耦合: $S = \sum_{\ell,m} 8(-i)^{-\ell_1+\ell_2+\ell} C_{\ell_1 m_1, \ell_2 m_2}^{\ell m} Y_\ell^m(\hat{\mathbf{R}}) I$

---

## 2. 架构设计

### 2.1 模块结构

```
deepx_dock/
└── compute/
    └── overlap/
        └── openmx/                      # OpenMX overlap模块
            │
            ├── __init__.py              # 模块初始化
            ├── _cli.py                  # CLI命令接口
            ├── calculator.py            # Python高层接口
            │
            ├── basis/                   # Basis管理(完全在此)
            │   ├── __init__.py
            │   ├── schema.py           # Basis HDF5数据格式定义
            │   ├── parser.py           # PAO文件解析器
            │   ├── converter.py        # PAO→H5转换器
            │   └── database/           # 预编译数据库
            │       └── openmx/
            │           ├── H.h5
            │           ├── C.h5
            │           ├── N.h5
            │           ├── O.h5
            │           └── ...
            │
            └── cpp/                     # C++核心计算库
                ├── include/
                │   ├── overlap.hpp     # 主计算接口
                │   ├── basis.hpp       # Basis数据结构
                │   ├── bessel.hpp      # 球贝塞尔函数
                │   ├── gaunt.hpp       # Gaunt系数
                │   ├── fourier.hpp     # 傅里叶变换
                │   ├── integral.hpp    # k空间积分
                │   └── spherical.hpp   # 球谐函数
                │
                ├── src/
                │   ├── overlap.cpp
                │   ├── basis.cpp
                │   ├── bessel.cpp
                │   ├── gaunt.cpp
                │   ├── fourier.cpp
                │   ├── integral.cpp
                │   └── spherical.cpp
                │
                ├── binding/
                │   └── pybind.cpp      # pybind11绑定
                │
                ├── CMakeLists.txt      # CMake配置
                ├── setup.py            # Python编译脚本
                └── README.md           # C++库说明
```

### 2.2 数据流

```
OpenMX .pao文件
    ↓ [parser.py解析]
PAORawData (Python dataclass)
    ↓ [converter.py转换]
Basis HDF5文件
    ↓ [C++ basis.cpp加载]
BasisSet (C++ class)
    ↓ [傅里叶变换]
KSpaceData (C++ class)
    ↓ [k空间积分]
RadialIntegralResult
    ↓ [角度耦合]
Overlap Matrix
```

### 2.3 设计决策

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 网格类型 | 支持多种(对数/线性) | 兼容不同DFT软件，OpenMX用对数网格 |
| Python绑定 | pybind11 | 现代C++绑定，支持numpy，类型安全 |
| Basis分发 | 随包发布H5文件 | 用户开箱即用，避免运行时转换 |
| 模块位置 | `compute.overlap.openmx` | 独立实现，不依赖HPRO |
| 矩阵存储 | Eigen稀疏矩阵 | 截断距离外元素为零，节省内存 |

---

## 3. 数据格式规范

### 3.1 Basis HDF5文件结构

```h5
element_basis.h5
│
├── metadata/
│   ├── version: "1.0.0"              # 数据格式版本
│   ├── created: "2026-03-13"         # 创建时间
│   ├── source: "openmx"              # 来源: openmx/siesta/abacus
│   └── description: "C7.0 basis"     # 描述
│
├── atomic_number: int                # 原子序数 (如 6)
├── symbol: str                       # 元素符号 (如 "C")
├── valence_electrons: float          # 价电子数
├── mass: float                       # 原子质量
│
└── basis_sets/
    └── 7.0/                          # Basis set名称(截断半径)
        ├── metadata/
        │   ├── radial_cutoff: 7.0    # Bohr
        │   ├── lmax: 3               # 最大角动量
        │   ├── num_mu: 15            # 每个L的径向函数数
        │   ├── grid_type: "log"      # 网格类型
        │   ├── grid_num: 500         # 网格点数
        │   └── eigenvalues: array    # (lmax+1, num_mu)
        │
        ├── radial_grid/
        │   ├── x: array (N,)         # log(r) 或 r
        │   ├── r: array (N,)         # 径向距离 (Bohr)
        │   └── dr: array (N,)        # 网格间距
        │
        ├── radial_wf/
        │   └── data: array (lmax+1, num_mu, N)
        │       # R[L][mu][i] - 径向波函数
        │
        ├── k_space/                  # 可选,预计算
        │   ├── k_grid: array (N_k,)
        │   ├── wf: array (lmax+1, num_mu, N_k)
        │   │   # R̃[L][mu][k] - k空间波函数
        │   ├── k_max: 20.0
        │   └── num_k: 500
        │
        └── valence_density/          # 可选
            └── data: array (N,)
```

### 3.2 Python数据类

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np
import h5py

class GridType(Enum):
    LOG = "log"
    LINEAR = "linear"

@dataclass
class RadialGrid:
    grid_type: GridType
    num_points: int
    x: np.ndarray           # (N,)
    r: np.ndarray           # (N,)
    dr: np.ndarray          # (N,)

@dataclass
class BasisMetadata:
    radial_cutoff: float
    lmax: int
    num_mu: int
    grid_type: GridType
    grid_num: int
    eigenvalues: np.ndarray  # (lmax+1, num_mu)

@dataclass
class KSpaceData:
    k_grid: np.ndarray       # (N_k,)
    wf: np.ndarray           # (lmax+1, num_mu, N_k)
    k_max: float
    num_k: int

@dataclass
class BasisSet:
    name: str
    metadata: BasisMetadata
    radial_grid: RadialGrid
    radial_wf: np.ndarray    # (lmax+1, num_mu, N)
    k_space: Optional[KSpaceData] = None
    valence_density: Optional[np.ndarray] = None
    
    def save_h5(self, group: h5py.Group): ...
    @classmethod
    def load_h5(cls, group: h5py.Group) -> 'BasisSet': ...

@dataclass
class ElementBasis:
    atomic_number: int
    symbol: str
    valence_electrons: float
    mass: float
    basis_sets: dict  # name -> BasisSet
    
    def save_h5(self, filepath: str): ...
    @classmethod
    def load_h5(cls, filepath: str) -> 'ElementBasis': ...
```

---

## 4. Python模块设计

### 4.1 PAO解析器 (parser.py)

```python
@dataclass
class PAORawData:
    atom_species: int
    total_electrons: float
    valence_electrons: float
    grid_xmin: float
    grid_xmax: float
    grid_num_total: int
    grid_num_output: int
    lmax: int
    num_mu: int
    radial_cutoff: float
    xv: np.ndarray              # (N,)
    rv: np.ndarray              # (N,)
    radial_wf: dict             # L -> (num_mu, N)
    eigenvalues: dict           # L -> (num_mu,)
    valence_density: np.ndarray # (N,)

def parse_pao_file(filepath: str | Path) -> PAORawData:
    """解析OpenMX .pao文件"""
    ...

def convert_pao_to_basis_set(pao_data: PAORawData) -> BasisSet:
    """将PAO数据转换为BasisSet"""
    ...
```

### 4.2 转换器 (converter.py)

```python
def convert_pao_file_to_h5(
    pao_file: str | Path,
    output_file: str | Path,
    compute_k_space: bool = False,
    k_max: float = 20.0,
    num_k: int = 500
):
    """
    将PAO文件转换为HDF5格式
    
    Parameters
    ----------
    pao_file : PAO文件路径
    output_file : 输出H5文件路径
    compute_k_space : 是否预计算k空间数据
    k_max : k空间最大值 (a.u.^-1)
    num_k : k空间网格点数
    """
    ...

def batch_convert_pao_dir(
    pao_dir: str | Path,
    output_dir: str | Path,
    compute_k_space: bool = False
):
    """批量转换目录下所有PAO文件"""
    ...
```

### 4.3 计算器 (calculator.py)

```python
class OverlapCalculator:
    """
    OpenMX风格overlap矩阵计算器
    
    Examples
    --------
    >>> calc = OverlapCalculator(basis_database_dir="./basis/database")
    >>> 
    >>> positions = np.array([[0, 0, 0], [1.42, 0, 0]])
    >>> species = np.array([6, 6])
    >>> calc.set_structure(positions, species)
    >>> 
    >>> calc.set_basis({6: "7.0"})
    >>> 
    >>> S = calc.compute(cutoff=10.0)
    """
    
    def __init__(self, basis_database_dir: str | Path, lmax_gaunt: int = 6):
        """
        Parameters
        ----------
        basis_database_dir : Basis数据库目录
        lmax_gaunt : Gaunt系数最大角动量
        """
        ...
    
    def set_structure(
        self,
        positions: np.ndarray,    # (N_atom, 3)
        species_ids: np.ndarray,  # (N_atom,)
        cell: Optional[np.ndarray] = None  # (3, 3)
    ):
        """设置原子结构"""
        ...
    
    def set_basis(self, basis_names: Dict[int, str]):
        """
        设置基组
        
        Parameters
        ----------
        basis_names : {atomic_number: basis_name}
            如 {6: "7.0", 1: "5.0"}
        """
        ...
    
    def compute(
        self,
        cutoff: float = 15.0,
        compute_derivative: bool = False
    ) -> sp.spmatrix:
        """
        计算overlap矩阵
        
        Parameters
        ----------
        cutoff : 截断距离 (Angstrom)
        compute_derivative : 是否计算导数
        
        Returns
        -------
        S : 稀疏矩阵 (N_basis, N_basis)
        """
        ...
    
    def compute_with_derivatives(
        self,
        cutoff: float = 15.0
    ) -> Tuple[sp.spmatrix, sp.spmatrix, sp.spmatrix, sp.spmatrix]:
        """
        计算overlap矩阵及导数
        
        Returns
        -------
        S, dS_dx, dS_dy, dS_dz
        """
        ...
    
    @property
    def total_basis_size(self) -> int:
        """总基函数数"""
        ...
```

### 4.4 CLI接口 (_cli.py)

```python
@register(
    cli_name="calc",
    cli_help="Calculate overlap matrix using OpenMX-style algorithm",
    cli_args=[
        click.argument('data_dir', type=click.Path(exists=True)),
        click.argument('basis_dir', type=click.Path(exists=True)),
        click.option('--cutoff', '-c', type=float, default=15.0),
        click.option('--output', '-o', type=click.Path(), default='overlap.h5'),
    ],
)
def calc_overlap(data_dir: Path, basis_dir: Path, cutoff: float, output: Path):
    """
    计算overlap矩阵
    
    Example
    -------
    dock compute overlap openmx calc ./data ./basis -c 10.0 -o overlap.h5
    """
    ...
```

---

## 5. C++核心库设计

### 5.1 数据结构

```cpp
namespace deepx {
namespace openmx {

enum class GridType { LOG, LINEAR };

struct RadialGrid {
    GridType grid_type;
    int num_points;
    Eigen::VectorXd x, r, dr;
    
    static RadialGrid load_h5(const H5::Group& group);
};

struct BasisMetadata {
    double radial_cutoff;
    int lmax, num_mu;
    Eigen::MatrixXd eigenvalues;
    
    static BasisMetadata load_h5(const H5::Group& group);
};

struct KSpaceData {
    Eigen::VectorXd k_grid;
    Eigen::VectorXcd wf;  // flatten: (lmax+1)*num_mu*N_k
    double k_max;
    int num_k, lmax, num_mu;
    
    Eigen::VectorXcd get_wf(int L, int mu) const;
    static KSpaceData load_h5(const H5::Group& group);
};

class BasisSet {
public:
    BasisSet(const std::string& h5_filepath);
    
    Eigen::VectorXd get_radial_wf(int L, int mu) const;
    const KSpaceData& get_k_space(double k_max = 20.0, int num_k = 500);
    
private:
    std::string name_;
    BasisMetadata metadata_;
    RadialGrid radial_grid_;
    Eigen::VectorXcd radial_wf_;
    std::shared_ptr<KSpaceData> k_space_;
    
    void compute_fourier_transform(double k_max, int num_k);
};

} // namespace openmx
} // namespace deepx
```

### 5.2 数学模块

#### 5.2.1 球贝塞尔函数 (bessel.hpp)

```cpp
class SphericalBessel {
public:
    static double compute(int l, double x);
    
    static Eigen::MatrixXd compute_batch(
        int lmax, 
        const Eigen::VectorXd& x_array
    );
    
    static double compute_derivative(int l, double x);
    
    static std::pair<Eigen::MatrixXd, Eigen::MatrixXd> 
    compute_batch_with_derivative(int lmax, const Eigen::VectorXd& x_array);
    
private:
    static void forward_recursion(double x, int lmax, std::vector<double>& jl);
};
```

**实现要点**:
- `j_0(x) = sin(x)/x`
- `j_1(x) = sin(x)/x² - cos(x)/x`
- 递推: `j_{l+1}(x) = (2l+1)/x * j_l(x) - j_{l-1}(x)`
- 导数: `j_l'(x) = j_{l-1}(x) - (l+1)/x * j_l(x)`
- 处理x=0的奇异性: `j_l(0) = δ_{l0}`

#### 5.2.2 Gaunt系数 (gaunt.hpp)

```cpp
class GauntCoefficients {
public:
    explicit GauntCoefficients(int lmax);
    
    double get(int l1, int m1, int l2, int m2, int l, int m) const;
    
    std::vector<std::tuple<int, int, double>> 
    get_all(int l1, int m1, int l2, int m2) const;
    
private:
    int lmax_;
    std::map<std::tuple<int,int,int,int,int,int>, double> cache_;
    
    static double wigner_3j(int j1, int j2, int j3, int m1, int m2, int m3);
    static double clebsch_gordan(int j1, int m1, int j2, int m2, int j, int m);
    static bool selection_rule(int l1, int m1, int l2, int m2, int l, int m);
    
    void precompute();
};
```

**数学公式**:
```
Gaunt系数:
C(l1,m1,l2,m2,l,m) = √[(2l1+1)(2l2+1)/(4π(2l+1))] 
                     × <l1,0,l2,0|l,0> 
                     × <l1,m1,l2,m2|l,m>

选择定则:
- |l1-l2| ≤ l ≤ l1+l2
- m = m1 + m2
```

#### 5.2.3 球谐函数 (spherical.hpp)

```cpp
class SphericalHarmonics {
public:
    static std::complex<double> compute(int l, int m, double theta, double phi);
    
    static double compute_real(int l, int m, double theta, double phi);
    
    static std::tuple<std::complex<double>, 
                      std::complex<double>, 
                      std::complex<double>>
    compute_with_derivatives(int l, int m, double theta, double phi);
};
```

**实现**:
```cpp
// 复数球谐函数
Y_l^m(θ,φ) = N * P_l^|m|(cos θ) * e^{imφ}

其中 N = √[(2l+1)/(4π) * (l-|m|)!/(l+|m|)!]

// 实数球谐函数
Y_l^m (real) = {
    Y_l^0,                               m=0
    √2 * Re[Y_l^|m|],                   m>0
    √2 * Im[Y_l^|m|],                   m<0
}
```

#### 5.2.4 傅里叶变换 (fourier.hpp)

```cpp
class FourierTransformer {
public:
    FourierTransformer(double k_max = 20.0, int num_k = 500);
    
    Eigen::VectorXcd transform(
        const Eigen::VectorXd& radial_wf,
        const Eigen::VectorXd& r_grid,
        int L
    );
    
    KSpaceData transform_all(const BasisSet& basis);
    
private:
    double k_max_;
    int num_k_;
    Eigen::VectorXd k_grid_;
    
    static double trapezoidal_integrate(
        const Eigen::VectorXd& y,
        const Eigen::VectorXd& x
    );
};
```

**算法**:
```cpp
// 傅里叶变换: R̃_L(k) = ∫ j_L(kr) R_L(r) r² dr
for each k in k_grid:
    kr = k * r_grid
    j_L = spherical_bessel(L, kr)
    integrand = radial_wf * j_L * r_grid²
    R̃_L(k) = trapezoidal_integrate(integrand, r_grid)
```

#### 5.2.5 k空间积分 (integral.hpp)

```cpp
struct RadialIntegralResult {
    Eigen::Tensor<double, 5> sumS0;   // (lmax_A+1, num_mu_A, lmax_B+1, num_mu_B, lmax_int+1)
    Eigen::Tensor<double, 5> sumSr0;  // 导数积分
    
    int lmax_A, num_mu_A, lmax_B, num_mu_B, lmax_int;
};

class KSpaceIntegrator {
public:
    static RadialIntegralResult compute_radial_integrals(
        const KSpaceData& k_space_A,
        const KSpaceData& k_space_B,
        double R,
        int lmax_int
    );
    
private:
    static std::pair<double, double> compute_single_integral(
        const Eigen::VectorXcd& wf_A,
        const Eigen::VectorXcd& wf_B,
        const Eigen::VectorXd& k_grid,
        double R,
        int l
    );
};
```

**算法**:
```cpp
// k空间积分
SumS0 = ∫ R̃_A(k) R̃_B(k) j_l(kR) k² dk
SumSr0 = ∫ R̃_A(k) R̃_B(k) j_l'(kR) k³ dk  // 用于力的计算

for l in 0..lmax_int:
    kR = k_grid * R
    j_l, j_l' = spherical_bessel_batch(l, kR)
    
    for L0, mu0 in basis_A:
        for L1, mu1 in basis_B:
            f = wf_A[L0, mu0] * wf_B[L1, mu1]
            
            SumS0[L0, mu0, L1, mu1, l] = ∫ f * j_l * k² dk
            SumSr0[L0, mu0, L1, mu1, l] = ∫ f * j_l' * k³ dk
```

### 5.3 主计算模块

```cpp
struct Atom {
    int index;
    int species_id;
    Eigen::Vector3d position;
    int basis_start;
    int num_basis;
};

struct AtomPair {
    int atom_i, atom_j;
    Eigen::Vector3i cell_offset;
    Eigen::Vector3d R_vec;
    double distance;
    double theta, phi;  // 球坐标
};

class OverlapCalculator {
public:
    OverlapCalculator(const std::string& basis_database_dir, int lmax_gaunt = 6);
    
    void set_structure(
        const Eigen::MatrixXd& positions,
        const Eigen::VectorXi& species_ids,
        const Eigen::Matrix3d& cell = Eigen::Matrix3d::Zero()
    );
    
    void set_basis(const std::map<int, std::string>& basis_names);
    
    Eigen::SparseMatrix<std::complex<double>> 
    compute(double cutoff = 15.0, bool compute_derivative = false);
    
    std::tuple<Eigen::SparseMatrix<std::complex<double>>,
               Eigen::SparseMatrix<std::complex<double>>,
               Eigen::SparseMatrix<std::complex<double>>,
               Eigen::SparseMatrix<std::complex<double>>>
    compute_with_derivatives(double cutoff = 15.0);
    
private:
    std::vector<AtomPair> find_atom_pairs(double cutoff);
    
    void compute_pair_overlap(
        const AtomPair& pair,
        Eigen::VectorXcd& values,
        std::vector<int>& rows,
        std::vector<int>& cols,
        bool compute_derivative
    );
    
    std::tuple<std::complex<double>, 
               std::complex<double>,
               std::complex<double>,
               std::complex<double>>
    compute_overlap_element(
        int L0, int mu0, int m0,
        int L1, int mu1, int m1,
        const RadialIntegralResult& radial,
        double theta, double phi,
        bool compute_derivative
    );
    
    static std::tuple<double, double, double> 
    cartesian_to_spherical(const Eigen::Vector3d& R_vec);
    
    static std::tuple<std::complex<double>, 
                      std::complex<double>, 
                      std::complex<double>>
    spherical_to_cartesian_derivatives(
        std::complex<double> dS_dr,
        std::complex<double> dS_dtheta,
        std::complex<double> dS_dphi,
        double theta, double phi, double R
    );
};
```

**核心算法流程**:
```cpp
1. 找到截断距离内的所有原子对
   for each atom_i:
       for each atom_j within cutoff:
           R_vec = position[j] - position[i] + cell_offset * cell
           R, theta, phi = cartesian_to_spherical(R_vec)
           pairs.push_back({i, j, cell_offset, R_vec, R, theta, phi})

2. 对每个原子对计算overlap
   for each pair:
       // 加载两个原子的基组
       basis_A = get_basis(atom_i)
       basis_B = get_basis(atom_j)
       
       // k空间径向积分
       lmax_int = 2 * max(basis_A.lmax, basis_B.lmax)
       radial = compute_radial_integrals(
           basis_A.k_space, basis_B.k_space, pair.R, lmax_int
       )
       
       // 角度耦合
       for L0, mu0, m0 in basis_A:
           for L1, mu1, m1 in basis_B:
               S = 0
               for l in |L0-L1| .. L0+L1:
                   m = m0 + m1
                   if |m| > l: continue
                   
                   gaunt = gaunt_coefficients.get(L0, m0, L1, m1, l, m)
                   Ylm = spherical_harmonic(l, m, pair.theta, pair.phi)
                   phase = 8 * (-i)^{-L0+L1+l}
                   
                   S += phase * gaunt * Ylm * radial.sumS0[L0, mu0, L1, mu1, l]
               
               // 填充矩阵
               i_basis = atom_i.basis_start + basis_index(L0, mu0, m0)
               j_basis = atom_j.basis_start + basis_index(L1, mu1, m1)
               S_matrix(i_basis, j_basis) = S

3. 组装稀疏矩阵
   triplet_list = [(i, j, S_ij) for each computed element]
   S_sparse.setFromTriplets(triplet_list)
```

### 5.4 pybind11绑定

```cpp
PYBIND11_MODULE(overlap_openmx, m) {
    m.doc() = "OpenMX-style overlap matrix calculation";
    
    py::class_<BasisSet, std::shared_ptr<BasisSet>>(m, "BasisSet")
        .def(py::init<const std::string&>())
        .def("get_radial_wf", &BasisSet::get_radial_wf)
        .def("get_k_space", &BasisSet::get_k_space)
        .def_property_readonly("name", &BasisSet::name)
        .def_property_readonly("lmax", ...)
        .def_property_readonly("num_mu", ...);
    
    py::class_<OverlapCalculator>(m, "OverlapCalculator")
        .def(py::init<const std::string&, int>())
        .def("set_structure", &OverlapCalculator::set_structure)
        .def("set_basis", &OverlapCalculator::set_basis)
        .def("compute", &OverlapCalculator::compute)
        .def("compute_with_derivatives", 
             &OverlapCalculator::compute_with_derivatives)
        .def_property_readonly("total_basis_size", 
                               &OverlapCalculator::total_basis_size);
    
    m.def("spherical_bessel", &SphericalBessel::compute);
    m.def("gaunt_coefficient", ...);
}
```

---

## 6. 编译系统

### 6.1 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.15)
project(overlap_openmx LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release)

# 优化选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math")

# 查找依赖
find_package(Eigen3 3.3 REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(pybind11 REQUIRED)

# 源文件
set(SOURCES
    src/basis.cpp
    src/bessel.cpp
    src/gaunt.cpp
    src/spherical.cpp
    src/fourier.cpp
    src/integral.cpp
    src/overlap.cpp
    binding/pybind.cpp
)

# 创建Python模块
pybind11_add_module(overlap_openmx ${SOURCES})

target_link_libraries(overlap_openmx PRIVATE
    Eigen3::Eigen
    ${HDF5_CXX_LIBRARIES}
    ${HDF5_LIBRARIES}
)

target_include_directories(overlap_openmx PRIVATE
    ${HDF5_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
    include
)

# 安装
install(TARGETS overlap_openmx LIBRARY DESTINATION ".")
```

### 6.2 编译脚本

```python
# setup.py
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import os
from pathlib import Path

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            if isinstance(ext, CMakeExtension):
                self.build_cmake(ext)
    
    def build_cmake(self, ext):
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        
        subprocess.check_call([
            'cmake', ext.sourcedir,
            f'-DCMAKE_BUILD_TYPE=Release',
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={self.build_lib}',
        ], cwd=build_temp)
        
        subprocess.check_call([
            'cmake', '--build', '.',
            '--config', 'Release',
            '-j', '4'
        ], cwd=build_temp)

setup(
    name='overlap_openmx',
    version='0.1.0',
    ext_modules=[CMakeExtension('overlap_openmx', '.')],
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
)
```

### 6.3 编译命令

```bash
# 开发时编译
cd deepx_dock/compute/overlap/openmx/cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

# 或通过setup.py
python setup.py build_ext --inplace

# 安装到Python环境
pip install -e .
```

---

## 7. 测试方案

### 7.1 单元测试

#### 测试球贝塞尔函数

```python
def test_spherical_bessel_l0():
    """j_0(x) = sin(x)/x"""
    x = np.linspace(0.01, 10, 100)
    for xi in x:
        expected = np.sin(xi) / xi
        actual = _cpp.spherical_bessel(0, xi)
        assert np.isclose(actual, expected, rtol=1e-10)

def test_spherical_bessel_recursion():
    """递推关系验证"""
    x = 2.5
    for l in range(1, 5):
        j_prev = _cpp.spherical_bessel(l-1, x)
        j_curr = _cpp.spherical_bessel(l, x)
        j_next = _cpp.spherical_bessel(l+1, x)
        expected = (2*l + 1)/x * j_curr - j_prev
        assert np.isclose(j_next, expected, rtol=1e-10)

def test_spherical_bessel_vs_scipy():
    """与scipy对比"""
    from scipy.special import spherical_jn
    x = np.linspace(0.1, 10, 50)
    for l in range(6):
        for xi in x:
            cpp_val = _cpp.spherical_bessel(l, xi)
            scipy_val = spherical_jn(l, xi)
            assert np.isclose(cpp_val, scipy_val, rtol=1e-8)
```

#### 测试Gaunt系数

```python
def test_gaunt_selection_rule():
    """测试选择定则"""
    gaunt = GauntCoefficients(lmax=4)
    
    # 违反选择定则应为0
    assert gaunt.get(2, 0, 2, 0, 0, 0) != 0  # l=0 在 |2-2|..4 范围内
    assert gaunt.get(2, 0, 2, 0, 5, 0) == 0  # l=5 超出范围
    
    # m = m1 + m2
    assert gaunt.get(1, 1, 1, 1, 2, 2) != 0  # m = 1+1 = 2
    assert gaunt.get(1, 1, 1, 1, 2, 0) == 0  # m ≠ 1+1

def test_gaunt_symmetry():
    """测试对称性"""
    gaunt = GauntCoefficients(lmax=4)
    
    # C(l1,m1,l2,m2,l,m) = C(l2,m2,l1,m1,l,m) (球谐函数交换)
    assert np.isclose(
        gaunt.get(1, 0, 2, 1, 2, 1),
        gaunt.get(2, 1, 1, 0, 2, 1),
        rtol=1e-10
    )
```

#### 测试傅里叶变换

```python
def test_fourier_transform_normalization():
    """测试傅里叶变换归一化"""
    from deepx_dock.compute.overlap.openmx.basis import PAORawData
    
    # 创建简单的测试数据
    r = np.linspace(0.01, 10, 500)
    R = np.exp(-r)  # 指数衰减函数
    
    pao_data = create_test_pao_data(r, R)
    basis = convert_pao_to_basis_set(pao_data)
    
    # 傅里叶变换
    k_space = basis.get_k_space(k_max=20.0, num_k=500)
    
    # 验证 Parseval 定理: ∫|R(r)|² r² dr ≈ ∫|R̃(k)|² k² dk
    integral_r = np.trapz(np.abs(R)**2 * r**2, r)
    integral_k = np.trapz(np.abs(k_space.wf[0, 0])**2 * k_space.k_grid**2, 
                          k_space.k_grid)
    
    # 允许一定误差
    assert np.isclose(integral_r, integral_k, rtol=0.1)
```

### 7.2 集成测试

#### 测试简单分子

```python
def test_h2_overlap():
    """测试H2分子"""
    calc = OverlapCalculator(basis_database_dir="./test_basis")
    
    # H2: 间距0.74 Å
    positions = np.array([[0, 0, 0], [0.74, 0, 0]])
    species = np.array([1, 1])
    calc.set_structure(positions, species)
    calc.set_basis({1: "5.0"})
    
    S = calc.compute(cutoff=5.0)
    
    # 验证
    assert S.shape[0] == S.shape[1]
    
    # 对角元素应接近1
    S_dense = S.toarray()
    assert np.allclose(np.diag(S_dense), 1.0, atol=1e-4)
    
    # 非对角元素非零
    off_diag = S_dense - np.diag(np.diag(S_dense))
    assert np.any(np.abs(off_diag) > 0.01)

def test_c2_overlap():
    """测试C2分子"""
    calc = OverlapCalculator(basis_database_dir="./test_basis")
    
    # C2: 间距1.24 Å
    positions = np.array([[0, 0, 0], [1.24, 0, 0]])
    species = np.array([6, 6])
    calc.set_structure(positions, species)
    calc.set_basis({6: "7.0"})
    
    S = calc.compute(cutoff=10.0)
    
    # 基本验证
    assert S.shape[0] > 0
    assert S.nnz > 0  # 非零元素数
```

#### 与OpenMX对比

```python
def test_vs_openmx_simple():
    """与OpenMX结果对比"""
    # 从OpenMX输出读取overlap
    openmx_overlap = read_openmx_scfout("openmx.scfout")
    
    # 使用我们的方法计算
    calc = OverlapCalculator(basis_database_dir="./basis")
    # ... 设置相同的结构和基组 ...
    our_overlap = calc.compute(cutoff=10.0).toarray()
    
    # 对比
    diff = np.abs(openmx_overlap - our_overlap)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"Max difference: {max_diff}")
    print(f"Mean difference: {mean_diff}")
    
    # 允许1%的相对误差
    assert max_diff < 0.01 * np.max(np.abs(openmx_overlap))
```

### 7.3 性能测试

```python
def test_performance_scaling():
    """测试性能随体系大小变化"""
    import time
    
    sizes = [10, 20, 50, 100]  # 原子数
    times = []
    
    calc = OverlapCalculator(basis_database_dir="./basis")
    
    for n_atom in sizes:
        # 创建随机结构
        positions = np.random.rand(n_atom, 3) * 10
        species = np.ones(n_atom, dtype=int) * 6  # 全碳
        
        calc.set_structure(positions, species)
        calc.set_basis({6: "7.0"})
        
        t0 = time.time()
        S = calc.compute(cutoff=10.0)
        t1 = time.time()
        
        times.append(t1 - t0)
        print(f"N_atom={n_atom}, Time={t1-t0:.2f}s, "
              f"N_basis={calc.total_basis_size}")
    
    # 绘制性能曲线
    import matplotlib.pyplot as plt
    plt.plot(sizes, times, 'o-')
    plt.xlabel('Number of atoms')
    plt.ylabel('Time (s)')
    plt.savefig('performance.png')
```

---

## 8. 实施计划

### 8.1 阶段1: 基础设施 (1周)

**任务**:
- [x] 创建design文档
- [ ] 创建模块目录结构
- [ ] 实现Basis HDF5格式规范 (schema.py)
- [ ] 实现PAO文件解析器 (parser.py)
- [ ] 实现PAO→H5转换器 (converter.py)
- [ ] 准备测试数据 (常见元素的PAO文件)

**输出**:
- 可运行的Python模块
- 至少转换3个元素(H, C, O)的basis到H5格式

### 8.2 阶段2: C++核心库 (2周)

**任务**:
- [ ] 实现球贝塞尔函数模块
- [ ] 实现Gaunt系数模块
- [ ] 实现球谐函数模块
- [ ] 实现傅里叶变换模块
- [ ] 实现k空间积分模块
- [ ] 实现overlap主计算模块
- [ ] 编译系统 (CMakeLists.txt, setup.py)
- [ ] pybind11绑定
- [ ] C++单元测试

**输出**:
- 可编译的C++库
- Python可调用的C++扩展模块

### 8.3 阶段3: Python接口 (3天)

**任务**:
- [ ] 实现Python高层接口 (calculator.py)
- [ ] 实现CLI命令 (_cli.py)
- [ ] 集成测试

**输出**:
- 完整的Python API
- CLI命令可用

### 8.4 阶段4: 测试和优化 (1周)

**任务**:
- [ ] 与OpenMX结果对比测试
- [ ] 性能测试和优化
- [ ] 文档编写
- [ ] 预编译basis数据库

**输出**:
- 验证正确性的测试报告
- 性能基准
- 完整文档

### 8.5 时间估算

| 阶段 | 时间 | 优先级 |
|------|------|--------|
| 阶段1: 基础设施 | 7天 | 高 |
| 阶段2: C++核心库 | 14天 | 高 |
| 阶段3: Python接口 | 3天 | 中 |
| 阶段4: 测试优化 | 7天 | 中 |
| **总计** | **31天** | |

---

## 9. 风险和缓解

### 9.1 技术风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 数值精度不足 | k空间积分误差大 | 使用高精度积分方法,增加网格点数 |
| 球贝塞尔函数递推不稳定 | 大l小x时失效 | 使用Miller算法或混合递推 |
| 编译依赖复杂 | 跨平台编译困难 | 提供Docker环境,详细的编译文档 |
| 性能不达预期 | 计算时间长 | 预计算k空间数据,OpenMP并行 |

### 9.2 缓解策略

1. **分阶段实现**: 先实现最核心功能,逐步扩展
2. **单元测试驱动**: 每个模块都有完整的单元测试
3. **参考OpenMX源码**: 遇到问题时参考OpenMX实现
4. **性能profiling**: 使用性能分析工具定位瓶颈

---

## 10. 参考资料

1. **OpenMX文档**: https://www.openmx-square.org/
2. **算法参考**: `/home/deeph/software/calc/OpenMX/build/openmx3.9/OVERLAP_ALGORITHM.md`
3. **OpenMX源码**: `/home/deeph/software/calc/OpenMX/build/openmx3.9/source/`
4. **Eigen3文档**: https://eigen.tuxfamily.org/
5. **pybind11文档**: https://pybind11.readthedocs.io/
6. **球谐函数**: https://en.wikipedia.org/wiki/Spherical_harmonics
7. **Gaunt系数**: https://en.wikipedia.org/wiki/Gaunt_coefficient

---

## 附录A: OpenMX相关源码文件

```
OpenMX source files:
- Overlap_Band.c: Band structure overlap matrix construction
- Overlap_Cluster.c: Cluster/molecular system overlap
- Spherical_Bessel.c: Spherical Bessel functions
- RF_BesselF.c: Radial functions in k-space
- Gaunt.c: Gaunt coefficients
- xyz2spherical.c: Cartesian to spherical coordinate conversion

Key functions:
- Overlap_Band(): Main overlap calculation for periodic systems
- Spherical_Bessel(): Spherical Bessel function computation
- Gaunt(): Gaunt coefficient calculation
- TwoCenter_Integral(): Two-center integral evaluation
```

---

## 附录B: 数学公式汇总

### B.1 球贝塞尔函数

$$j_0(x) = \frac{\sin x}{x}$$

$$j_1(x) = \frac{\sin x}{x^2} - \frac{\cos x}{x}$$

$$j_{\ell+1}(x) = \frac{2\ell+1}{x}j_\ell(x) - j_{\ell-1}(x)$$

$$j_\ell'(x) = j_{\ell-1}(x) - \frac{\ell+1}{x}j_\ell(x)$$

### B.2 球谐函数

$$Y_\ell^m(\theta,\phi) = N_{\ell m} P_\ell^{|m|}(\cos\theta) e^{im\phi}$$

$$N_{\ell m} = \sqrt{\frac{2\ell+1}{4\pi}\frac{(\ell-|m|)!}{(\ell+|m|)!}}$$

### B.3 Gaunt系数

$$C_{\ell_1 m_1, \ell_2 m_2}^{\ell m} = \int Y_{\ell_1}^{m_1} Y_{\ell_2}^{m_2} Y_\ell^{m*} d\Omega$$

$$= \sqrt{\frac{(2\ell_1+1)(2\ell_2+1)}{4\pi(2\ell+1)}} \langle \ell_1 0 \ell_2 0 | \ell 0 \rangle \langle \ell_1 m_1 \ell_2 m_2 | \ell m \rangle$$

### B.4 Overlap矩阵元素

$$S_{\alpha\beta} = \sum_{\ell,m} 8(-i)^{-\ell_1+\ell_2+\ell} C_{\ell_1 m_1, \ell_2 m_2}^{\ell m} Y_\ell^m(\hat{\mathbf{R}}) I_{\ell_1 \mu_1, \ell_2 \mu_2}^\ell(R)$$

$$I_{\ell_1 \mu_1, \ell_2 \mu_2}^\ell(R) = \int_0^\infty \tilde{R}_{\ell_1\mu_1}(k) \tilde{R}_{\ell_2\mu_2}(k) j_\ell(kR) k^2 dk$$

---

**文档结束**
