#pragma once

#include <string>
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <H5Cpp.h>

namespace deepx {
namespace openmx {

/**
 * @brief Grid type for radial functions
 */
enum class GridType {
    LOG,      // Logarithmic grid: x = log(r)
    LINEAR    // Linear grid
};

/**
 * @brief Radial grid
 */
struct RadialGrid {
    GridType grid_type;
    int num_points;
    Eigen::VectorXd x;   // log(r) or r
    Eigen::VectorXd r;   // radial distance in Bohr
    Eigen::VectorXd dr;  // grid spacing
    
    static RadialGrid load_h5(const H5::Group& group);
};

/**
 * @brief Basis set metadata
 */
struct BasisMetadata {
    double radial_cutoff;
    int lmax;
    int num_mu;
    GridType grid_type;
    int grid_num;
    Eigen::MatrixXd eigenvalues;  // (lmax+1, num_mu)
    
    static BasisMetadata load_h5(const H5::Group& group);
};

/**
 * @brief k-space data (precomputed Fourier transform)
 */
struct KSpaceData {
    Eigen::VectorXd k_grid;               // (N_k,)
    Eigen::VectorXcd wf;                  // flattened (lmax+1, num_mu, N_k)
    double k_max;
    int num_k;
    int lmax;
    int num_mu;
    
    /**
     * @brief Get k-space wave function for specific (L, mu)
     */
    Eigen::VectorXcd get_wf(int L, int mu) const;
    
    static KSpaceData load_h5(const H5::Group& group);
};

/**
 * @brief Single basis set (e.g., C7.0)
 */
class BasisSet {
public:
    /**
     * @brief Load from HDF5 file
     */
    explicit BasisSet(const std::string& h5_filepath);
    
    // Getters
    const std::string& name() const { return name_; }
    const BasisMetadata& metadata() const { return metadata_; }
    const RadialGrid& radial_grid() const { return radial_grid_; }
    
    /**
     * @brief Get radial wave function for specific (L, mu)
     * @return R_L,mu(r), shape: (N,)
     */
    Eigen::VectorXd get_radial_wf(int L, int mu) const;
    
    /**
     * @brief Get or compute k-space data
     */
    const KSpaceData& get_k_space(double k_max = 20.0, int num_k = 500);
    
    /**
     * @brief Save k-space data to HDF5 (optional)
     */
    void save_k_space_h5(const std::string& filepath);
    
private:
    std::string name_;
    BasisMetadata metadata_;
    RadialGrid radial_grid_;
    Eigen::VectorXcd radial_wf_;  // flattened (lmax+1, num_mu, N)
    
    std::shared_ptr<KSpaceData> k_space_;
    
    /**
     * @brief Perform Fourier transform
     */
    void compute_fourier_transform(double k_max, int num_k);
};

/**
 * @brief Element basis (all basis sets for one element)
 */
class ElementBasis {
public:
    explicit ElementBasis(const std::string& h5_filepath);
    
    int atomic_number() const { return atomic_number_; }
    const std::string& symbol() const { return symbol_; }
    
    /**
     * @brief Get specific basis set by name
     */
    const BasisSet& get_basis_set(const std::string& name) const;
    
    /**
     * @brief Get default basis set (largest cutoff)
     */
    const BasisSet& get_default_basis_set() const;
    
    /**
     * @brief List all available basis set names
     */
    std::vector<std::string> list_basis_sets() const;
    
private:
    int atomic_number_;
    std::string symbol_;
    double valence_electrons_;
    double mass_;
    
    std::map<std::string, std::unique_ptr<BasisSet>> basis_sets_;
    std::string default_basis_name_;
};

} // namespace openmx
} // namespace deepx
