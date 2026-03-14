#include "basis.hpp"
#include "bessel.hpp"
#include <stdexcept>
#include <algorithm>

namespace deepx {
namespace openmx {

RadialGrid RadialGrid::load_h5(const H5::Group& group) {
    RadialGrid grid;
    
    std::string grid_type_str;
    group.attr("grid_type").read(grid_type_str);
    grid.grid_type = (grid_type_str == "log") ? GridType::LOG : GridType::LINEAR;
    
    int grid_num;
    group.attr("grid_num").read(grid_num);
    grid.num_points = grid_num;
    
    {
        H5::DataSet ds = group.openDataSet("x");
        H5::DataSpace space = ds.getSpace();
        hsize_t dims[1];
        space.getSimpleExtentDims(dims);
        grid.x.resize(dims[0]);
        ds.read(grid.x.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    {
        H5::DataSet ds = group.openDataSet("r");
        H5::DataSpace space = ds.getSpace();
        hsize_t dims[1];
        space.getSimpleExtentDims(dims);
        grid.r.resize(dims[0]);
        ds.read(grid.r.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    {
        H5::DataSet ds = group.openDataSet("dr");
        H5::DataSpace space = ds.getSpace();
        hsize_t dims[1];
        space.getSimpleExtentDims(dims);
        grid.dr.resize(dims[0]);
        ds.read(grid.dr.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    return grid;
}

BasisMetadata BasisMetadata::load_h5(const H5::Group& group) {
    BasisMetadata meta;
    
    group.attr("radial_cutoff").read(meta.radial_cutoff);
    
    int lmax, num_mu, grid_num;
    group.attr("lmax").read(lmax);
    group.attr("num_mu").read(num_mu);
    group.attr("grid_num").read(grid_num);
    meta.lmax = lmax;
    meta.num_mu = num_mu;
    meta.grid_num = grid_num;
    
    std::string grid_type_str;
    group.attr("grid_type").read(grid_type_str);
    meta.grid_type = (grid_type_str == "log") ? GridType::LOG : GridType::LINEAR;
    
    {
        H5::DataSet ds = group.openDataSet("eigenvalues");
        H5::DataSpace space = ds.getSpace();
        hsize_t dims[2];
        space.getSimpleExtentDims(dims);
        meta.eigenvalues.resize(dims[0], dims[1]);
        ds.read(meta.eigenvalues.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    return meta;
}

Eigen::VectorXcd KSpaceData::get_wf(int L, int mu) const {
    if (L < 0 || L > lmax || mu < 0 || mu >= num_mu) {
        throw std::out_of_range("Invalid (L, mu) for k-space wave function");
    }
    
    int N_k = k_grid.size();
    Eigen::VectorXcd result(N_k);
    
    for (int ik = 0; ik < N_k; ++ik) {
        int idx = (L * num_mu + mu) * N_k + ik;
        result[ik] = wf[idx];
    }
    
    return result;
}

KSpaceData KSpaceData::load_h5(const H5::Group& group) {
    KSpaceData ks;
    
    {
        H5::DataSet ds = group.openDataSet("k_grid");
        H5::DataSpace space = ds.getSpace();
        hsize_t dims[1];
        space.getSimpleExtentDims(dims);
        ks.k_grid.resize(dims[0]);
        ds.read(ks.k_grid.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    {
        H5::DataSet ds = group.openDataSet("wf");
        H5::DataSpace space = ds.getSpace();
        hsize_t dims[1];
        space.getSimpleExtentDims(dims);
        ks.wf.resize(dims[0]);
        ds.read(ks.wf.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    group.attr("k_max").read(ks.k_max);
    
    int num_k;
    group.attr("num_k").read(num_k);
    ks.num_k = num_k;
    
    ks.lmax = 0;
    ks.num_mu = 0;
    
    return ks;
}

BasisSet::BasisSet(const std::string& h5_filepath) {
    H5::H5File file(h5_filepath, H5F_ACC_RDONLY);
    
    file.attr("name").read(name_);
    
    H5::Group meta_group = file.openGroup("metadata");
    metadata_ = BasisMetadata::load_h5(meta_group);
    
    H5::Group grid_group = file.openGroup("radial_grid");
    radial_grid_ = RadialGrid::load_h5(grid_group);
    
    {
        H5::DataSet ds = file.openDataSet("radial_wf/data");
        H5::DataSpace space = ds.getSpace();
        hsize_t dims[3];
        space.getSimpleExtentDims(dims, nullptr);
        radial_wf_.resize(dims[0] * dims[1] * dims[2]);
        ds.read(radial_wf_.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    if (file.exists("k_space")) {
        H5::Group ks_group = file.openGroup("k_space");
        k_space_ = std::make_shared<KSpaceData>(KSpaceData::load_h5(ks_group));
    }
    
    file.close();
}

Eigen::VectorXd BasisSet::get_radial_wf(int L, int mu) const {
    if (L < 0 || L > metadata_.lmax) {
        throw std::out_of_range("L out of range");
    }
    if (mu < 0 || mu >= metadata_.num_mu) {
        throw std::out_of_range("mu out of range");
    }
    
    int N = radial_grid_.num_points;
    Eigen::VectorXd result(N);
    
    for (int i = 0; i < N; ++i) {
        int idx = (L * metadata_.num_mu + mu) * N + i;
        result[i] = radial_wf_[idx].real();
    }
    
    return result;
}

const KSpaceData& BasisSet::get_k_space(double k_max, int num_k) {
    if (k_space_ && std::abs(k_space_->k_max - k_max) < 1e-10 && k_space_->num_k == num_k) {
        return *k_space_;
    }
    
    compute_fourier_transform(k_max, num_k);
    return *k_space_;
}

void BasisSet::compute_fourier_transform(double k_max, int num_k) {
    k_space_ = std::make_shared<KSpaceData>();
    k_space_->k_max = k_max;
    k_space_->num_k = num_k;
    k_space_->lmax = metadata_.lmax;
    k_space_->num_mu = metadata_.num_mu;
    
    k_space_->k_grid = Eigen::VectorXd::LinSpaced(num_k, 0.0, k_max);
    
    int N_k = num_k;
    int total_size = (metadata_.lmax + 1) * metadata_.num_mu * N_k;
    k_space_->wf.resize(total_size);
    
    for (int L = 0; L <= metadata_.lmax; ++L) {
        for (int mu = 0; mu < metadata_.num_mu; ++mu) {
            Eigen::VectorXd R_r = get_radial_wf(L, mu);
            
            for (int ik = 0; ik < N_k; ++ik) {
                double k = k_space_->k_grid[ik];
                
                Eigen::VectorXd kr = k * radial_grid_.r;
                Eigen::VectorXd j_L = SphericalBessel::compute_array(L, kr);
                
                Eigen::VectorXd integrand = R_r.array() * j_L.array() * 
                                           radial_grid_.r.array().square();
                
                double integral = 0.0;
                for (int i = 0; i < integrand.size() - 1; ++i) {
                    integral += 0.5 * (integrand[i] + integrand[i + 1]) * radial_grid_.dr[i];
                }
                
                int idx = (L * metadata_.num_mu + mu) * N_k + ik;
                k_space_->wf[idx] = integral;
            }
        }
    }
}

void BasisSet::save_k_space_h5(const std::string& filepath) {
    if (!k_space_) {
        throw std::runtime_error("k-space data not computed");
    }
    
    H5::H5File file(filepath, H5F_ACC_TRUNC);
    
    {
        H5::DataSpace space(1, H5::DataSpace(H5::PredType::NATIVE_DOUBLE, 
                                              k_space_->k_grid.size()));
        H5::DataSet ds = file.createDataSet("k_grid", H5::PredType::NATIVE_DOUBLE, space);
        ds.write(k_space_->k_grid.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    {
        H5::DataSpace space(1, H5::DataSpace(H5::PredType::NATIVE_DOUBLE, 
                                              k_space_->wf.size()));
        H5::DataSet ds = file.createDataSet("wf", H5::PredType::NATIVE_DOUBLE, space);
        ds.write(k_space_->wf.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    file.attr("k_max").write(k_space_->k_max);
    file.attr("num_k").write(k_space_->num_k);
    
    file.close();
}

ElementBasis::ElementBasis(const std::string& h5_filepath) {
    H5::H5File file(h5_filepath, H5F_ACC_RDONLY);
    
    file.attr("atomic_number").read(atomic_number_);
    
    std::string symbol;
    file.attr("symbol").read(symbol);
    symbol_ = symbol;
    
    file.attr("valence_electrons").read(valence_electrons_);
    file.attr("mass").read(mass_);
    
    if (file.exists("basis_sets")) {
        H5::Group basis_group = file.openGroup("basis_sets");
        hsize_t num_objs = basis_group.getNumObjs();
        
        double max_cutoff = 0.0;
        
        for (hsize_t i = 0; i < num_objs; ++i) {
            std::string basis_name = basis_group.getObjnameByIdx(i);
            H5::Group bs_group = basis_group.openGroup(basis_name);
            
            BasisSet* bs = new BasisSet(h5_filepath);
            basis_sets_[basis_name] = std::unique_ptr<BasisSet>(bs);
            
            if (bs->metadata().radial_cutoff > max_cutoff) {
                max_cutoff = bs->metadata().radial_cutoff;
                default_basis_name_ = basis_name;
            }
        }
    }
    
    file.close();
}

const BasisSet& ElementBasis::get_basis_set(const std::string& name) const {
    auto it = basis_sets_.find(name);
    if (it == basis_sets_.end()) {
        throw std::out_of_range("Basis set not found: " + name);
    }
    return *it->second;
}

const BasisSet& ElementBasis::get_default_basis_set() const {
    if (default_basis_name_.empty()) {
        throw std::runtime_error("No basis sets available");
    }
    return get_basis_set(default_basis_name_);
}

std::vector<std::string> ElementBasis::list_basis_sets() const {
    std::vector<std::string> names;
    for (const auto& pair : basis_sets_) {
        names.push_back(pair.first);
    }
    return names;
}

} // namespace openmx
} // namespace deepx
