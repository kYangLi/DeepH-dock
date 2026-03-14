#include "bessel.hpp"
#include <cmath>
#include <algorithm>

namespace deepx {
namespace openmx {

double SphericalBessel::compute(int l, double x) {
    if (x < 1e-15) {
        return handle_small_x(l, x);
    }
    
    if (l == 0) {
        return std::sin(x) / x;
    }
    
    if (l == 1) {
        return std::sin(x) / (x * x) - std::cos(x) / x;
    }
    
    std::vector<double> jl(l + 1);
    forward_recursion(x, l, jl);
    return jl[l];
}

Eigen::VectorXd SphericalBessel::compute_array(int l, const Eigen::VectorXd& x_array) {
    Eigen::VectorXd result(x_array.size());
    
    for (int i = 0; i < x_array.size(); ++i) {
        result[i] = compute(l, x_array[i]);
    }
    
    return result;
}

Eigen::MatrixXd SphericalBessel::compute_batch(int lmax, const Eigen::VectorXd& x_array) {
    const int N = x_array.size();
    Eigen::MatrixXd jl(lmax + 1, N);
    
    for (int i = 0; i < N; ++i) {
        double x = x_array[i];
        
        if (x < 1e-15) {
            for (int l_idx = 0; l_idx <= lmax; ++l_idx) {
                jl(l_idx, i) = handle_small_x(l_idx, x);
            }
        } else {
            std::vector<double> jl_vec(lmax + 1);
            forward_recursion(x, lmax, jl_vec);
            for (int l_idx = 0; l_idx <= lmax; ++l_idx) {
                jl(l_idx, i) = jl_vec[l_idx];
            }
        }
    }
    
    return jl;
}

double SphericalBessel::compute_derivative(int l, double x) {
    if (x < 1e-15) {
        if (l == 0) {
            return 0.0;
        } else if (l == 1) {
            return 1.0 / 3.0;
        } else {
            return 0.0;
        }
    }
    
    if (l == 0) {
        return -compute(1, x);
    }
    
    double jl = compute(l, x);
    double jl_prev = compute(l - 1, x);
    
    return jl_prev - (l + 1) / x * jl;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> 
SphericalBessel::compute_batch_with_derivative(int lmax, const Eigen::VectorXd& x_array) {
    const int N = x_array.size();
    Eigen::MatrixXd jl(lmax + 1, N);
    Eigen::MatrixXd jlp(lmax + 1, N);
    
    for (int i = 0; i < N; ++i) {
        double x = x_array[i];
        
        if (x < 1e-15) {
            for (int l_idx = 0; l_idx <= lmax; ++l_idx) {
                jl(l_idx, i) = handle_small_x(l_idx, x);
                if (l_idx == 0) {
                    jlp(l_idx, i) = 0.0;
                } else if (l_idx == 1) {
                    jlp(l_idx, i) = 1.0 / 3.0;
                } else {
                    jlp(l_idx, i) = 0.0;
                }
            }
        } else {
            std::vector<double> jl_vec(lmax + 1);
            forward_recursion(x, lmax, jl_vec);
            
            for (int l_idx = 0; l_idx <= lmax; ++l_idx) {
                jl(l_idx, i) = jl_vec[l_idx];
            }
            
            jlp(0, i) = -jl_vec[1];
            for (int l_idx = 1; l_idx <= lmax; ++l_idx) {
                jlp(l_idx, i) = jl_vec[l_idx - 1] - (l_idx + 1) / x * jl_vec[l_idx];
            }
        }
    }
    
    return {jl, jlp};
}

void SphericalBessel::forward_recursion(double x, int lmax, std::vector<double>& jl) {
    jl.resize(lmax + 1);
    
    jl[0] = std::sin(x) / x;
    
    if (lmax >= 1) {
        jl[1] = std::sin(x) / (x * x) - std::cos(x) / x;
    }
    
    for (int l = 1; l < lmax; ++l) {
        jl[l + 1] = (2.0 * l + 1.0) / x * jl[l] - jl[l - 1];
    }
}

double SphericalBessel::handle_small_x(int l, double x) {
    if (l == 0) {
        return 1.0 - x * x / 6.0;
    } else if (l == 1) {
        return x / 3.0;
    } else {
        return 0.0;
    }
}

} // namespace openmx
} // namespace deepx
