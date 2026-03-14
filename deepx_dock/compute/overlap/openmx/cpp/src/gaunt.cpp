#include "gaunt.hpp"
#include <cmath>
#include <map>

namespace deepx {
namespace openmx {

static std::map<int, double> factorial_cache = {{0, 1.0}, {1, 1.0}};

double GauntCoefficients::factorial(int n) {
    if (n < 0) return 0.0;
    
    auto it = factorial_cache.find(n);
    if (it != factorial_cache.end()) {
        return it->second;
    }
    
    double result = n * factorial(n - 1);
    factorial_cache[n] = result;
    return result;
}

GauntCoefficients::GauntCoefficients(int lmax) : lmax_(lmax) {
    precompute();
}

double GauntCoefficients::get(int l1, int m1, int l2, int m2, int l, int m) const {
    if (!selection_rule(l1, m1, l2, m2, l, m)) {
        return 0.0;
    }
    
    auto key = std::make_tuple(l1, m1, l2, m2, l, m);
    auto it = cache_.find(key);
    
    if (it != cache_.end()) {
        return it->second;
    }
    
    return 0.0;
}

std::vector<std::tuple<int, int, double>> 
GauntCoefficients::get_all(int l1, int m1, int l2, int m2) const {
    std::vector<std::tuple<int, int, double>> result;
    
    int lmin = std::abs(l1 - l2);
    int lmax_val = l1 + l2;
    
    for (int l = lmin; l <= lmax_val; ++l) {
        int m = m1 + m2;
        
        if (std::abs(m) > l) continue;
        
        double gaunt = get(l1, m1, l2, m2, l, m);
        
        if (std::abs(gaunt) > 1e-15) {
            result.push_back(std::make_tuple(l, m, gaunt));
        }
    }
    
    return result;
}

bool GauntCoefficients::selection_rule(int l1, int m1, int l2, int m2, int l, int m) {
    if (m != m1 + m2) return false;
    
    if (l < std::abs(l1 - l2) || l > l1 + l2) return false;
    
    if (std::abs(m1) > l1 || std::abs(m2) > l2 || std::abs(m) > l) return false;
    
    return true;
}

double GauntCoefficients::wigner_3j(int j1, int j2, int j3, int m1, int m2, int m3) {
    if (m1 + m2 + m3 != 0) return 0.0;
    
    if (j3 < std::abs(j1 - j2) || j3 > j1 + j2) return 0.0;
    
    if ((j1 + j2 + j3) % 2 != 0) return 0.0;
    
    int t1 = j2 - j1 - m3;
    int t2 = j1 - j2 - m3;
    int t3 = -j1 + j2 - m3;
    
    int tmin = std::max({0, t1, t2});
    int tmax = std::min({j3 - j1 + m2, j3 - j2 - m1, j1 + j2 - j3});
    
    if (tmin > tmax) return 0.0;
    
    double prefactor = std::pow(-1.0, tmin);
    
    double tri_norm = factorial(j1 + j2 - j3) * factorial(j1 - j2 + j3) * 
                      factorial(-j1 + j2 + j3) / factorial(j1 + j2 + j3 + 1);
    prefactor *= std::sqrt(tri_norm);
    
    double m_norm = factorial(j1 - m1) * factorial(j1 + m1) *
                    factorial(j2 - m2) * factorial(j2 + m2) *
                    factorial(j3 - m3) * factorial(j3 + m3);
    prefactor *= std::sqrt(m_norm);
    
    double sum_val = 0.0;
    for (int t = tmin; t <= tmax; ++t) {
        double num = std::pow(-1.0, t);
        double denom = factorial(t) * factorial(j3 - j1 + m2 - t) * 
                       factorial(j3 - j2 - m1 - t) * factorial(j1 + j2 - j3 - t) *
                       factorial(j1 - m1 - t) * factorial(j2 + m2 - t);
        sum_val += num / denom;
    }
    
    return prefactor * sum_val;
}

double GauntCoefficients::clebsch_gordan(int j1, int m1, int j2, int m2, int j, int m) {
    if (m1 + m2 != m) return 0.0;
    
    double phase = std::pow(-1.0, j1 - j2 + m);
    double w3j = wigner_3j(2*j1, 2*j2, 2*j, 2*m1, 2*m2, -2*m);
    
    return phase * std::sqrt(2.0 * j + 1.0) * w3j;
}

void GauntCoefficients::precompute() {
    for (int l1 = 0; l1 <= lmax_; ++l1) {
        for (int m1 = -l1; m1 <= l1; ++m1) {
            for (int l2 = 0; l2 <= lmax_; ++l2) {
                for (int m2 = -l2; m2 <= l2; ++m2) {
                    int lmin = std::abs(l1 - l2);
                    int lmax_val = l1 + l2;
                    
                    for (int l = lmin; l <= lmax_val; ++l) {
                        if (l > lmax_) continue;
                        
                        int m = m1 + m2;
                        if (std::abs(m) > l) continue;
                        
                        double cg1 = clebsch_gordan(l1, 0, l2, 0, l, 0);
                        double cg2 = clebsch_gordan(l1, m1, l2, m2, l, m);
                        
                        double prefactor = std::sqrt((2.0 * l1 + 1.0) * (2.0 * l2 + 1.0) / 
                                                     (4.0 * M_PI * (2.0 * l + 1.0)));
                        
                        double gaunt = prefactor * cg1 * cg2;
                        
                        auto key = std::make_tuple(l1, m1, l2, m2, l, m);
                        cache_[key] = gaunt;
                    }
                }
            }
        }
    }
}

std::complex<double> SphericalHarmonics::compute(int l, int m, double theta, double phi) {
    int abs_m = std::abs(m);
    double norm = normalization_factor(l, abs_m);
    double P = associated_legendre(l, abs_m, std::cos(theta));
    
    std::complex<double> result;
    if (m >= 0) {
        result = norm * P * std::exp(std::complex<double>(0, m * phi));
    } else {
        result = norm * P * std::exp(std::complex<double>(0, m * phi)) * 
                 std::pow(-1.0, abs_m);
    }
    
    return result;
}

double SphericalHarmonics::compute_real(int l, int m, double theta, double phi) {
    if (m == 0) {
        return compute(l, 0, theta, phi).real();
    }
    
    std::complex<double> Y_m = compute(l, std::abs(m), theta, phi);
    std::complex<double> Y_minus_m = compute(l, -std::abs(m), theta, phi);
    
    if (m > 0) {
        return (Y_m + std::pow(-1.0, m) * Y_minus_m).real() / std::sqrt(2.0);
    } else {
        return std::complex<double>(0, 1) * (Y_m - std::pow(-1.0, m) * Y_minus_m).real() / 
               std::sqrt(2.0);
    }
}

std::tuple<std::complex<double>, std::complex<double>, std::complex<double>>
SphericalHarmonics::compute_with_derivatives(int l, int m, double theta, double phi) {
    const double eps = 1e-8;
    
    std::complex<double> Y = compute(l, m, theta, phi);
    
    std::complex<double> Y_plus = compute(l, m, theta + eps, phi);
    std::complex<double> Y_minus = compute(l, m, theta - eps, phi);
    std::complex<double> dY_dtheta = (Y_plus - Y_minus) / (2.0 * eps);
    
    std::complex<double> Y_phi_plus = compute(l, m, theta, phi + eps);
    std::complex<double> Y_phi_minus = compute(l, m, theta, phi - eps);
    std::complex<double> dY_dphi = (Y_phi_plus - Y_phi_minus) / (2.0 * eps);
    
    return {Y, dY_dtheta, dY_dphi};
}

double SphericalHarmonics::associated_legendre(int l, int m, double cos_theta) {
    if (m < 0 || m > l) return 0.0;
    
    double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
    
    double P_mm = 1.0;
    if (m > 0) {
        double factor = 1.0;
        for (int i = 1; i <= m; ++i) {
            factor *= (2.0 * i - 1.0) * sin_theta;
        }
        P_mm = factor;
    }
    
    if (l == m) {
        return P_mm;
    }
    
    double P_mmp1 = cos_theta * (2.0 * m + 1.0) * P_mm;
    
    if (l == m + 1) {
        return P_mmp1;
    }
    
    double P_ll = 0.0;
    for (int ll = m + 2; ll <= l; ++ll) {
        P_ll = (cos_theta * (2.0 * ll - 1.0) * P_mmp1 - (ll + m - 1.0) * P_mm) / 
               (ll - m);
        P_mm = P_mmp1;
        P_mmp1 = P_ll;
    }
    
    return P_ll;
}

double SphericalHarmonics::normalization_factor(int l, int m) {
    double norm = std::sqrt((2.0 * l + 1.0) / (4.0 * M_PI) * 
                            factorial(l - m) / factorial(l + m));
    return norm;
}

} // namespace openmx
} // namespace deepx
