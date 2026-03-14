#pragma once

#include <vector>
#include <map>
#include <tuple>
#include <complex>
#include <Eigen/Dense>

namespace deepx {
namespace openmx {

/**
 * @brief Gaunt coefficients calculator
 * 
 * Computes Gaunt coefficients for angular momentum coupling:
 * 
 * C(l1, m1, l2, m2, l, m) = ∫ Y_{l1}^{m1} Y_{l2}^{m2} Y_l^{m*} dΩ
 * 
 * Using the relation to Clebsch-Gordan coefficients:
 * C = √[(2l1+1)(2l2+1)/(4π(2l+1))] × <l1,0,l2,0|l,0> × <l1,m1,l2,m2|l,m>
 */
class GauntCoefficients {
public:
    /**
     * @brief Constructor - precomputes coefficients up to lmax
     * @param lmax Maximum angular momentum
     */
    explicit GauntCoefficients(int lmax);
    
    /**
     * @brief Get a Gaunt coefficient
     * @param l1, m1 First spherical harmonic
     * @param l2, m2 Second spherical harmonic
     * @param l, m Third spherical harmonic
     * @return Gaunt coefficient value
     */
    double get(int l1, int m1, int l2, int m2, int l, int m) const;
    
    /**
     * @brief Get all valid (l, m) and their Gaunt coefficients
     * @param l1, m1 First spherical harmonic
     * @param l2, m2 Second spherical harmonic
     * @return Vector of (l, m, gaunt_value)
     */
    std::vector<std::tuple<int, int, double>> 
    get_all(int l1, int m1, int l2, int m2) const;
    
    /**
     * @brief Get maximum angular momentum
     */
    int lmax() const { return lmax_; }

private:
    int lmax_;
    
    // Cache: key = (l1, m1, l2, m2, l, m), value = coefficient
    std::map<std::tuple<int,int,int,int,int,int>, double> cache_;
    
    /**
     * @brief Compute Wigner 3j symbol
     */
    static double wigner_3j(int j1, int j2, int j3, int m1, int m2, int m3);
    
    /**
     * @brief Compute Clebsch-Gordan coefficient
     */
    static double clebsch_gordan(int j1, int m1, int j2, int m2, int j, int m);
    
    /**
     * @brief Check selection rules
     */
    static bool selection_rule(int l1, int m1, int l2, int m2, int l, int m);
    
    /**
     * @brief Precompute all coefficients up to lmax_
     */
    void precompute();
    
    /**
     * @brief Compute factorial (memoized)
     */
    static double factorial(int n);
};

/**
 * @brief Spherical harmonics calculator
 */
class SphericalHarmonics {
public:
    /**
     * @brief Compute complex spherical harmonic Y_l^m(theta, phi)
     */
    static std::complex<double> compute(int l, int m, double theta, double phi);
    
    /**
     * @brief Compute real spherical harmonic
     */
    static double compute_real(int l, int m, double theta, double phi);
    
    /**
     * @brief Compute spherical harmonic and its angular derivatives
     * @return {Y, dY_dtheta, dY_dphi}
     */
    static std::tuple<std::complex<double>, 
                      std::complex<double>, 
                      std::complex<double>>
    compute_with_derivatives(int l, int m, double theta, double phi);

private:
    /**
     * @brief Compute associated Legendre polynomial P_l^|m|(cos(theta))
     */
    static double associated_legendre(int l, int m, double cos_theta);
    
    /**
     * @brief Compute normalization factor
     */
    static double normalization_factor(int l, int m);
};

} // namespace openmx
} // namespace deepx
