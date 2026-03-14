#pragma once

#include <vector>
#include <cmath>
#include <Eigen/Dense>

namespace deepx {
namespace openmx {

/**
 * @brief Spherical Bessel function calculator
 * 
 * Computes spherical Bessel functions j_l(x) using forward recursion.
 * 
 * j_0(x) = sin(x)/x
 * j_1(x) = sin(x)/x^2 - cos(x)/x
 * j_{l+1}(x) = (2l+1)/x * j_l(x) - j_{l-1}(x)
 */
class SphericalBessel {
public:
    /**
     * @brief Compute j_l(x) for a single value
     * @param l Order of the Bessel function
     * @param x Argument
     * @return j_l(x)
     */
    static double compute(int l, double x);
    
    /**
     * @brief Compute j_l(x) for an array of x values
     * @param l Order
     * @param x_array Array of arguments, shape (N,)
     * @return j_l(x) for each x, shape (N,)
     */
    static Eigen::VectorXd compute_array(int l, const Eigen::VectorXd& x_array);
    
    /**
     * @brief Compute j_l(x) for l = 0, 1, ..., lmax
     * @param lmax Maximum order
     * @param x_array Array of arguments, shape (N,)
     * @return jl[l][i] = j_l(x_array[i]), shape (lmax+1, N)
     */
    static Eigen::MatrixXd compute_batch(int lmax, const Eigen::VectorXd& x_array);
    
    /**
     * @brief Compute j_l'(x) (derivative)
     * @param l Order
     * @param x Argument
     * @return j_l'(x)
     */
    static double compute_derivative(int l, double x);
    
    /**
     * @brief Compute both j_l(x) and j_l'(x) for all l
     * @param lmax Maximum order
     * @param x_array Array of arguments
     * @return {jl, jlp}, each shape (lmax+1, N)
     */
    static std::pair<Eigen::MatrixXd, Eigen::MatrixXd> 
    compute_batch_with_derivative(int lmax, const Eigen::VectorXd& x_array);

private:
    /**
     * @brief Forward recursion for j_l(x)
     */
    static void forward_recursion(double x, int lmax, std::vector<double>& jl);
    
    /**
     * @brief Handle small x values (near singularity)
     */
    static double handle_small_x(int l, double x);
};

} // namespace openmx
} // namespace deepx
