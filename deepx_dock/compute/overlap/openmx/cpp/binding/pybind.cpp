#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "bessel.hpp"
#include "gaunt.hpp"
#include "basis.hpp"

namespace py = pybind11;
using namespace deepx::openmx;

PYBIND11_MODULE(overlap_openmx, m) {
    m.doc() = "OpenMX-style overlap matrix calculation";
    
    py::enum_<GridType>(m, "GridType")
        .value("LOG", GridType::LOG)
        .value("LINEAR", GridType::LINEAR);
    
    py::class_<RadialGrid>(m, "RadialGrid")
        .def(py::init<>())
        .def_readwrite("grid_type", &RadialGrid::grid_type)
        .def_readwrite("num_points", &RadialGrid::num_points)
        .def_readwrite("x", &RadialGrid::x)
        .def_readwrite("r", &RadialGrid::r)
        .def_readwrite("dr", &RadialGrid::dr)
        .def_static("load_h5", &RadialGrid::load_h5);
    
    py::class_<BasisMetadata>(m, "BasisMetadata")
        .def(py::init<>())
        .def_readwrite("radial_cutoff", &BasisMetadata::radial_cutoff)
        .def_readwrite("lmax", &BasisMetadata::lmax)
        .def_readwrite("num_mu", &BasisMetadata::num_mu)
        .def_readwrite("grid_type", &BasisMetadata::grid_type)
        .def_readwrite("grid_num", &BasisMetadata::grid_num)
        .def_readwrite("eigenvalues", &BasisMetadata::eigenvalues)
        .def_static("load_h5", &BasisMetadata::load_h5);
    
    py::class_<KSpaceData>(m, "KSpaceData")
        .def(py::init<>())
        .def_readwrite("k_grid", &KSpaceData::k_grid)
        .def_readwrite("wf", &KSpaceData::wf)
        .def_readwrite("k_max", &KSpaceData::k_max)
        .def_readwrite("num_k", &KSpaceData::num_k)
        .def_readwrite("lmax", &KSpaceData::lmax)
        .def_readwrite("num_mu", &KSpaceData::num_mu)
        .def("get_wf", &KSpaceData::get_wf)
        .def_static("load_h5", &KSpaceData::load_h5);
    
    py::class_<BasisSet, std::shared_ptr<BasisSet>>(m, "BasisSet")
        .def(py::init<const std::string&>(), py::arg("h5_filepath"))
        .def("get_radial_wf", &BasisSet::get_radial_wf, 
             py::arg("L"), py::arg("mu"))
        .def("get_k_space", &BasisSet::get_k_space,
             py::arg("k_max") = 20.0, py::arg("num_k") = 500,
             py::return_value_policy::reference_internal)
        .def("save_k_space_h5", &BasisSet::save_k_space_h5, py::arg("filepath"))
        .def_property_readonly("name", &BasisSet::name)
        .def_property_readonly("metadata", &BasisSet::metadata,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("radial_grid", &BasisSet::radial_grid,
                               py::return_value_policy::reference_internal);
    
    py::class_<ElementBasis>(m, "ElementBasis")
        .def(py::init<const std::string&>(), py::arg("h5_filepath"))
        .def("get_basis_set", &ElementBasis::get_basis_set, py::arg("name"),
             py::return_value_policy::reference_internal)
        .def("get_default_basis_set", &ElementBasis::get_default_basis_set,
             py::return_value_policy::reference_internal)
        .def("list_basis_sets", &ElementBasis::list_basis_sets)
        .def_property_readonly("atomic_number", &ElementBasis::atomic_number)
        .def_property_readonly("symbol", &ElementBasis::symbol);
    
    py::class_<SphericalBessel>(m, "SphericalBessel")
        .def_static("compute", &SphericalBessel::compute,
                    py::arg("l"), py::arg("x"))
        .def_static("compute_array", &SphericalBessel::compute_array,
                    py::arg("l"), py::arg("x_array"))
        .def_static("compute_batch", &SphericalBessel::compute_batch,
                    py::arg("lmax"), py::arg("x_array"))
        .def_static("compute_derivative", &SphericalBessel::compute_derivative,
                    py::arg("l"), py::arg("x"))
        .def_static("compute_batch_with_derivative", 
                    &SphericalBessel::compute_batch_with_derivative,
                    py::arg("lmax"), py::arg("x_array"));
    
    py::class_<GauntCoefficients>(m, "GauntCoefficients")
        .def(py::init<int>(), py::arg("lmax") = 6)
        .def("get", &GauntCoefficients::get,
             py::arg("l1"), py::arg("m1"), py::arg("l2"), py::arg("m2"),
             py::arg("l"), py::arg("m"))
        .def("get_all", &GauntCoefficients::get_all,
             py::arg("l1"), py::arg("m1"), py::arg("l2"), py::arg("m2"))
        .def_property_readonly("lmax", &GauntCoefficients::lmax);
    
    py::class_<SphericalHarmonics>(m, "SphericalHarmonics")
        .def_static("compute", &SphericalHarmonics::compute,
                    py::arg("l"), py::arg("m"), py::arg("theta"), py::arg("phi"))
        .def_static("compute_real", &SphericalHarmonics::compute_real,
                    py::arg("l"), py::arg("m"), py::arg("theta"), py::arg("phi"))
        .def_static("compute_with_derivatives", 
                    &SphericalHarmonics::compute_with_derivatives,
                    py::arg("l"), py::arg("m"), py::arg("theta"), py::arg("phi"));
    
    m.def("spherical_bessel", &SphericalBessel::compute,
          "Compute spherical Bessel function j_l(x)",
          py::arg("l"), py::arg("x"));
    
    m.def("spherical_bessel_array", &SphericalBessel::compute_array,
          "Compute spherical Bessel function j_l(x) for an array",
          py::arg("l"), py::arg("x_array"));
    
    m.def("spherical_bessel_batch", &SphericalBessel::compute_batch,
          "Compute spherical Bessel functions j_l(x) for l=0..lmax",
          py::arg("lmax"), py::arg("x_array"));
    
    m.def("gaunt_coefficient",
          [](int l1, int m1, int l2, int m2, int l, int m, int lmax) {
              GauntCoefficients gaunt(lmax);
              return gaunt.get(l1, m1, l2, m2, l, m);
          },
          "Compute Gaunt coefficient",
          py::arg("l1"), py::arg("m1"), py::arg("l2"), py::arg("m2"),
          py::arg("l"), py::arg("m"), py::arg("lmax") = 6);
}
