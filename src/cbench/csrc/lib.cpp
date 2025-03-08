// #include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

namespace py = pybind11;

// #include "zstd_wrapper.hpp"

// torch::Tensor test_sigmoid(torch::Tensor z) {
//   return torch::sigmoid(z);
// }

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
PYBIND11_MODULE(clib, m) {
  // m.def("test_sigmoid", &test_sigmoid, "test_sigmoid");

  // m.def("FSECompress", &FSECompress, "FSECompress");
  // m.def("FSEDecompress", &FSEDecompress, "FSEDecompress");
}
