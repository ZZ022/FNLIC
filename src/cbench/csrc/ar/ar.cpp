#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

namespace py = pybind11;

#include "ar_funcs.hpp"

std::vector<ssize_t> create_ar_ptr_offsets(py::array_t<uint32_t> indexes, py::array_t<uint32_t> ar_offsets) {
    // initialize ar offsets as ptr offset
  if (ar_offsets.ndim() != 2 || ar_offsets.shape(1) > indexes.ndim()-1) {
    throw pybind11::value_error("ar_offset should be 2-dimensional with shape (*, <=data_dims)");
  }
  // if (uint32_pow(max_symbol_value, ar_offsets.shape(0)) != _ctables.shape(1)) {
  //   throw pybind11::value_error("ctables size incorrect for a lookup table!");
  // }

  std::vector<ssize_t> ar_ptr_offsets(ar_offsets.shape(0));
  for (ssize_t i = 0; i < ar_offsets.shape(0); ++i) {
    // std::vector<ssize_t> cur_offsets(ar_offsets.shape(1));
    ssize_t ar_offset = (indexes.shape(0) - 1) * indexes.strides(0) / sizeof(int32_t);
    for (ssize_t j = 0; j < indexes.ndim()-1; ++j) {
      const int32_t cur_offset = j < ar_offsets.shape(1) ? ar_offsets.at(i, j) : 0;
      
      if (cur_offset > 0) {
        throw pybind11::value_error("ar_offset should be non-positive!");
      }
      // NOTE: indexes has a batch dim, so j+1
      ar_offset += (indexes.shape(j+1) - 1 + cur_offset) * indexes.strides(j+1) / sizeof(int32_t);
      // printf("indexes.shape(j+1)=%u, ar_offset=%i, indexes.strides(j+1)=%i\n", indexes.shape(j+1), cur_offset, indexes.strides(j+1));
    }
    ar_ptr_offsets[i] = indexes.size() - 1 - ar_offset;
    // printf("indexes.size()=%u, ar_offset=%i\n", indexes.size(), ar_offset);
  }

  return ar_ptr_offsets;

}

template <typename T>
py::array_t<T> autoregressive_transform_3way_static(
  py::array_t<T> input, 
  // std::vector<uint32_t> ar_offset0, std::vector<uint32_t> ar_offset1, std::vector<uint32_t> ar_offset2
  py::array_t<uint32_t> ar_offsets
){
  std::vector<ssize_t> ar_ptr_offsets = create_ar_ptr_offsets(input, ar_offsets);
  assert(ar_ptr_offsets.size() == 3);

  const T* input_ptr = input.data();
  py::array_t<T> output(input.request(true));
  T* output_ptr = output.mutable_data();

  for (ssize_t i=0; i<input.size(); i++) {
#define GET_AR_VALUE(off) ((i >= off) ? input_ptr[i - off] : 0)
    auto pred = (GET_AR_VALUE(ar_ptr_offsets[0]) + GET_AR_VALUE(ar_ptr_offsets[1]) + GET_AR_VALUE(ar_ptr_offsets[2])) / 3;
    output_ptr[i] = input_ptr[i] - pred;
  }

  return output;
}

template <typename T>
py::array_t<T> autoregressive_transform_3way(
  py::array_t<T> input, std::function<T(T,T,T)> ar_func,
  // std::vector<uint32_t> ar_offset0, std::vector<uint32_t> ar_offset1, std::vector<uint32_t> ar_offset2
  py::array_t<uint32_t> ar_offsets
){
  std::vector<ssize_t> ar_ptr_offsets = create_ar_ptr_offsets(input, ar_offsets);
  assert(ar_ptr_offsets.size() == 3);

  const T* input_ptr = input.data();
  py::array_t<T> output(input.request(true));
  T* output_ptr = output.mutable_data();

  for (ssize_t i=0; i<input.size(); i++) {
#define GET_AR_VALUE(off) ((i >= off) ? input_ptr[i - off] : 0)
    auto pred = ar_func(GET_AR_VALUE(ar_ptr_offsets[0]), GET_AR_VALUE(ar_ptr_offsets[1]), GET_AR_VALUE(ar_ptr_offsets[2]));
    output_ptr[i] = input_ptr[i] - pred;
  }

  return output;
}

template <typename T>
py::array_t<T> autoregressive_transform_3way_staticfunc(
  py::array_t<T> input, 
  // std::vector<uint32_t> ar_offset0, std::vector<uint32_t> ar_offset1, std::vector<uint32_t> ar_offset2
  py::array_t<uint32_t> ar_offsets
){
  return autoregressive_transform_3way<T>(input, &ar_3way_mean<T>, ar_offsets);
}

template <typename T, unsigned int ways>
py::array_t<T> autoregressive_transform_fixways(
  py::array_t<T> input, std::function<T(std::array<T, ways>)> ar_func,
  // std::vector<uint32_t> ar_offset0, std::vector<uint32_t> ar_offset1, std::vector<uint32_t> ar_offset2
  py::array_t<uint32_t> ar_offsets
){
  std::vector<ssize_t> ar_ptr_offsets = create_ar_ptr_offsets(input, ar_offsets);
  assert(ar_ptr_offsets.size() == ways);

  const T* input_ptr = input.data();
  py::array_t<T> output(input.request(true));
  T* output_ptr = output.mutable_data();

  for (ssize_t i=0; i<input.size(); i++) {
#define GET_AR_VALUE(off) ((i >= off) ? input_ptr[i - off] : 0)
    
    std::array<T, ways> ar_values;
    for (ssize_t ar_idx=0; ar_idx<ar_ptr_offsets.size(); ar_idx++)
    {
      ar_values[ar_idx] = GET_AR_VALUE(ar_ptr_offsets[ar_idx]);
    }
    auto pred = ar_func(ar_values);
    output_ptr[i] = input_ptr[i] - pred;
  }

  return output;
}

template <typename T, unsigned int ways>
py::array_t<T> autoregressive_transform_fixways_op(
  py::array_t<T> input, const ar_op<T>& arop,
  // std::vector<uint32_t> ar_offset0, std::vector<uint32_t> ar_offset1, std::vector<uint32_t> ar_offset2
  py::array_t<uint32_t> ar_offsets
){
  std::vector<ssize_t> ar_ptr_offsets = create_ar_ptr_offsets(input, ar_offsets);
  assert(ar_ptr_offsets.size() == ways);

  const T* input_ptr = input.data();
  py::array_t<T> output(input.request(true));
  T* output_ptr = output.mutable_data();

  for (ssize_t i=0; i<input.size(); i++) {
#define GET_AR_VALUE(off) ((i >= off) ? input_ptr[i - off] : 0)
    
    std::array<T, ways> ar_values;
    for (ssize_t ar_idx=0; ar_idx<ar_ptr_offsets.size(); ar_idx++)
    {
      ar_values[ar_idx] = GET_AR_VALUE(ar_ptr_offsets[ar_idx]);
    }
    auto pred = arop(ar_values.begin(), ar_values.end());
    output_ptr[i] = input_ptr[i] - pred;
  }

  return output;
}

template <typename T>
py::array_t<T> autoregressive_transform(
  py::array_t<T> input, std::function<T(std::vector<T>)> ar_func,
  py::array_t<uint32_t> ar_offsets, T default_input_bias
){
  std::vector<ssize_t> ar_ptr_offsets = create_ar_ptr_offsets(input, ar_offsets);

  const T* input_ptr = input.data();
  py::array_t<T> output(input.request(true));
  T* output_ptr = output.mutable_data();

  for (ssize_t i=0; i<input.size(); i++) {
#define GET_AR_VALUE_DEFAULT(off) ((i >= off) ? (input_ptr[i - off] + default_input_bias) : 0)
    std::vector<T> ar_values(ar_ptr_offsets.size());
    for (ssize_t ar_idx=0; ar_idx<ar_ptr_offsets.size(); ar_idx++)
    {
      ar_values[ar_idx] = GET_AR_VALUE_DEFAULT(ar_ptr_offsets[ar_idx]);
    }
    auto pred = ar_func(ar_values);
    output_ptr[i] = input_ptr[i] - pred;
  }

  return output;
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
PYBIND11_MODULE(ar, m) {
  // ar_funcs
  m.def("ar_3way_mean", &ar_3way_mean<float>, "ar_3way_mean");
  m.def("ar_3way_mean_array", &ar_3way_mean_array<float>, "ar_3way_mean_array");

  py::class_<ar_op<float>>(m, "ar_op_float")
    .def(py::init());
    // .def("__call__", &ar_op<float>::op_vector);
  py::class_<ar_op<int>>(m, "ar_op_int")
    .def(py::init());

  py::class_<ar_linear_op<float>, ar_op<float>>(m, "ar_linear_op")
    .def(py::init<std::vector<float>, float>())
    .def("__call__", &ar_linear_op<float>::op_vector);
  py::class_<ar_lookup_op<int>, ar_op<int>>(m, "ar_lookup_op")
    .def(py::init<std::vector<int>, std::vector<size_t>>());

  m.def("autoregressive_transform_3way_staticfunc", &autoregressive_transform_3way_staticfunc<float>, "autoregressive_transform_3way_staticfunc");
  m.def("autoregressive_transform_3way_static", &autoregressive_transform_3way_static<float>, "autoregressive_transform_3way_static");
  m.def("autoregressive_transform_3way", &autoregressive_transform_3way<float>, "autoregressive_transform_3way");
  m.def("autoregressive_transform_3way_tpl", &autoregressive_transform_fixways<float, 3>, "autoregressive_transform_3way_tpl");
  m.def("autoregressive_transform_3way_op_tpl", &autoregressive_transform_fixways_op<float, 3>, "autoregressive_transform_3way_op_tpl");
  m.def("autoregressive_transform", &autoregressive_transform<float>, "autoregressive_transform",
    py::arg("input"), py::arg("ar_func"), py::arg("ar_offsets"), py::arg("default_input_bias")=0);
}
