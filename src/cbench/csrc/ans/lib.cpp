#include <pybind11/pybind11.h>

// rans interface
#include "ans_interface.hpp"
#include "rans64.hpp"
#include "tans.hpp"
#include "ar_funcs.hpp"


PYBIND11_MODULE(ans, m) {
    m.doc() = "Numpy based ANS entropy coding library.";

    // py::class_<ar_op<float>>(m, "ar_op_np")
    //     .def(py::init());
        // .def("__call__", &ar_op<NUMPY_ARRAY_TYPE>::op_vector);
    py::class_<ar_op<NUMPY_ARRAY_TYPE>>(m, "ar_op_default")
      .def(py::init());

    py::class_<ar_linear_op<NUMPY_ARRAY_TYPE, float>, ar_op<NUMPY_ARRAY_TYPE>>(m, "ar_linear_op")
        .def(py::init<std::vector<float>, float, float>())
        .def("__call__", &ar_linear_op<NUMPY_ARRAY_TYPE, float>::op_vector);
    py::class_<ar_limited_scaled_add_linear_op<NUMPY_ARRAY_TYPE, float>, ar_op<NUMPY_ARRAY_TYPE>>(m, "ar_limited_scaled_add_linear_op")
        .def(py::init<std::vector<float>, float, float, float, float>())
        .def("__call__", &ar_limited_scaled_add_linear_op<NUMPY_ARRAY_TYPE, float>::op_vector);
    // py::class_<ar_lookup_op<NUMPY_ARRAY_TYPE>, ar_op<NUMPY_ARRAY_TYPE>>(m, "ar_lookup_op")
    //     .def(py::init<std::vector<NUMPY_ARRAY_TYPE>, std::vector<size_t>>());

    // PYBIND11_ANS_CLASSES(m);

    PYBIND11_RANS64_CLASSES(m);

    PYBIND11_TANS_CLASSES(m);
}
