#include "ans_interface.hpp"

std::vector<ssize_t> ANSBase::create_ar_ptr_offsets(py::array_t<NUMPY_ARRAY_TYPE> indexes, std::vector<std::vector<NUMPY_ARRAY_TYPE>> ar_offsets) {
  // initialize ar offsets as ptr offset
  // if (ar_offsets.ndim() != 2 || ar_offsets.shape(1) > indexes.ndim()-1) {
  //   throw pybind11::value_error("ar_offset should be 2-dimensional with shape (*, <=data_dims)");
  // }
  // if (uint32_pow(max_symbol_value, ar_offsets.shape(0)) != _ctables.shape(1)) {
  //   throw pybind11::value_error("ctables size incorrect for a lookup table!");
  // }

  std::vector<ssize_t> ar_ptr_offsets(ar_offsets.size());
  for (size_t i = 0; i < ar_offsets.size(); ++i) {
    // std::vector<ssize_t> cur_offsets(ar_offsets.shape(1));
    ssize_t ar_offset = (indexes.shape(0) - 1) * indexes.strides(0) / sizeof(NUMPY_ARRAY_TYPE);
    for (size_t j = 0; j < indexes.ndim()-1; ++j) {
      const NUMPY_ARRAY_TYPE cur_offset = j < ar_offsets[i].size() ? ar_offsets[i][j] : 0;
      
      if (cur_offset > 0) {
        throw pybind11::value_error("ar_offset should be non-positive!");
      }
      // NOTE: indexes has a batch dim, so j+1
      ar_offset += (indexes.shape(j+1) - 1 + cur_offset) * indexes.strides(j+1) / sizeof(NUMPY_ARRAY_TYPE);
      // printf("indexes.shape(j+1)=%u, ar_offset=%i, indexes.strides(j+1)=%i\n", indexes.shape(j+1), cur_offset, indexes.strides(j+1));
    }
    ar_ptr_offsets[i] = indexes.size() - 1 - ar_offset;
    // printf("indexes.size()=%u, ar_offset=%i\n", indexes.size(), ar_offset);
  }

  return ar_ptr_offsets;

}

std::vector<std::vector<ssize_t>> ANSBase::create_ar_ptrs(py::array_t<NUMPY_ARRAY_TYPE> indexes, std::vector<std::vector<NUMPY_ARRAY_TYPE>> ar_offsets) {
  std::vector<std::vector<ssize_t>> ar_ptrs_all(ar_offsets.size());
  for (size_t i = 0; i < ar_offsets.size(); ++i) {
    std::vector<std::tuple<ssize_t, ssize_t, ssize_t>> ar_offset_stride;

    ssize_t ar_offset = (indexes.shape(0) - 1) * indexes.strides(0) / sizeof(NUMPY_ARRAY_TYPE);
    for (size_t j = 0; j < indexes.ndim()-1; ++j) {

      const NUMPY_ARRAY_TYPE cur_offset = j < ar_offsets[i].size() ? ar_offsets[i][j] : 0;
      if (cur_offset > 0) {
        throw pybind11::value_error("ar_offset should be non-positive!");
      }
      if (cur_offset < 0) {
        ar_offset_stride.emplace_back(-cur_offset, indexes.strides(j) / sizeof(NUMPY_ARRAY_TYPE), indexes.strides(j+1) / sizeof(NUMPY_ARRAY_TYPE));
      }
      // NOTE: indexes has a batch dim, so j+1
      ar_offset += (indexes.shape(j+1) - 1 + cur_offset) * indexes.strides(j+1) / sizeof(NUMPY_ARRAY_TYPE);
      // printf("indexes.shape(j+1)=%u, ar_offset=%i, indexes.strides(j+1)=%i\n", indexes.shape(j+1), cur_offset, indexes.strides(j+1));
    }
    ar_offset = indexes.size() - 1 - ar_offset;
    // printf("indexes.size()=%u, ar_offset=%i\n", indexes.size(), ar_offset);

    auto& ar_ptrs = ar_ptrs_all[i];
    ar_ptrs.resize(indexes.size());
    for (size_t k = 0; k < indexes.size(); ++k) {
      if (std::accumulate(ar_offset_stride.begin(), ar_offset_stride.end(), true,
        [=](bool result, auto offset_stride){
          return result && (k % std::get<2>(offset_stride) >= std::get<1>(offset_stride));
          })) {
        ar_ptrs[k] = k - ar_offset;
      }
      else {
        ar_ptrs[k] = -1;
      }
    }
  }

  return ar_ptrs_all;

}

void ANSBase::init_ar_params(const py::array_t<NUMPY_ARRAY_TYPE> &ar_tables,
                           const py::array_t<NUMPY_ARRAY_TYPE> &ar_offsets){

  
  const auto ndim = ar_tables.ndim();
  const auto norder = ndim - 2;
  
  if (ar_offsets.ndim() != 3 || ar_offsets.shape(1) != norder || ar_offsets.shape(0) != ar_tables.shape(0)) {
    throw pybind11::value_error("ar_offset should be 3-dimensional with shape (ar_tables_size, ar_order, <=data_dims)");
  }

  if (norder <= 0) {
    throw pybind11::value_error("ar_tables should be at least 3-dimensional with shape (ar_tables_size, index_dim, *ar_order_dims)");
  }

  _ar_offsets.resize(ar_offsets.shape(0));
  for (ssize_t i = 0; i < ar_offsets.shape(0); ++i) {
      _ar_offsets[i].resize(ar_offsets.shape(1));
      for (ssize_t j = 0; j < ar_offsets.shape(1); ++j) {
        _ar_offsets[i][j].resize(ar_offsets.shape(2));
        for (ssize_t k = 0; k < ar_offsets.shape(2); ++k) {
          _ar_offsets[i][j][k] = ar_offsets.at(i, j, k);
        }
      }
  }


  switch (norder)
  {
  case 1:
    printf("Initializing 3d ar tables\n");
    _ar_tables_3d.resize(ar_tables.shape(0));
    for (ssize_t i = 0; i < ar_tables.shape(0); ++i) {
      _ar_tables_3d[i].resize(ar_tables.shape(1));
      for (ssize_t j = 0; j < ar_tables.shape(1); ++j) {
        _ar_tables_3d[i][j].resize(ar_tables.shape(2));
        for (ssize_t k = 0; k < ar_tables.shape(2); ++k) {
          _ar_tables_3d[i][j][k] = ar_tables.at(i, j, k);
        }
      }
    }
    break;
  case 2:
    printf("Initializing 4d ar tables\n");
    _ar_tables_4d.resize(ar_tables.shape(0));
    for (ssize_t i = 0; i < ar_tables.shape(0); ++i) {
      _ar_tables_4d[i].resize(ar_tables.shape(1));
      for (ssize_t j = 0; j < ar_tables.shape(1); ++j) {
        _ar_tables_4d[i][j].resize(ar_tables.shape(2));
        for (ssize_t k = 0; k < ar_tables.shape(2); ++k) {
          _ar_tables_4d[i][j][k].resize(ar_tables.shape(3));
          for (ssize_t l = 0; l < ar_tables.shape(3); ++l) {
            _ar_tables_4d[i][j][k][l] = ar_tables.at(i, j, k, l);
          }
        }
      }
    }
    break;
  default:
    throw py::value_error("Too many dimensions!");
  }

  _is_ar_initialized=true; 
}