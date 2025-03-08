#pragma once
#if __cplusplus > 199711L
#define register      // Deprecated in C++11.
#endif  // #if __cplusplus > 199711L

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "ar_funcs.hpp"

#define NUMPY_ARRAY_TYPE int32_t
// TODO: maybe add those meta for stable version?
#define NUMPY_ARRAY_META py::array::c_style | py::array::forcecast

namespace py = pybind11;

// template <size_t AROrder=0>
class ANSBase {
public:
  ANSBase() = default;
  virtual ~ANSBase() = default;

  ANSBase(const ANSBase &) = delete;
  ANSBase(ANSBase &&) = delete;
  ANSBase &operator=(const ANSBase &) = delete;
  ANSBase &operator=(ANSBase &&) = delete;

  // TODO: freqs may allow float/double?
  virtual void init_params(const py::array_t<NUMPY_ARRAY_TYPE> &freqs,
                           const py::array_t<NUMPY_ARRAY_TYPE> &num_symbols,
                           const py::array_t<NUMPY_ARRAY_TYPE> &offsets) { _is_initialized=true; };

  void init_ar_params(const py::array_t<NUMPY_ARRAY_TYPE> &ar_table,
                           const py::array_t<NUMPY_ARRAY_TYPE> &ar_offsets
                           );

  // TODO: use base class ar_op for better compability! How to implement this in pybind11?
  void init_custom_ar_ops(std::vector<ar_limited_scaled_add_linear_op<NUMPY_ARRAY_TYPE, float>> arop) {
    if (arop.size() > 0) {
      _custom_ar_ops.resize(arop.size());
      // for (size_t i=0; i<arop.size(); i++ ) {
      //   _custom_ar_ops[i] = std::move(arop[i]);
      // }
      std::copy(arop.begin(), arop.end(), _custom_ar_ops.begin());
      _is_ar_initialized = true;
    }
  }

  std::vector<std::vector<ssize_t>> create_ar_ptrs(py::array_t<NUMPY_ARRAY_TYPE> indexes, std::vector<std::vector<NUMPY_ARRAY_TYPE>> ar_offsets);

protected:

  // TODO: using offset is insufficient for ar! Find another way!
  std::vector<ssize_t> create_ar_ptr_offsets(py::array_t<NUMPY_ARRAY_TYPE> indexes, std::vector<std::vector<NUMPY_ARRAY_TYPE>> ar_offsets);
  

  inline NUMPY_ARRAY_TYPE ar_update_index(std::vector<const NUMPY_ARRAY_TYPE*> ar_ptr_offsets, const NUMPY_ARRAY_TYPE ar_index, const NUMPY_ARRAY_TYPE index, const NUMPY_ARRAY_TYPE* symbols_ptr, const ssize_t current_offset)
  {
    if (_custom_ar_ops.size() > 0) {
#define GET_AR_VALUE_NORMAL(off) (off[current_offset] > 0) ? symbols_ptr[current_offset - off[current_offset]] : 0
      // if (AROrder > 0){
      //   _custom_ar_ops[ar_index].op(ar_ptr_offsets.begin(), ar_ptr_offsets.begin()+AROrder);
      // }
      // else{
      //   _custom_ar_ops[ar_index].op_vector(ar_ptr_offsets);
      // }
      
      // std::vector<NUMPY_ARRAY_TYPE> v = {GET_AR_VALUE_NORMAL(ar_ptr_offsets[0])};
      // return _custom_ar_ops[ar_index].op_vector(v);
      if (ar_ptr_offsets.size() == 1) {
        return _custom_ar_ops[ar_index].op_vector({index, GET_AR_VALUE_NORMAL(ar_ptr_offsets[0])});
      }
      else if (ar_ptr_offsets.size() == 2) {
        return _custom_ar_ops[ar_index].op_vector({index, GET_AR_VALUE_NORMAL(ar_ptr_offsets[0]), GET_AR_VALUE_NORMAL(ar_ptr_offsets[1])});
      }
      else if (ar_ptr_offsets.size() == 3) {
        // printf("ar_index=%i, dist_idx=%i, current_offset=%i, current_symbol=%i, result=%i\n",
        //   ar_index, index, current_offset, symbols_ptr[current_offset], _custom_ar_ops[ar_index].op_vector({index, GET_AR_VALUE_NORMAL(ar_ptr_offsets[0]), GET_AR_VALUE_NORMAL(ar_ptr_offsets[1]), GET_AR_VALUE_NORMAL(ar_ptr_offsets[2])}));
        return _custom_ar_ops[ar_index].op_vector({index, GET_AR_VALUE_NORMAL(ar_ptr_offsets[0]), GET_AR_VALUE_NORMAL(ar_ptr_offsets[1]), GET_AR_VALUE_NORMAL(ar_ptr_offsets[2])});
      }
      else {
        throw py::value_error("Too many dimensions!");
      } 

    }
    // NOTE: for compability
    else {
#define GET_AR_VALUE_DEFAULT(off) (off[current_offset] > 0) ? symbols_ptr[current_offset - off[current_offset]]+1 : 0
      // NOTE: here we do not check if the offset is valid! Do check this in python!
      if (ar_ptr_offsets.size() == 1) {
        // printf("ar_ptr_offsets[0]=%i, ar_index=%i, dist_idx=%i, current_offset=%i, current_symbol=%i, ar_symbol=%i, result=%i\n", 
        //   ar_ptr_offsets[0][current_offset], ar_index, index, current_offset, symbols_ptr[current_offset], GET_AR_VALUE_DEFAULT(ar_ptr_offsets[0]), _ar_tables_3d[ar_index][index][GET_AR_VALUE_DEFAULT(ar_ptr_offsets[0])]);
        return _ar_tables_3d[ar_index][index][GET_AR_VALUE_DEFAULT(ar_ptr_offsets[0])];
      }
      else if (ar_ptr_offsets.size() == 2) {
        // printf("ar_ptr_offsets[0]=%i, ar_ptr_offsets[1]=%i, ar_index=%i, dist_idx=%i, current_offset=%i, current_symbol=%i, ar_symbol=%i\n",
        //   ar_ptr_offsets[0][current_offset], ar_ptr_offsets[1][current_offset], ar_index, index, current_offset, symbols_ptr[current_offset], GET_AR_VALUE_DEFAULT(ar_ptr_offsets[0]));
        return _ar_tables_4d[ar_index][index][GET_AR_VALUE_DEFAULT(ar_ptr_offsets[0])][GET_AR_VALUE_DEFAULT(ar_ptr_offsets[1])];
      }
      else {
        throw py::value_error("Too many dimensions!");
      } 
    }
  };

  bool _is_initialized = false;
  bool _is_ar_initialized = false;
//   const unsigned _freq_precision;
//   const unsigned _state_precision;

  std::vector<ar_limited_scaled_add_linear_op<NUMPY_ARRAY_TYPE, float>> _custom_ar_ops;

  std::vector<std::vector<std::vector<NUMPY_ARRAY_TYPE>>> _ar_offsets;

  // std::vector<size_t> _ar_tables_1d;
  // std::vector<std::vector<NUMPY_ARRAY_TYPE>> _ar_tables_2d;
  std::vector<std::vector<std::vector<NUMPY_ARRAY_TYPE>>> _ar_tables_3d;
  std::vector<std::vector<std::vector<std::vector<NUMPY_ARRAY_TYPE>>>> _ar_tables_4d;
};


class ANSEncoder {
public:
  virtual py::bytes encode_with_indexes(const py::array_t<NUMPY_ARRAY_TYPE> &symbols,
                           const py::array_t<NUMPY_ARRAY_TYPE> &indexes,
                           const std::optional<py::array_t<NUMPY_ARRAY_TYPE>> &ar_indexes,
                           const std::optional<py::array_t<NUMPY_ARRAY_TYPE>> &ar_offsets,
                           const std::optional<bool> cache) = 0;

  virtual py::bytes flush() = 0;

// protected:
//   const unsigned _freq_precision;
//   const unsigned _state_precision;
};

class ANSDecoder {
public:
  virtual py::array_t<NUMPY_ARRAY_TYPE> decode_with_indexes(const std::string &encoded,
                      const py::array_t<NUMPY_ARRAY_TYPE> &indexes,
                      const std::optional<py::array_t<NUMPY_ARRAY_TYPE>> &ar_indexes,
                      const std::optional<py::array_t<NUMPY_ARRAY_TYPE>> &ar_offsets
                      ) = 0;

  virtual void set_stream(const std::string &stream) {
    _stream = stream;
  };

  virtual py::array_t<NUMPY_ARRAY_TYPE> decode_stream(
    const py::array_t<NUMPY_ARRAY_TYPE> &indexes,
    const std::optional<py::array_t<NUMPY_ARRAY_TYPE>> &ar_indexes,
    const std::optional<py::array_t<NUMPY_ARRAY_TYPE>> &ar_offsets
  ) = 0;

// protected:
//   const unsigned _freq_precision;
//   const unsigned _state_precision;

private:
  std::string _stream;
};

#define PYBIND11_ANS_CLASSES(m) \
  py::class_<ANSEncoder>(m, "ANSEncoder");\
  py::class_<ANSDecoder>(m, "ANSDecoder");

// TODO: add overridable virtual functions in python (https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python)