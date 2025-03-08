#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "ans_interface.hpp"
#include "rans64.h"

namespace py = pybind11;

struct Rans64Symbol {
  uint16_t start;
  uint16_t range;
  bool bypass; // bypass flag to write raw bits to the stream
};

std::vector<NUMPY_ARRAY_TYPE> pmf_to_quantized_cdf(const std::vector<float> &pmf,
                                           int precision);

class Rans64Base : public ANSBase {
public:
  Rans64Base(
    unsigned freq_precision=16, 
    bool bypass_coding=false,
    unsigned bypass_precision=4
    ): 
    _freq_precision(freq_precision), _bypass_coding(bypass_coding),
    _bypass_precision(bypass_precision) 
    {_max_bypass_val = (1 << bypass_precision) - 1;};

  virtual void init_params(const py::array_t<NUMPY_ARRAY_TYPE> &freqs,
                           const py::array_t<NUMPY_ARRAY_TYPE> &num_symbols,
                           const py::array_t<NUMPY_ARRAY_TYPE> &offsets) override;

  void init_cdf_params(const py::array_t<NUMPY_ARRAY_TYPE> &cdfs,
                           const py::array_t<NUMPY_ARRAY_TYPE> &cdfs_sizes,
                           const py::array_t<NUMPY_ARRAY_TYPE> &offsets);

  py::array_t<NUMPY_ARRAY_TYPE> get_cdfs() {
    if (!_is_initialized) return py::array_t<NUMPY_ARRAY_TYPE>();
    
    size_t max_cdf_length = *std::max_element(_cdfs_sizes.begin(), _cdfs_sizes.end());
    py::array_t<NUMPY_ARRAY_TYPE> output({_cdfs.size(), max_cdf_length});
    for (size_t i=0; i < _cdfs.size(); i++) {
      for (size_t j=0; j < _cdfs[i].size(); j++) {
        output.mutable_at(i,j) = _cdfs[i][j];
      }
    }
    return output;
  };

protected:
  const unsigned _state_precision = 64;
  unsigned _freq_precision;
  bool _bypass_coding;
  unsigned _bypass_precision;
  uint16_t _max_bypass_val;

  bool _is_initialized = false;
  std::vector<std::vector<NUMPY_ARRAY_TYPE>> _cdfs;
  std::vector<NUMPY_ARRAY_TYPE> _cdfs_sizes;
  std::vector<NUMPY_ARRAY_TYPE> _offsets;
};

class Rans64Encoder : public ANSEncoder, public Rans64Base {
  using Rans64Base::Rans64Base;

public:

  py::bytes encode_with_indexes(const py::array_t<NUMPY_ARRAY_TYPE> &symbols,
                           const py::array_t<NUMPY_ARRAY_TYPE> &indexes,
                           const std::optional<py::array_t<NUMPY_ARRAY_TYPE>> &ar_indexes,
                           const std::optional<py::array_t<NUMPY_ARRAY_TYPE>> &ar_offsets,
                           const std::optional<bool> cache) override;

  py::bytes flush() override;

  py::array_t<NUMPY_ARRAY_TYPE> peek_cache() {
    py::array_t<NUMPY_ARRAY_TYPE> output({_syms.size(), (size_t) 3});
    for (size_t i=0; i < _syms.size(); i++) {
      output.mutable_at(i,0) = (NUMPY_ARRAY_TYPE) _syms[i].start;
      output.mutable_at(i,1) = (NUMPY_ARRAY_TYPE) _syms[i].range;
      output.mutable_at(i,2) = (NUMPY_ARRAY_TYPE) _syms[i].bypass;
    }
    return output;
  };

private:
  std::vector<Rans64Symbol> _syms;
};

class Rans64Decoder : public ANSDecoder, public Rans64Base {
  using Rans64Base::Rans64Base;

public:

  py::array_t<NUMPY_ARRAY_TYPE> decode_with_indexes(const std::string &encoded,
                      const py::array_t<NUMPY_ARRAY_TYPE> &indexes,
                      const std::optional<py::array_t<NUMPY_ARRAY_TYPE>> &ar_indexes,
                      const std::optional<py::array_t<NUMPY_ARRAY_TYPE>> &ar_offsets) override;

  void set_stream(const std::string &stream) override {
    ANSDecoder::set_stream(stream);
    uint32_t *ptr = (uint32_t *)_stream.data();
    assert(ptr != nullptr);
    _ptr = ptr;
    Rans64DecInit(&_rans, &_ptr);
  };

  py::array_t<NUMPY_ARRAY_TYPE> decode_stream(
    const py::array_t<NUMPY_ARRAY_TYPE> &indexes,
    const std::optional<py::array_t<NUMPY_ARRAY_TYPE>> &ar_indexes,
    const std::optional<py::array_t<NUMPY_ARRAY_TYPE>> &ar_offsets
  ) override;

private:
  Rans64State _rans;
  std::string _stream;
  uint32_t *_ptr;

};

// TODO: add binding for pure classes
#define PYBIND11_RANS64_CLASSES(m) \
  py::class_<Rans64Encoder>(m, "Rans64Encoder", py::module_local())\
    .def(py::init<unsigned, bool, unsigned>(), py::arg("freq_precision")=16, py::arg("bypass_coding")=true, py::arg("bypass_precision")=4)\
    .def("init_params", &Rans64Encoder::init_params)\
    .def("init_ar_params", &Rans64Encoder::init_ar_params)\
    .def("init_custom_ar_ops", &Rans64Encoder::init_custom_ar_ops)\
    .def("init_cdf_params", &Rans64Encoder::init_cdf_params)\
    .def("get_cdfs", &Rans64Encoder::get_cdfs)\
    .def("peek_cache", &Rans64Encoder::peek_cache)\
    .def("create_ar_ptrs", &Rans64Encoder::create_ar_ptrs)\
    .def("encode_with_indexes", &Rans64Encoder::encode_with_indexes, py::arg("symbols"), py::arg("indexes"), py::arg("ar_indexes")=py::none(), py::arg("ar_offsets")=py::none(), py::arg("cache")=0);\
  py::class_<Rans64Decoder>(m, "Rans64Decoder", py::module_local())\
    .def(py::init<unsigned, bool, unsigned>(), py::arg("freq_precision")=16, py::arg("bypass_coding")=true, py::arg("bypass_precision")=4)\
    .def("init_params", &Rans64Decoder::init_params)\
    .def("init_ar_params", &Rans64Decoder::init_ar_params)\
    .def("init_custom_ar_ops", &Rans64Encoder::init_custom_ar_ops)\
    .def("init_cdf_params", &Rans64Decoder::init_cdf_params)\
    .def("get_cdfs", &Rans64Decoder::get_cdfs)\
    .def("decode_with_indexes", &Rans64Decoder::decode_with_indexes, py::arg("encoded"), py::arg("indexes"), py::arg("ar_indexes")=py::none(), py::arg("ar_offsets")=py::none())\
    .def("set_stream", &Rans64Decoder::set_stream)\
    .def("decode_stream", &Rans64Decoder::decode_stream, py::arg("indexes"), py::arg("ar_indexes")=py::none(), py::arg("ar_indexes")=py::none());\
  m.def("pmf_to_quantized_cdf", &pmf_to_quantized_cdf, "Return quantized CDF for a given PMF");
