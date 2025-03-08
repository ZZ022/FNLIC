#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "ans_interface.hpp"

#define FSE_STATIC_LINKING_ONLY
#include "../FSE/fse.h"

#define TANS_DECODE_STATE_TYPE uint32_t
#define TANS_DECODE_SYMBOL_TYPE uint16_t
#define TANS_FUNCTION_TYPE TANS_DECODE_SYMBOL_TYPE
#define TANS_MAX_TABLELOG 16 // TODO: In fact this is limited by TANS_DECODE_STATE_TYPE. We just follow FSE

namespace py = pybind11;

struct TansSymbol {
  uint16_t value;
  uint16_t index;
  bool bypass; // bypass flag to write raw bits to the stream
};

// py::array_t<NUMPY_ARRAY_TYPE> tans_normalize_freq(py::array_t<NUMPY_ARRAY_TYPE> freq, unsigned tableLog, unsigned maxSymbolValue);
// py::array_t<NUMPY_ARRAY_TYPE> tans_build_ctable(py::array_t<NUMPY_ARRAY_TYPE> normalizedCounter, unsigned tableLog, unsigned maxSymbolValue);
// py::array_t<NUMPY_ARRAY_TYPE> tans_build_dtable(py::array_t<NUMPY_ARRAY_TYPE> normalizedCounter, unsigned tableLog, unsigned maxSymbolValue);

typedef struct
{
    ptrdiff_t   value;
    unsigned    stateLog;
} TansState;

class TansBase : public ANSBase {
public:
  TansBase(
    unsigned table_log=FSE_DEFAULT_TABLELOG, 
    unsigned max_symbol_value=FSE_MAX_SYMBOL_VALUE,
    bool bypass_coding=false,
    unsigned bypass_precision=4
    ): 
    _freq_precision(table_log), _state_precision(32), 
    _max_symbol_value(max_symbol_value), _bypass_coding(bypass_coding),
    _bypass_precision(bypass_precision) 
    {_max_bypass_val = (1 << bypass_precision) - 1;};

  virtual void init_params(const py::array_t<NUMPY_ARRAY_TYPE> &freqs,
                           const py::array_t<NUMPY_ARRAY_TYPE> &num_symbols,
                           const py::array_t<NUMPY_ARRAY_TYPE> &offsets) override;

  virtual void init_tables(const py::array_t<NUMPY_ARRAY_TYPE> &freqs,
                           const py::array_t<NUMPY_ARRAY_TYPE> &num_symbols) = 0;

protected:
  unsigned _freq_precision;
  unsigned _state_precision;
  unsigned _max_symbol_value;
  bool _bypass_coding;
  unsigned _bypass_precision;
  uint16_t _max_bypass_val;

  std::vector<std::vector<uint32_t>> _tables;
  // std::vector<uint32_t> _table_sizes;
  std::vector<NUMPY_ARRAY_TYPE> _offsets;
  
  std::vector<uint32_t> _table_bypass;
};


typedef struct
{
    const void* stateTable;
    const void* symbolTT;
    unsigned    maxSymbolValue;
} Tans_CTable_struct;

class TansEncoder : public TansBase, public ANSEncoder {
  using TansBase::TansBase;

public:
  void init_tables(const py::array_t<NUMPY_ARRAY_TYPE> &freqs,
                           const py::array_t<NUMPY_ARRAY_TYPE> &num_symbols) override;

  py::bytes encode_with_indexes(const py::array_t<NUMPY_ARRAY_TYPE> &symbols,
                           const py::array_t<NUMPY_ARRAY_TYPE> &indexes,
                           const std::optional<py::array_t<NUMPY_ARRAY_TYPE>> &ar_indexes,
                           const std::optional<py::array_t<NUMPY_ARRAY_TYPE>> &ar_offsets,
                           const std::optional<bool> cache) override;

  py::bytes flush() override;

private:
  std::vector<Tans_CTable_struct> _table_structs;
  Tans_CTable_struct _bypass_table_struct;
  std::vector<TansSymbol> _syms;
};

// TODO: this struct is limited by the maximum of symbols and states! 
// Reconsider its definition!
typedef struct
{
    TANS_DECODE_STATE_TYPE newState;
    TANS_DECODE_SYMBOL_TYPE symbol;
    TANS_DECODE_SYMBOL_TYPE nbBits;
} Tans_Decode_struct;

typedef struct
{
    const void* table;
    unsigned    maxSymbolValue;
    unsigned    fastMode;
} Tans_DTable_struct;

// NOTE: sizeof(Tans_Decode_struct) may cannot be reinterpreted with U32
#define TANS_FSE_DTABLE_SIZE_U32(maxTableLog) (1 + (1<<maxTableLog) * sizeof(Tans_Decode_struct) / sizeof(U32) + 1)

class TansDecoder : public TansBase, public ANSDecoder {
  using TansBase::TansBase;

public:
  void init_tables(const py::array_t<NUMPY_ARRAY_TYPE> &freqs,
                           const py::array_t<NUMPY_ARRAY_TYPE> &num_symbols) override;

  py::array_t<NUMPY_ARRAY_TYPE> decode_with_indexes(const std::string &encoded,
                      const py::array_t<NUMPY_ARRAY_TYPE> &indexes,
                      const std::optional<py::array_t<NUMPY_ARRAY_TYPE>> &ar_indexes,
                      const std::optional<py::array_t<NUMPY_ARRAY_TYPE>> &ar_offsets) override;

  py::array_t<NUMPY_ARRAY_TYPE> decode_fsar(const std::string &encoded, 
                      const int init_value,
                      const int num_symbol,
                      const int height,
                      const int width);
  void set_stream(const std::string &stream) override;

  py::array_t<NUMPY_ARRAY_TYPE> decode_stream(
    const py::array_t<NUMPY_ARRAY_TYPE> &indexes,
    const std::optional<py::array_t<NUMPY_ARRAY_TYPE>> &ar_indexes,
    const std::optional<py::array_t<NUMPY_ARRAY_TYPE>> &ar_offsets) override;

private:
  std::vector<Tans_DTable_struct> _table_structs;
  Tans_DTable_struct _bypass_table_struct;

  BIT_DStream_t _bitD;
  TansState _DState;

};
//py::arg("table_log")=FSE_DEFAULT_TABLELOG, py::arg("max_symbol_value")=FSE_MAX_SYMBOL_VALUE, py::arg("bypass_coding")=false
// TODO: add binding for pure classes
#define PYBIND11_TANS_CLASSES(m) \
  py::class_<TansEncoder>(m, "TansEncoder", py::module_local())\
    .def(py::init<unsigned, unsigned, bool, unsigned>(), py::arg("table_log")=FSE_DEFAULT_TABLELOG, py::arg("max_symbol_value")=FSE_MAX_SYMBOL_VALUE, py::arg("bypass_coding")=false, py::arg("bypass_precision")=4)\
    .def("init_params", &TansEncoder::init_params)\
    .def("init_ar_params", &TansEncoder::init_ar_params)\
    .def("encode_with_indexes", &TansEncoder::encode_with_indexes, py::arg("symbols"), py::arg("indexes"), py::arg("ar_indexes")=py::none(), py::arg("ar_offsets")=py::none(), py::arg("cache")=0);\
  py::class_<TansDecoder>(m, "TansDecoder", py::module_local())\
    .def(py::init<unsigned, unsigned, bool, unsigned>(), py::arg("table_log")=FSE_DEFAULT_TABLELOG, py::arg("max_symbol_value")=FSE_MAX_SYMBOL_VALUE, py::arg("bypass_coding")=false, py::arg("bypass_precision")=4)\
    .def("init_params", &TansDecoder::init_params)\
    .def("init_ar_params", &TansDecoder::init_ar_params)\
    .def("decode_with_indexes", &TansDecoder::decode_with_indexes, py::arg("encoded"), py::arg("indexes"), py::arg("ar_indexes")=py::none(), py::arg("ar_offsets")=py::none())\
    .def("decode_fsar", &TansDecoder::decode_fsar, py::arg("encoded"), py::arg("init_value"), py::arg("num_symbol"), py::arg("height"), py::arg("width"));
