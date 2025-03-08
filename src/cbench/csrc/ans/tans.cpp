#include "tans.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>

namespace py = pybind11;

/* C style code from FSE*/
namespace {

static unsigned FSE_minTableLog(size_t srcSize, unsigned maxSymbolValue)
{
	U32 minBitsSrc = BIT_highbit32((U32)(srcSize - 1)) + 1;
	U32 minBitsSymbols = BIT_highbit32(maxSymbolValue) + 2;
	U32 minBits = minBitsSrc < minBitsSymbols ? minBitsSrc : minBitsSymbols;
	return minBits;
}

/* Secondary normalization method.
   To be used when primary method fails. */
static size_t Tans_normalizeM2(short* norm, U32 tableLog, const unsigned* count, size_t total, U32 maxSymbolValue)
{
    U32 s;
    U32 distributed = 0;
    U32 ToDistribute;

    /* Init */
    U32 lowThreshold = (U32)(total >> tableLog);
    U32 lowOne = (U32)((total * 3) >> (tableLog + 1));

    for (s=0; s<=maxSymbolValue; s++) {
        if (count[s] == 0) {
            norm[s]=0;
            continue;
        }
        if (count[s] <= lowThreshold) {
            norm[s] = -1;
            distributed++;
            total -= count[s];
            continue;
        }
        if (count[s] <= lowOne) {
            norm[s] = 1;
            distributed++;
            total -= count[s];
            continue;
        }
        norm[s]=-2;
    }
    ToDistribute = (1 << tableLog) - distributed;

    if ((total / ToDistribute) > lowOne) {
        /* risk of rounding to zero */
        lowOne = (U32)((total * 3) / (ToDistribute * 2));
        for (s=0; s<=maxSymbolValue; s++) {
            if ((norm[s] == -2) && (count[s] <= lowOne)) {
                norm[s] = 1;
                distributed++;
                total -= count[s];
                continue;
        }   }
        ToDistribute = (1 << tableLog) - distributed;
    }

    if (distributed == maxSymbolValue+1) {
        /* all values are pretty poor;
           probably incompressible data (should have already been detected);
           find max, then give all remaining points to max */
        U32 maxV = 0, maxC = 0;
        for (s=0; s<=maxSymbolValue; s++)
            if (count[s] > maxC) maxV=s, maxC=count[s];
        norm[maxV] += (short)ToDistribute;
        return 0;
    }

    {
        U64 const vStepLog = 62 - tableLog;
        U64 const mid = (1ULL << (vStepLog-1)) - 1;
        U64 const rStep = ((((U64)1<<vStepLog) * ToDistribute) + mid) / total;   /* scale on remaining */
        U64 tmpTotal = mid;
        for (s=0; s<=maxSymbolValue; s++) {
            if (norm[s]==-2) {
                U64 end = tmpTotal + (count[s] * rStep);
                U32 sStart = (U32)(tmpTotal >> vStepLog);
                U32 sEnd = (U32)(end >> vStepLog);
                U32 weight = sEnd - sStart;
                if (weight < 1)
                    return ERROR(GENERIC);
                norm[s] = (short)weight;
                tmpTotal = end;
    }   }   }

    return 0;
}

size_t Tans_normalizeCount (short* normalizedCounter, unsigned tableLog,
                           const unsigned* count, size_t total,
                           unsigned maxSymbolValue)
{
    /* Sanity checks */
    if (tableLog==0) tableLog = FSE_DEFAULT_TABLELOG;
    // if (tableLog < FSE_MIN_TABLELOG) return ERROR(GENERIC);   /* Unsupported size */
    // if (tableLog > FSE_MAX_TABLELOG) return ERROR(tableLog_tooLarge);   /* Unsupported size */
    if (tableLog < FSE_minTableLog(total, maxSymbolValue)) return ERROR(GENERIC);   /* Too small tableLog, compression potentially impossible */

    {   U32 const rtbTable[] = {     0, 473195, 504333, 520860, 550000, 700000, 750000, 830000 };

        U64 const scale = 62 - tableLog;
        U64 const step = ((U64)1<<62) / total;   /* <== here, one division ! */
        U64 const vStep = 1ULL<<(scale-20);
        int stillToDistribute = 1<<tableLog;
        unsigned s;
        unsigned largest=0;
        short largestP=0;
        U32 lowThreshold = (U32)(total >> tableLog);

        for (s=0; s<=maxSymbolValue; s++) {
            if (count[s] == total) return 0;   /* rle special case */
            if (count[s] == 0) { normalizedCounter[s]=0; continue; }
            if (count[s] <= lowThreshold) {
                normalizedCounter[s] = -1;
                stillToDistribute--;
            } else {
                short proba = (short)((count[s]*step) >> scale);
                if (proba<8) {
                    U64 restToBeat = vStep * rtbTable[proba];
                    proba += (count[s]*step) - ((U64)proba<<scale) > restToBeat;
                }
                if (proba > largestP) largestP=proba, largest=s;
                normalizedCounter[s] = proba;
                stillToDistribute -= proba;
        }   }
        if (-stillToDistribute >= (normalizedCounter[largest] >> 1)) {
            /* corner case, need another normalization method */
            size_t errorCode = Tans_normalizeM2(normalizedCounter, tableLog, count, total, maxSymbolValue);
            if (ERR_isError(errorCode)) return errorCode;
        }
        else normalizedCounter[largest] += (short)stillToDistribute;
    }

    return tableLog;
}

size_t Tans_buildCTable(FSE_CTable* ct, const short* normalizedCounter, unsigned maxSymbolValue, unsigned tableLog)
{
    U32 const tableSize = 1 << tableLog;
    U32 const tableMask = tableSize - 1;
    void* const ptr = ct;
    U16* const tableU16 = ( (U16*) ptr) + 2;
    void* const FSCT = ((U32*)ptr) + 1 /* header */ + (tableLog ? tableSize>>1 : 1) ;
    FSE_symbolCompressionTransform* const symbolTT = (FSE_symbolCompressionTransform*) (FSCT);
    U32 const step = FSE_TABLESTEP(tableSize);
    U32 cumul[maxSymbolValue+2];

    TANS_FUNCTION_TYPE tableSymbol[tableSize]; /* memset() is not necessary, even if static analyzer complain about it */
    U32 highThreshold = tableSize-1;

    /* CTable header */
    tableU16[-2] = (U16) tableLog;
    tableU16[-1] = (U16) maxSymbolValue;

    /* For explanations on how to distribute symbol values over the table :
    *  http://fastcompression.blogspot.fr/2014/02/fse-distributing-symbol-values.html */

    /* symbol start positions */
    {   U32 u;
        cumul[0] = 0;
        for (u=1; u<=maxSymbolValue+1; u++) {
            if (normalizedCounter[u-1]==-1) {  /* Low proba symbol */
                cumul[u] = cumul[u-1] + 1;
                tableSymbol[highThreshold--] = (TANS_FUNCTION_TYPE)(u-1);
            } else {
                cumul[u] = cumul[u-1] + normalizedCounter[u-1];
        }   }
        cumul[maxSymbolValue+1] = tableSize+1;
    }

    /* Spread symbols */
    {   U32 position = 0;
        U32 symbol;
        for (symbol=0; symbol<=maxSymbolValue; symbol++) {
            int nbOccurences;
            for (nbOccurences=0; nbOccurences<normalizedCounter[symbol]; nbOccurences++) {
                tableSymbol[position] = (TANS_FUNCTION_TYPE)symbol;
                position = (position + step) & tableMask;
                while (position > highThreshold) position = (position + step) & tableMask;   /* Low proba area */
        }   }

        if (position!=0) return ERROR(GENERIC);   /* Must have gone through all positions */
    }

    /* Build table */
    {   U32 u; for (u=0; u<tableSize; u++) {
        TANS_FUNCTION_TYPE s = tableSymbol[u];   /* note : static analyzer may not understand tableSymbol is properly initialized */
        tableU16[cumul[s]++] = (U16) (tableSize+u);   /* TableU16 : sorted by symbol order; gives next state value */
    }   }

    /* Build Symbol Transformation Table */
    {   unsigned total = 0;
        unsigned s;
        for (s=0; s<=maxSymbolValue; s++) {
            switch (normalizedCounter[s])
            {
            case  0: break;

            case -1:
            case  1:
                symbolTT[s].deltaNbBits = (tableLog << 16) - (1<<tableLog);
                symbolTT[s].deltaFindState = total - 1;
                total ++;
                break;
            default :
                {
                    U32 const maxBitsOut = tableLog - BIT_highbit32 (normalizedCounter[s]-1);
                    U32 const minStatePlus = normalizedCounter[s] << maxBitsOut;
                    symbolTT[s].deltaNbBits = (maxBitsOut << 16) - minStatePlus;
                    symbolTT[s].deltaFindState = total - normalizedCounter[s];
                    total +=  normalizedCounter[s];
    }   }   }   }

    return 0;
}

MEM_STATIC void Tans_initCTable(Tans_CTable_struct* tablePtr, const FSE_CTable* ct)
{
    const void* ptr = ct;
    const U16* u16ptr = (const U16*) ptr;
    const U32 maxSymbolValue = MEM_read16(u16ptr+1);
    const U32 tableLog = MEM_read16(u16ptr);
    tablePtr->maxSymbolValue = maxSymbolValue;
    tablePtr->stateTable = u16ptr+2;
    tablePtr->symbolTT = ((const U32*)ct + 1 + (tableLog ? (1<<(tableLog-1)) : 1));
}

MEM_STATIC void Tans_initState(TansState* statePtr, uint16_t tableLog)
{
    statePtr->value = (ptrdiff_t)1<<tableLog;
    statePtr->stateLog = tableLog;
}

MEM_STATIC void Tans_encodeSymbol(BIT_CStream_t* bitC, const Tans_CTable_struct* tablePtr, TansState* statePtr, uint16_t symbol)
{
    const FSE_symbolCompressionTransform symbolTT = ((const FSE_symbolCompressionTransform*)(tablePtr->symbolTT))[symbol];
    const U16* const stateTable = (const U16*)(tablePtr->stateTable);
    U32 nbBitsOut  = (U32)((statePtr->value + symbolTT.deltaNbBits) >> 16);
    BIT_addBits(bitC, statePtr->value, nbBitsOut);
    statePtr->value = stateTable[ (statePtr->value >> nbBitsOut) + symbolTT.deltaFindState];
}

MEM_STATIC void Tans_flushCState(BIT_CStream_t* bitC, const TansState* statePtr)
{
    BIT_addBits(bitC, statePtr->value, statePtr->stateLog);
    BIT_flushBits(bitC);
}

size_t Tans_buildDTable(FSE_DTable* dt, const short* normalizedCounter, unsigned maxSymbolValue, unsigned tableLog)
{
    void* const tdPtr = dt+1;   /* because *dt is unsigned, 32-bits aligned on 32-bits */
    Tans_Decode_struct* const tableDecode = (Tans_Decode_struct*) (tdPtr);
    U16 symbolNext[maxSymbolValue+1];

    U32 const maxSV1 = maxSymbolValue + 1;
    U32 const tableSize = 1 << tableLog;
    U32 highThreshold = tableSize-1;

    /* Sanity Checks */
    if (maxSymbolValue > std::numeric_limits<TANS_DECODE_SYMBOL_TYPE>::max()-1) return ERROR(maxSymbolValue_tooLarge);
    if (tableLog > TANS_MAX_TABLELOG) return ERROR(tableLog_tooLarge);

    /* Init, lay down lowprob symbols */
    {   FSE_DTableHeader DTableH;
        DTableH.tableLog = (U16)tableLog;
        DTableH.fastMode = 1;
        {   S16 const largeLimit= (S16)(1 << (tableLog-1));
            U32 s;
            for (s=0; s<maxSV1; s++) {
                if (normalizedCounter[s]==-1) {
                    tableDecode[highThreshold--].symbol = (TANS_DECODE_SYMBOL_TYPE)s;
                    symbolNext[s] = 1;
                } else {
                    if (normalizedCounter[s] >= largeLimit) DTableH.fastMode=0;
                    symbolNext[s] = normalizedCounter[s];
        }   }   }
        memcpy(dt, &DTableH, sizeof(DTableH));
    }

    /* Spread symbols */
    {   U32 const tableMask = tableSize-1;
        U32 const step = FSE_TABLESTEP(tableSize);
        U32 s, position = 0;
        for (s=0; s<maxSV1; s++) {
            int i;
            for (i=0; i<normalizedCounter[s]; i++) {
                tableDecode[position].symbol = (TANS_DECODE_SYMBOL_TYPE)s;
                position = (position + step) & tableMask;
                while (position > highThreshold) position = (position + step) & tableMask;   /* lowprob area */
        }   }

        if (position!=0) return ERROR(GENERIC);   /* position must reach all cells once, otherwise normalizedCounter is incorrect */
    }

    /* Build Decoding table */
    {   U32 u;
        for (u=0; u<tableSize; u++) {
            TANS_DECODE_SYMBOL_TYPE const symbol = (TANS_DECODE_SYMBOL_TYPE)(tableDecode[u].symbol);
            U16 nextState = symbolNext[symbol]++;
            tableDecode[u].nbBits = (BYTE) (tableLog - BIT_highbit32 ((U32)nextState) );
            tableDecode[u].newState = (TANS_DECODE_STATE_TYPE) ( (nextState << tableDecode[u].nbBits) - tableSize);
    }   }

    return 0;
}

MEM_STATIC void Tans_initDTable(Tans_DTable_struct* DTablePtr, const FSE_DTable* dt, unsigned maxSymbolValue)
{
    const FSE_DTableHeader* DTableH = (const FSE_DTableHeader*)dt;
    const U32 fastMode = DTableH->fastMode;

    DTablePtr->table = dt + 1;
    DTablePtr->maxSymbolValue = maxSymbolValue;
    DTablePtr->fastMode = DTableH->fastMode;
}

MEM_STATIC void Tans_initDState(TansState* DStatePtr, BIT_DStream_t* bitD, uint16_t tableLog)
{
    DStatePtr->value = BIT_readBits(bitD, tableLog);
    BIT_reloadDStream(bitD);
    DStatePtr->stateLog = tableLog;
}

MEM_STATIC TANS_DECODE_SYMBOL_TYPE Tans_peekSymbol(const Tans_DTable_struct* DTablePtr, const TansState* DStatePtr)
{
    Tans_Decode_struct const DInfo = ((const Tans_Decode_struct*)(DTablePtr->table))[DStatePtr->value];
    return DInfo.symbol;
}

MEM_STATIC TANS_DECODE_SYMBOL_TYPE Tans_decodeSymbol(const Tans_DTable_struct* DTablePtr, TansState* DStatePtr, BIT_DStream_t* bitD)
{
    Tans_Decode_struct const DInfo = ((const Tans_Decode_struct*)(DTablePtr->table))[DStatePtr->value];
    U32 const nbBits = DInfo.nbBits;
    TANS_DECODE_SYMBOL_TYPE const symbol = DInfo.symbol;
    size_t const lowBits = BIT_readBits(bitD, nbBits);

    DStatePtr->value = DInfo.newState + lowBits;
    return symbol;
}

/*! FSE_decodeSymbolFast() :
    unsafe, only works if no symbol has a probability > 50% */
MEM_STATIC TANS_DECODE_SYMBOL_TYPE Tans_decodeSymbolFast(const Tans_DTable_struct* DTablePtr, TansState* DStatePtr, BIT_DStream_t* bitD)
{
    Tans_Decode_struct const DInfo = ((const Tans_Decode_struct*)(DTablePtr->table))[DStatePtr->value];
    U32 const nbBits = DInfo.nbBits;
    TANS_DECODE_SYMBOL_TYPE const symbol = DInfo.symbol;
    size_t const lowBits = BIT_readBitsFast(bitD, nbBits);

    DStatePtr->value = DInfo.newState + lowBits;
    return symbol;
}


} // namespace

void TansBase::init_params(const py::array_t<NUMPY_ARRAY_TYPE> &freqs,
                           const py::array_t<NUMPY_ARRAY_TYPE> &num_symbols,
                           const py::array_t<NUMPY_ARRAY_TYPE> &offsets) {
  
  if (freqs.ndim() != 2 || freqs.shape(0) != num_symbols.size()) {
    throw py::value_error("freqs should be 2-dimensional with shape (num_symbols.size(), >num_symbols.max())");
  }

  init_tables(freqs, num_symbols);

  _offsets = std::vector<NUMPY_ARRAY_TYPE>(offsets.data(), offsets.data() + offsets.size());

  _is_initialized = true;
}

void TansEncoder::init_tables(const py::array_t<NUMPY_ARRAY_TYPE> &freqs,
                           const py::array_t<NUMPY_ARRAY_TYPE> &num_symbols) {
  
  if (freqs.ndim() != 2 || freqs.shape(0) != num_symbols.size()) {
    throw py::value_error("freqs should be 2-dimensional with shape (num_symbols.size(), >num_symbols.max())");
  }

  _tables = std::vector<std::vector<uint32_t>>(freqs.shape(0));
  _table_structs = std::vector<Tans_CTable_struct>(freqs.shape(0));
  // _table_sizes = std::vector<uint32_t>(freqs.shape(0));
  for (ssize_t idx=0; idx < freqs.shape(0); idx++) {
    const uint32_t nsym = static_cast<uint32_t>(num_symbols.at(idx));
    const NUMPY_ARRAY_TYPE* freq_ptr = freqs.data(idx);

    const uint32_t table_size = FSE_CTABLE_SIZE_U32(_freq_precision, nsym-1);
    // std::vector<uint32_t> table(table_size);
    _tables[idx].resize(table_size);
    const auto &table = _tables[idx];

    // FSE code (C style) 
    U32   count[nsym];
    S16   norm[nsym];
    size_t errorCode;

    size_t total=0;
    for (size_t i=0; i<nsym; i++) {
      const auto freq = freq_ptr[i];
      count[i] = freq;
      total += freq;
    }

    errorCode = Tans_normalizeCount (norm, _freq_precision, count, total, nsym-1);
    if (ERR_isError(errorCode)) {
      throw py::value_error(ERR_getErrorName(errorCode));
    }
    errorCode = Tans_buildCTable(static_cast<FSE_CTable*>(&_tables[idx][0]), norm, nsym-1, _freq_precision);
    if (ERR_isError(errorCode)) {
      throw py::value_error(ERR_getErrorName(errorCode));
    }

    // _tables[idx] = table;
    Tans_initCTable(&_table_structs[idx], static_cast<const FSE_CTable*>(table.data()));
    // _table_sizes[idx] = table_size;
  }

  if (_bypass_coding) {
    const uint32_t table_size = FSE_CTABLE_SIZE_U32(_freq_precision, _max_bypass_val);
    _table_bypass.resize(table_size);
    
    // FSE code (C style) 
    U32   count[_max_bypass_val+1];
    S16   norm[_max_bypass_val+1];
    size_t errorCode;

    size_t total=0;
    for (size_t i=0; i<_max_bypass_val+1; i++) {
      count[i] = 1;
      total += 1;
    }

    errorCode = Tans_normalizeCount (norm, _freq_precision, count, total, _max_bypass_val);
    if (ERR_isError(errorCode)) {
      throw py::value_error(ERR_getErrorName(errorCode));
    }
    errorCode = Tans_buildCTable(static_cast<FSE_CTable*>(&_table_bypass[0]), norm, _max_bypass_val, _freq_precision);
    if (ERR_isError(errorCode)) {
      throw py::value_error(ERR_getErrorName(errorCode));
    }

    Tans_initCTable(&_bypass_table_struct, static_cast<const FSE_CTable*>(_table_bypass.data()));
    
  }

}

void TansDecoder::init_tables(const py::array_t<NUMPY_ARRAY_TYPE> &freqs,
                           const py::array_t<NUMPY_ARRAY_TYPE> &num_symbols) {
  
  if (freqs.ndim() != 2 || freqs.shape(0) != num_symbols.size()) {
    throw py::value_error("freqs should be 2-dimensional with shape (num_symbols.size(), >num_symbols.max())");
  }

  _tables = std::vector<std::vector<uint32_t>>(freqs.shape(0));
  _table_structs = std::vector<Tans_DTable_struct>(freqs.shape(0));
  // _table_sizes = std::vector<uint32_t>(freqs.shape(0));
  for (ssize_t idx=0; idx < freqs.shape(0); idx++) {
    const uint32_t nsym = static_cast<uint32_t>(num_symbols.at(idx));
    const NUMPY_ARRAY_TYPE* freq_ptr = freqs.data(idx);

    const uint32_t table_size = TANS_FSE_DTABLE_SIZE_U32(_freq_precision);
    _tables[idx].resize(table_size);
    const auto &table = _tables[idx];

    // FSE code (C style) 
    U32   count[nsym];
    S16   norm[nsym];
    size_t errorCode;

    size_t total=0;
    for (size_t i=0; i<nsym; i++) {
      const auto freq = freq_ptr[i];
      count[i] = freq;
      total += freq;
    }

    errorCode = Tans_normalizeCount (norm, _freq_precision, count, total, nsym-1);
    if (ERR_isError(errorCode)) {
      throw py::value_error(ERR_getErrorName(errorCode));
    }
    errorCode = Tans_buildDTable(static_cast<FSE_DTable*>(&_tables[idx][0]), norm, nsym-1, _freq_precision);
    if (ERR_isError(errorCode)) {
      throw py::value_error(ERR_getErrorName(errorCode));
    }

    // _tables[idx] = table;
    Tans_initDTable(&_table_structs[idx], static_cast<const FSE_DTable*>(table.data()), nsym-1);
    // _table_sizes[idx] = table_size;
  }

  if (_bypass_coding) {
    const uint32_t table_size = TANS_FSE_DTABLE_SIZE_U32(_freq_precision);
    _table_bypass.resize(table_size);

    // FSE code (C style) 
    U32   count[_max_bypass_val+1];
    S16   norm[_max_bypass_val+1];
    size_t errorCode;

    size_t total=0;
    for (size_t i=0; i<_max_bypass_val+1; i++) {
      count[i] = 1;
      total += 1;
    }

    errorCode = Tans_normalizeCount (norm, _freq_precision, count, total, _max_bypass_val);
    if (ERR_isError(errorCode)) {
      throw py::value_error(ERR_getErrorName(errorCode));
    }
    errorCode = Tans_buildDTable(static_cast<FSE_DTable*>(&_table_bypass[0]), norm, _max_bypass_val, _freq_precision);
    if (ERR_isError(errorCode)) {
      throw py::value_error(ERR_getErrorName(errorCode));
    }

    Tans_initDTable(&_bypass_table_struct, static_cast<const FSE_DTable*>(_table_bypass.data()), _max_bypass_val);
  }
}


py::bytes TansEncoder::encode_with_indexes(
    const py::array_t<NUMPY_ARRAY_TYPE> &symbols, const py::array_t<NUMPY_ARRAY_TYPE> &indexes, 
    const std::optional<py::array_t<NUMPY_ARRAY_TYPE>> &ar_indexes, 
    const std::optional<py::array_t<NUMPY_ARRAY_TYPE>> &ar_offsets,
    const std::optional<bool> cache) {

  if (!_is_initialized) {
    throw py::value_error("ANS not initialized!");
  }

  const NUMPY_ARRAY_TYPE* symbols_ptr = symbols.data();
  const NUMPY_ARRAY_TYPE* indexes_ptr = indexes.data();
  const NUMPY_ARRAY_TYPE* ar_indexes_ptr = nullptr;
  std::vector<const NUMPY_ARRAY_TYPE*> ar_offsets_ptrs;
  if (ar_indexes.has_value()) {
    ar_indexes_ptr = ar_indexes.value().data();
  }

  if (_is_ar_initialized) {
    if (!ar_offsets.has_value()) {
      throw py::value_error("ar_offsets is required for ar coding!");
    }
    if (ar_indexes.has_value()) {
      ar_indexes_ptr = ar_indexes.value().data();
    }
    // if (ar_offsets.has_value()) {
    for (ssize_t i=0; i<ar_offsets.value().shape(0); i++){
      ar_offsets_ptrs.push_back(ar_offsets.value().data(i));
    }
  }

  assert(symbols.size() == indexes.size());

  std::vector<uint8_t> output;
  BIT_CStream_t bitC;
  TansState CState;

  if (!cache.value_or(false)) {
    // worst case result (for none overflow)
    // output = std::vector<uint8_t>((indexes.size() * _freq_precision / 8) + 1, 0xCC);
    output = std::vector<uint8_t>((indexes.size() * _freq_precision / 8), 0xCC);

    /* init */
    const auto errorCode = BIT_initCStream(&bitC, (void*) &output[0], output.size());
    if (ERR_isError(errorCode)) {
      throw py::value_error(ERR_getErrorName(errorCode));
    }
    Tans_initState(&CState, _freq_precision);

  }

  // backward loop on symbols from the end;
  for (ssize_t i = symbols.size()-1; i >= 0; --i) {
    NUMPY_ARRAY_TYPE table_idx = indexes_ptr[i];
    assert(table_idx >= 0);
    assert(table_idx < _tables.size());

    if (_is_ar_initialized) {
      auto ar_idx = (ar_indexes_ptr == nullptr) ? 0 : ar_indexes_ptr[i];
      // auto ar_ptr_off = ar_ptr_offsets[ar_idx];
      table_idx = ar_update_index(ar_offsets_ptrs, ar_idx, table_idx, symbols_ptr, i);
    }

    const auto &table = _table_structs[table_idx];

    const NUMPY_ARRAY_TYPE max_value = table.maxSymbolValue;
    assert(max_value >= 0);

    NUMPY_ARRAY_TYPE value = symbols_ptr[i] - _offsets[table_idx];

    // std::cout << "table_idx: " << table_idx << std::endl;
    // std::cout << "value: " << value << std::endl;

    uint32_t raw_val = 0;
    if (value < 0) {
      raw_val = -2 * value - 1;
      value = max_value;
    } else if (value >= max_value) {
      raw_val = 2 * (value - max_value);
      value = max_value;
    }

    assert(value >= 0);
    assert(value <= table.maxSymbolValue);

    TansSymbol sym = {static_cast<uint16_t>(value),
                  static_cast<uint16_t>(table_idx),
                  false};

    /* Bypass coding mode (value == max_value -> sentinel flag) */
    if (_bypass_coding) {

      // NOTE: unlike rans, tans must keep the same table precision for bypass
      if (value == max_value) {
        std::vector<TansSymbol> bypass_syms;
        /* Determine the number of bypasses (in _freq_precision size) needed to
        * encode the raw value. */
        int32_t n_bypass = 0;
        while ((raw_val >> (n_bypass * _bypass_precision)) != 0) {
          ++n_bypass;
        }

        const uint16_t max_bypass_val = _max_bypass_val; // _bypass_table_struct.maxSymbolValue;

        /* Encode number of bypasses */
        int32_t val = n_bypass;
        while (val >= max_bypass_val) {
          bypass_syms.push_back({max_bypass_val, 0, true});
          val -= max_bypass_val;
        }
        bypass_syms.push_back(
            {static_cast<uint16_t>(val), 0, true});

        /* Encode raw value */
        for (int32_t j = 0; j < n_bypass; ++j) {
          const int32_t val =
              (raw_val >> (j * _bypass_precision)) & max_bypass_val;
          bypass_syms.push_back(
              {static_cast<uint16_t>(val), 0, true});
        }

        // bypass_syms should be encoded in reverse order!
        if (!cache.value_or(false)) {
          while (!bypass_syms.empty()) {
            const TansSymbol sym = bypass_syms.back();
            Tans_encodeSymbol(&bitC, &_bypass_table_struct, &CState, sym.value);
            BIT_flushBits(&bitC);
            bypass_syms.pop_back();
          }
        }
        else {
          _syms.insert(_syms.end(), bypass_syms.rbegin(), bypass_syms.rend());
        }

      }
    }

    if (!cache.value_or(false)) {
      Tans_encodeSymbol(&bitC, &table, &CState, value);
      BIT_flushBits(&bitC);
    }
    else {
      _syms.push_back(sym);
    }


  }

  if (!cache.value_or(false)) {
    Tans_flushCState(&bitC, &CState);

    const int nbytes = BIT_closeCStream(&bitC);
    return std::string(reinterpret_cast<char *>(&output[0]), nbytes);
  }
  else {
    // return empty string if cached
    return "";
  }

}

py::bytes TansEncoder::flush() {
  std::vector<uint8_t> output;
  BIT_CStream_t bitC;
  TansState CState;

  // worst case result (for none overflow)
  // output = std::vector<uint8_t>((indexes.size() * _freq_precision / 8) + 1, 0xCC);
  output = std::vector<uint8_t>((_syms.size() * _freq_precision / 8), 0xCC);

  /* init */
  const auto errorCode = BIT_initCStream(&bitC, (void*) &output[0], output.size());
  if (ERR_isError(errorCode)) {
    throw py::value_error(ERR_getErrorName(errorCode));
  }
  Tans_initState(&CState, _freq_precision);

  for (auto sym : _syms) {
    if (!sym.bypass) {
      Tans_encodeSymbol(&bitC, &_table_structs[sym.index], &CState, sym.value);
      BIT_flushBits(&bitC);
    } else {
      // TODO: bypass coding
      Tans_encodeSymbol(&bitC, &_bypass_table_struct, &CState, sym.value);
      BIT_flushBits(&bitC);
    }
  }

  _syms.clear();

  Tans_flushCState(&bitC, &CState);

  const int nbytes = BIT_closeCStream(&bitC);
  return std::string(reinterpret_cast<char *>(&output[0]), nbytes);

}

py::array_t<NUMPY_ARRAY_TYPE>
TansDecoder::decode_with_indexes(const std::string &encoded,
                                 const py::array_t<NUMPY_ARRAY_TYPE> &indexes,
                                 const std::optional<py::array_t<NUMPY_ARRAY_TYPE>> &ar_indexes,
                                 const std::optional<py::array_t<NUMPY_ARRAY_TYPE>> &ar_offsets) {

  if (!_is_initialized) {
    throw py::value_error("ANS not initialized!");
  }

  const NUMPY_ARRAY_TYPE* indexes_ptr = indexes.data();
  const NUMPY_ARRAY_TYPE* ar_indexes_ptr = nullptr;
  std::vector<const NUMPY_ARRAY_TYPE*> ar_offsets_ptrs;
  if (ar_indexes.has_value()) {
    ar_indexes_ptr = ar_indexes.value().data();
  }

  if (_is_ar_initialized) {
    if (!ar_offsets.has_value()) {
      throw py::value_error("ar_offsets is required for ar coding!");
    }
    if (ar_indexes.has_value()) {
      ar_indexes_ptr = ar_indexes.value().data();
    }
    // if (ar_offsets.has_value()) {
    for (ssize_t i=0; i<ar_offsets.value().shape(0); i++){
      ar_offsets_ptrs.push_back(ar_offsets.value().data(i));
    }
  }

  py::array_t<NUMPY_ARRAY_TYPE> output(indexes.request(true));
  NUMPY_ARRAY_TYPE* output_ptr = output.mutable_data();

  TansState DState;
  BIT_DStream_t bitD;

  const auto errorCode = BIT_initDStream(&bitD, (const void*)encoded.data(), encoded.size());
  if (ERR_isError(errorCode)) {
    throw py::value_error(ERR_getErrorName(errorCode));
  }
  Tans_initDState(&DState, &bitD, _freq_precision);

  for (ssize_t i = 0; i < indexes.size(); ++i) {
    NUMPY_ARRAY_TYPE table_idx = indexes_ptr[i];
    assert(table_idx >= 0);
    assert(table_idx < _tables.size());

    if (_is_ar_initialized) {
      auto ar_idx = (ar_indexes_ptr == nullptr) ? 0 : ar_indexes_ptr[i];
      // auto ar_ptr_off = ar_ptr_offsets[ar_idx];
      table_idx = ar_update_index(ar_offsets_ptrs, ar_idx, table_idx, output_ptr, i);
    }

    const auto &table = _table_structs[table_idx];

    const NUMPY_ARRAY_TYPE max_value = table.maxSymbolValue;
    assert(max_value >= 0);

    const NUMPY_ARRAY_TYPE offset = _offsets[table_idx];

    BIT_reloadDStream(&bitD);

#define MACRO_Tans_decodeSymbol(table) table.fastMode ? Tans_decodeSymbolFast(&table, &DState, &bitD) : Tans_decodeSymbol(&table, &DState, &bitD)
    NUMPY_ARRAY_TYPE value = static_cast<NUMPY_ARRAY_TYPE>(MACRO_Tans_decodeSymbol(table));

    // bypass coding
    if (_bypass_coding) {

      if (value == max_value) {
        /* Bypass decoding mode */
        uint32_t val = MACRO_Tans_decodeSymbol(_bypass_table_struct);
        uint32_t n_bypass = val;
        const unsigned max_bypass_val = _bypass_table_struct.maxSymbolValue;

        while (val == max_bypass_val) {
          val = MACRO_Tans_decodeSymbol(_bypass_table_struct);
          n_bypass += val;
        }

        uint32_t raw_val = 0;
        for (int j = 0; j < n_bypass; ++j) {
          val = MACRO_Tans_decodeSymbol(_bypass_table_struct);
          assert(val <= max_bypass_val);
          raw_val |= val << (j * _bypass_precision);
        }
        value = raw_val >> 1;
        if (raw_val & 1) {
          value = -value - 1;
        } else {
          value += max_value;
        }
      }

    }

    output_ptr[i] = value + offset;
  }

  return output;
}

py::array_t<NUMPY_ARRAY_TYPE> 
TansDecoder::decode_fsar(const std::string &encoded,
                         const int init_value,
                         const int num_symbol,
                         const int height,
                         const int width){
  if (!_is_initialized) {
    throw py::value_error("ANS not initialized!");
  }

  const int height_pad = height + 1;
  const int width_pad = width + 1;
  std::vector<NUMPY_ARRAY_TYPE> output(height_pad * width_pad, init_value);

  TansState DState;
  BIT_DStream_t bitD;

  const auto errorCode = BIT_initDStream(&bitD, (const void*)encoded.data(), encoded.size());
  if (ERR_isError(errorCode)) {
    throw py::value_error(ERR_getErrorName(errorCode));
  }
  Tans_initDState(&DState, &bitD, _freq_precision);

  for (int i = 1; i < height_pad; ++i) {
    for (int j = 1; j < width_pad; ++j) {
      const int cur_idx = i*width_pad + j;
      const int context_idx1 = (i-1)*width_pad + j;
      const int context_idx2 = i*width_pad + j-1;
      const int table_idx = output[context_idx1] * num_symbol + output[context_idx2];
      const auto &table = _table_structs[table_idx];
      const NUMPY_ARRAY_TYPE max_value = table.maxSymbolValue;
      assert(max_value >= 0);
      BIT_reloadDStream(&bitD);
      #define MACRO_Tans_decodeSymbol(table) table.fastMode ? Tans_decodeSymbolFast(&table, &DState, &bitD) : Tans_decodeSymbol(&table, &DState, &bitD)
      NUMPY_ARRAY_TYPE value = static_cast<NUMPY_ARRAY_TYPE>(MACRO_Tans_decodeSymbol(table));
      output[cur_idx] = value;
    }
  }
  py::array_t<NUMPY_ARRAY_TYPE> ret = py::array_t<NUMPY_ARRAY_TYPE>({height_pad, width_pad});
  auto ret_ptr = ret.mutable_data();
  std::copy(output.begin(), output.end(), ret_ptr);
  return ret;
}

void TansDecoder::set_stream(const std::string &stream) {
  ANSDecoder::set_stream(stream);

  // TODO: determine output size?
  // py::array_t<NUMPY_ARRAY_TYPE> output();
  // NUMPY_ARRAY_TYPE* output_ptr = output.mutable_data();

  // const auto errorCode = BIT_initDStream(&_bitD, (const void*)encoded.data(), encoded.size());
  // if (ERR_isError(errorCode)) {
  //   throw py::value_error(ERR_getErrorName(errorCode));
  // }
  // Tans_initDState(&_DState, &_bitD, _freq_precision);
};

py::array_t<NUMPY_ARRAY_TYPE>
TansDecoder::decode_stream(
  const py::array_t<NUMPY_ARRAY_TYPE> &indexes,
  const std::optional<py::array_t<NUMPY_ARRAY_TYPE>> &ar_indexes,
  const std::optional<py::array_t<NUMPY_ARRAY_TYPE>> &ar_offsets) {

  if (!_is_initialized) {
    throw py::value_error("ANS not initialized!");
  }

  py::array_t<NUMPY_ARRAY_TYPE> output(indexes.request(true));

  assert(_ptr != nullptr);

  // TODO: implement decode_stream with tans

  // for (int i = 0; i < static_cast<int>(indexes.size()); ++i) {
  //   const NUMPY_ARRAY_TYPE cdf_idx = indexes[i];
  //   assert(cdf_idx >= 0);
  //   assert(cdf_idx < cdfs.size());

  //   const auto &cdf = cdfs[cdf_idx];

  //   const NUMPY_ARRAY_TYPE max_value = cdfs_sizes[cdf_idx] - 2;
  //   assert(max_value >= 0);
  //   assert((max_value + 1) < cdf.size());

  //   const NUMPY_ARRAY_TYPE offset = offsets[cdf_idx];

  //   const uint32_t cum_freq = TansDecGet(&_rans, _freq_precision);

  //   const auto cdf_end = cdf.begin() + cdfs_sizes[cdf_idx];
  //   const auto it = std::find_if(cdf.begin(), cdf_end,
  //                                [cum_freq](int v) { return v > cum_freq; });
  //   assert(it != cdf_end + 1);
  //   const uint32_t s = std::distance(cdf.begin(), it) - 1;

  //   TansDecAdvance(&_rans, &_ptr, cdf[s], cdf[s + 1] - cdf[s], _freq_precision);

  //   NUMPY_ARRAY_TYPE value = static_cast<NUMPY_ARRAY_TYPE>(s);

  //   if (value == max_value) {
  //     /* Bypass decoding mode */
  //     int32_t val = TansDecGetBits(&_rans, &_ptr, bypass_precision);
  //     int32_t n_bypass = val;

  //     while (val == max_bypass_val) {
  //       val = TansDecGetBits(&_rans, &_ptr, bypass_precision);
  //       n_bypass += val;
  //     }

  //     int32_t raw_val = 0;
  //     for (int j = 0; j < n_bypass; ++j) {
  //       val = TansDecGetBits(&_rans, &_ptr, bypass_precision);
  //       assert(val <= max_bypass_val);
  //       raw_val |= val << (j * bypass_precision);
  //     }
  //     value = raw_val >> 1;
  //     if (raw_val & 1) {
  //       value = -value - 1;
  //     } else {
  //       value += max_value;
  //     }
  //   }

  //   output[i] = value + offset;
  // }

  return output;
}

