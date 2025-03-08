import numpy as np
from typing import List, Tuple

def create_ar_offsets(shape : Tuple[int], ar_dim_offsets: List[List[int]]) -> np.ndarray:
    """Convert dimensional offset to per-element pointer offsets (in c++). \
        Could be used as input for ANSEncoder/ANSDecoder. See tests/ans_test.py.

    Args:
        shape (Tuple[int]): Required tensor shape to generate offsets.
        ar_dim_offsets (List[List[int]]): Dimensional offsets.\
            Sub-lists start from channel dimension. e.g. (-1, ) gets [:, c-1, h, w], (0, -1, -1) gets [:, c, h-1, w-1]

    Returns:
        np.ndarray: per-element pointer offsets.
    """
    ar_offsets = []
    for i, dim_offsets in enumerate(ar_dim_offsets):
        ar_offset_array = -np.ones(shape, dtype=np.int32)
        cur_offset = 0
        for j, off in enumerate(dim_offsets):
            assert off <= 0
            # use j+1 to skip batch dim
            cur_offset += -off * ar_offset_array.strides[j+1] / ar_offset_array.itemsize
            if off < 0:
                for k in range(-off):
                    dim_idx = np.zeros_like(np.compress([True], ar_offset_array, axis=j+1)) + k
                    np.put_along_axis(ar_offset_array, dim_idx, 0, axis=j+1)
        ar_offset_array[ar_offset_array != 0] = cur_offset
        ar_offsets.append(ar_offset_array)
    return np.stack(ar_offsets, axis=0)


def create_ar_offsets_multichannel(shape : Tuple[int], ar_dim_offsets_per_channel: List[List[List[int]]]) -> np.ndarray:
    """Convert per-channel dimensional offset to per-element pointer offsets (in c++). \
        Could be used as input for ANSEncoder/ANSDecoder. See tests/ans_test.py.

    Args:
        shape (Tuple[int]): Required tensor shape to generate offsets.
        ar_dim_offsets (List[List[int]]): Dimensional offsets.\
            Sub-lists start from channel dimension. e.g. (-1, ) gets [:, c-1, h, w], (0, -1, -1) gets [:, c, h-1, w-1]

    Returns:
        np.ndarray: per-element pointer offsets.
    """
    ar_offsets = []
    for i, ar_dim_offsets in enumerate(ar_dim_offsets_per_channel):
        assert len(ar_dim_offsets) == shape[1]
        ar_offsets_channel = []
        for channel_idx, dim_offsets in enumerate(ar_dim_offsets):
            ar_offset_array = -np.ones(shape, dtype=np.int32)
            cur_offset = 0
            for j, off in enumerate(dim_offsets):
                assert off <= 0
                # use j+1 to skip batch dim
                cur_offset += -off * ar_offset_array.strides[j+1] / ar_offset_array.itemsize
                if off < 0:
                    for k in range(-off):
                        dim_idx = np.zeros_like(np.compress([True], ar_offset_array, axis=j+1)) + k
                        np.put_along_axis(ar_offset_array, dim_idx, 0, axis=j+1)
            ar_offset_array[ar_offset_array != 0] = cur_offset
            ar_offsets_channel.append(ar_offset_array[:, channel_idx])
        ar_offsets.append(np.stack(ar_offsets_channel, axis=1))
    return np.stack(ar_offsets, axis=0)