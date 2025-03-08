import struct
import io
from typing import List, Tuple

def _get_bytes_format(num_bytes=4):
    if num_bytes == 1:
        len_format = "B"
    elif num_bytes == 2:
        len_format = "H"
    elif num_bytes == 4:
        len_format = "I"
    elif num_bytes == 8:
        len_format = "L"
    else:
        raise ValueError("")
    return len_format


def merge_bytes(data: List[bytes], num_bytes_length=4, num_segments=None) -> bytes:
    stream = io.BytesIO(b"")
    len_format = _get_bytes_format(num_bytes_length)
    for i, bs in enumerate(data):
        len_bytes = struct.pack(len_format, len(bs))
        # no need to write length if num_segments is known!
        if num_segments is not None:
            assert(i < num_segments), "Number of segments exceed predefined {}".format(num_segments)
            if i < num_segments - 1:
                stream.write(len_bytes)
        else:
            stream.write(len_bytes)
        stream.write(bs)
    stream.flush()
    return stream.getvalue()

def split_merged_bytes(data: bytes, num_bytes_length=4, num_segments=None) -> List[bytes]:
    stream = io.BytesIO(data)
    len_format = _get_bytes_format(num_bytes_length)
    byte_strings = []
    while (stream.tell() < len(data)):
        if num_segments is not None and len(byte_strings) >= num_segments - 1:
            byte_string = stream.read()
        else:
            len_bytes = stream.read(num_bytes_length)
            length_stream = struct.unpack(len_format, len_bytes)[0]
            byte_string = stream.read(length_stream)
        byte_strings.append(byte_string)
    return byte_strings

def encode_shape(shape : Tuple[int]) -> bytes:
    assert len(shape) < (1 << 8)
    byte_strings = []
    byte_strings.append(struct.pack("B", len(shape)))
    for dim in shape:
        assert(dim < (1 << 16))
        byte_strings.append(struct.pack("<H", dim))
    return b''.join(byte_strings)

def decode_shape(byte_string : bytes) -> Tuple[Tuple[int], int]:
    num_shape_dims = struct.unpack("B", byte_string[:1])[0]
    flat_shape = []
    byte_ptr = 1
    for _ in range(num_shape_dims):
        flat_shape.append(struct.unpack("<H", byte_string[byte_ptr:(byte_ptr+2)])[0])
        byte_ptr += 2

    return flat_shape, byte_ptr