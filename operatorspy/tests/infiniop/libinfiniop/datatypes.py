import ctypes
from ctypes import c_ushort


class DataLayout(ctypes.LittleEndianStructure):
    _fields_ = [
        ("packed", c_ushort, 8),
        ("sign", c_ushort, 1),
        ("size", c_ushort, 7),
        ("mantissa", c_ushort, 8),
        ("exponent", c_ushort, 8),
    ]


I8 = DataLayout(1, 1, 1, 7, 0)
I16 = DataLayout(1, 1, 2, 15, 0)
I32 = DataLayout(1, 1, 4, 31, 0)
I64 = DataLayout(1, 1, 8, 63, 0)
U8 = DataLayout(1, 0, 1, 8, 0)
U16 = DataLayout(1, 0, 2, 16, 0)
U32 = DataLayout(1, 0, 4, 32, 0)
U64 = DataLayout(1, 0, 8, 64, 0)
F16 = DataLayout(1, 1, 2, 10, 5)
BF16 = DataLayout(1, 1, 2, 7, 8)
F32 = DataLayout(1, 1, 4, 23, 8)
F64 = DataLayout(1, 1, 8, 52, 11)

class InfiniDtype:
    INVALID = 0
    BYTE = 1
    BOOL = 2
    I8 = 3
    I16 = 4
    I32 = 5
    I64 = 6
    U8 = 7
    U16 = 8
    U32 = 9
    U64 = 10
    F8 = 11
    F16 = 12
    F32 = 13
    F64 = 14
    C8 = 15
    C16 = 16
    C32 = 17
    C64 = 18
    BF16 = 19
