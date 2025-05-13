from ctypes import POINTER, Structure, c_int32, c_uint64, c_void_p
import ctypes
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from operatorspy import (
    open_lib,
    to_tensor,
    CTensor,
    DeviceEnum,
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    create_handle,
    destroy_handle,
    check_error,
    rearrange_tensor,
)

from operatorspy.tests.test_utils import get_args
import torch
from enum import Enum, auto
from test_utils import PROFILE, NUM_PRERUN, NUM_ITERATIONS, TOLERANCE_MAP


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_A = auto()
    INPLACE_B = auto()


class SwiGLUDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopSwiGLUDescriptor_t = POINTER(SwiGLUDescriptor)


def swiglu(a, b):
    return a * b / (1 + torch.exp(-b.float()).to(b.dtype))


def test(lib,
         handle,
         torch_device,
         shape,
         a_stride=None,
         b_stride=None,
         c_stride=None,
         dtype=torch.float16,
         inplace=Inplace.OUT_OF_PLACE,
         sync=None, ):
    print(f"Testing SwiGLU on {torch_device} with shape:{shape} a_stride:{a_stride} b_stride:{b_stride} c_stride:{c_stride} dtype:{dtype}  inplace:{inplace}")

    # ------------------------------------------------------------------------------ #
    #                              准备数据                                           #
    # ------------------------------------------------------------------------------ #
    a = torch.rand(shape, dtype=dtype).to(torch_device)
    b = torch.rand(shape, dtype=dtype).to(torch_device)
    c = torch.rand(shape, dtype=dtype).to(torch_device) if inplace == Inplace.OUT_OF_PLACE else (a if inplace == Inplace.INPLACE_A else b)

    if a_stride is not None:
        a = rearrange_tensor(a, a_stride)
    if b_stride is not None:
        b = rearrange_tensor(b, b_stride)
    if c_stride is not None:
        c = rearrange_tensor(c, c_stride)

    ans = swiglu(a, b)

    if sync is not None:
        sync()

    a_tensor = to_tensor(a, lib)
    b_tensor = to_tensor(b, lib)
    c_tensor = to_tensor(c, lib) if inplace == Inplace.OUT_OF_PLACE else (a_tensor if inplace == Inplace.INPLACE_A else b_tensor)

    # ------------------------------------------------------------------------------ #
    #                              准备infiniop算子                                    #
    # ------------------------------------------------------------------------------ #
    descriptor = infiniopSwiGLUDescriptor_t()
    check_error(lib.infiniopCreateSwiGLUDescriptor(handle,
                                                   ctypes.byref(descriptor),
                                                   c_tensor.descriptor,
                                                   a_tensor.descriptor,
                                                   b_tensor.descriptor)
                )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    c_tensor.descriptor.contents.invalidate()
    a_tensor.descriptor.contents.invalidate()
    b_tensor.descriptor.contents.invalidate()

    # ------------------------------------------------------------------------------ #
    #                              先对比精度                                          #
    # ------------------------------------------------------------------------------ #
    def lib_SwiGLU():
        check_error(lib.infiniopSwiGLU(descriptor, c_tensor.data, a_tensor.data, b_tensor.data, None))


    lib_SwiGLU()
    torch.cuda.synchronize()
    c = c if inplace == Inplace.OUT_OF_PLACE else (a if inplace == Inplace.INPLACE_A else b)

    assert torch.allclose(c, ans, atol=TOLERANCE_MAP[dtype]["atol"], rtol=TOLERANCE_MAP[dtype]["rtol"])
    print("out-of-place Test passed!")
    # ------------------------------------------------------------------------------ #
    #                              计算pytorch算子                                     #
    # ------------------------------------------------------------------------------ #
    # Profiling workflow
    if PROFILE:
        # fmt: off
        from infiniop.libinfiniop.utils import profile_operation
        profile_operation("PyTorch", lambda: swiglu(a, b), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_SwiGLU(), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    # ------------------------------------------------------------------------------ #
    #                              计算pytorch算子                                     #
    # ------------------------------------------------------------------------------ #
    # for i in range(NUM_PRERUN if PROFILE else 1):
    #     _ = swiglu(a, b)
    #
    # if PROFILE:
    #     start_time = time.time()
    #     for i in range(NUM_ITERATIONS):
    #         _ = swiglu(a, b)
    #     elapsed = 1000 * (time.time() - start_time) / NUM_ITERATIONS
    #     print(f"pytorch time : {elapsed :6f}  ms")

    # ------------------------------------------------------------------------------ #
    #                              计算infiniop算子                                    #
    # ------------------------------------------------------------------------------ #
    # for i in range(NUM_PRERUN if PROFILE else 1):
    #     check_error(lib.infiniopSwiGLU(descriptor, c_tensor.data, a_tensor.data, b_tensor.data, None))
    #
    # if PROFILE:  # 应该得cuda同步才型把，记录event的时间
    #     start_time = time.time()
    #     for i in range(NUM_ITERATIONS):
    #         check_error(lib.infiniopSwiGLU(descriptor, c_tensor.data, a_tensor.data, b_tensor.data, None))
    #
    #     elapsed = 1000 * (time.time() - start_time) / NUM_ITERATIONS
    #     print(f"    lib time: {elapsed :6f} ms")

    # ------------------------------------------------------------------------------ #
    #                                 释放资源                                         #
    # ------------------------------------------------------------------------------ #
    check_error(lib.infiniopDestroySwiGLUDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)

    for shape, a_stride, b_stride, c_stride, dtype in test_cases:
        test_out_of_place(
            lib, handle, "cpu", shape, a_stride, b_stride, c_stride, dtype
        )
        # test_in_place1(lib, handle, "cpu", shape, a_stride, b_stride, dtype)
        # test_in_place2(lib, handle, "cpu", shape, a_stride, b_stride, dtype)

    destroy_handle(lib, handle)


def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)

    for shape, a_stride, b_stride, c_stride, dtype, inplace in test_cases:
        test(lib, handle, "cuda", shape, a_stride, b_stride, c_stride, dtype, inplace)

    destroy_handle(lib, handle)


def test_bang(lib, test_cases):
    import torch_mlu
    device = DeviceEnum.DEVICE_BANG
    handle = create_handle(lib, device)

    for shape, a_stride, b_stride, c_stride, dtype in test_cases:
        test_out_of_place(
            lib, handle, "mlu", shape, a_stride, b_stride, c_stride, dtype
        )
        test_in_place1(lib, handle, "mlu", shape, a_stride, b_stride, dtype)
        test_in_place2(lib, handle, "mlu", shape, a_stride, b_stride, dtype)

    destroy_handle(lib, handle)


def test_ascend(lib, test_cases):
    import torch_npu
    device = DeviceEnum.DEVICE_ASCEND
    handle = create_handle(lib, device)

    for shape, a_stride, b_stride, c_stride, dtype in test_cases:
        test_out_of_place(
            lib, handle, "npu", shape, a_stride, b_stride, c_stride, dtype, torch.npu.synchronize
        )
        test_in_place1(lib, handle, "npu", shape, a_stride, b_stride, dtype, torch.npu.synchronize)
        test_in_place2(lib, handle, "npu", shape, a_stride, b_stride, dtype, torch.npu.synchronize)

    destroy_handle(lib, handle)


def test_maca(lib, test_cases):
    device = DeviceEnum.DEVICE_MACA
    handle = create_handle(lib, device)

    for shape, a_stride, b_stride, c_stride, dtype in test_cases:
        test_out_of_place(
            lib, handle, "cuda", shape, a_stride, b_stride, c_stride, dtype)
        test_in_place1(lib, handle, "cuda", shape, a_stride, b_stride, dtype)
        test_in_place2(lib, handle, "cuda", shape, a_stride, b_stride, dtype)

    destroy_handle(lib, handle)


def test_musa(lib, test_cases):
    import torch_musa
    device = DeviceEnum.DEVICE_MUSA
    handle = create_handle(lib, device)

    for shape, a_stride, b_stride, c_stride, dtype in test_cases:
        test_out_of_place(
            lib, handle, "musa", shape, a_stride, b_stride, c_stride, dtype
        )
        test_in_place1(lib, handle, "musa", shape, a_stride, b_stride, dtype)
        test_in_place2(lib, handle, "musa", shape, a_stride, b_stride, dtype)

    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # shape, a_stride, b_stride, c_stride, dtype
        ((13, 4), None, None, None, torch.float16, Inplace.OUT_OF_PLACE),
        ((13, 4), (10, 1), (10, 1), (10, 1), torch.float16, Inplace.INPLACE_A),
        ((16, 5632), None, None, None, torch.float16, Inplace.INPLACE_B),
        ((16, 5632), (13312, 1), (13312, 1), (13312, 1), torch.float16, Inplace.OUT_OF_PLACE),
    ]
    args = get_args() # swiglu_PROFILE_ITERATIONS
    lib = open_lib()

    lib.infiniopCreateSwiGLUDescriptor.restype = c_int32
    lib.infiniopCreateSwiGLUDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopSwiGLUDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopSwiGLU.restype = c_int32
    lib.infiniopSwiGLU.argtypes = [
        infiniopSwiGLUDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroySwiGLUDescriptor.restype = c_int32
    lib.infiniopDestroySwiGLUDescriptor.argtypes = [
        infiniopSwiGLUDescriptor_t,
    ]
    PROFILE = args.profile
    if args.cpu:
        test_cpu(lib, test_cases)
    if args.cuda:
        test_cuda(lib, test_cases)
    if args.bang:
        test_bang(lib, test_cases)
    if args.ascend:
        test_ascend(lib, test_cases)
    if args.maca:
        test_maca(lib, test_cases)
    if args.musa:
        test_musa(lib, test_cases)
    print("\033[92mTest passed!\033[0m")
