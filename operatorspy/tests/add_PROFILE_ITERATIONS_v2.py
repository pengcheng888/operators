from ctypes import POINTER, Structure, c_int32, c_void_p
import ctypes
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from operatorspy import (
    open_lib,
    to_tensor,
    DeviceEnum,
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    create_handle,
    destroy_handle,
    check_error,
)

from operatorspy.tests.test_utils import get_args
from enum import Enum, auto
import torch

from test_utils import PROFILE, NUM_PRERUN, NUM_ITERATIONS, TOLERANCE_MAP


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_A = auto()
    INPLACE_B = auto()


class AddDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopAddDescriptor_t = POINTER(AddDescriptor)


def add(x, y):
    return torch.add(x, y)


def test(
        lib,
        handle,
        torch_device,
        c_shape,
        a_shape,
        b_shape,
        tensor_dtype=torch.float16,
        inplace=Inplace.OUT_OF_PLACE,
):
    print(
        f"Testing Add on {torch_device} with c_shape:{c_shape} a_shape:{a_shape} b_shape:{b_shape} dtype:{tensor_dtype} inplace: {inplace.name}"
    )
    if a_shape != b_shape and inplace != Inplace.OUT_OF_PLACE:
        print("Unsupported test: broadcasting does not support in-place")
        return

    # ------------------------------------------------------------------------------ #
    #                              准备数据                                            #
    # ------------------------------------------------------------------------------ #
    a = torch.rand(a_shape, dtype=tensor_dtype).to(torch_device)
    b = torch.rand(b_shape, dtype=tensor_dtype).to(torch_device)
    c = torch.rand(c_shape, dtype=tensor_dtype).to(torch_device) if inplace == Inplace.OUT_OF_PLACE else (a if inplace == Inplace.INPLACE_A else b)

    ans = add(a, b)

    a_tensor = to_tensor(a, lib)
    b_tensor = to_tensor(b, lib)
    c_tensor = to_tensor(c, lib) if inplace == Inplace.OUT_OF_PLACE else (a_tensor if inplace == Inplace.INPLACE_A else b_tensor)

    # ------------------------------------------------------------------------------ #
    #                              准备infiniop算子                                    #
    # ------------------------------------------------------------------------------ #
    descriptor = infiniopAddDescriptor_t()
    check_error(
        lib.infiniopCreateAddDescriptor(
            handle,
            ctypes.byref(descriptor),
            c_tensor.descriptor,
            a_tensor.descriptor,
            b_tensor.descriptor,
        )
    )
    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    c_tensor.descriptor.contents.invalidate()
    a_tensor.descriptor.contents.invalidate()
    b_tensor.descriptor.contents.invalidate()

    # ------------------------------------------------------------------------------ #
    #                              先对比精度                                          #
    # ------------------------------------------------------------------------------ #
    def lib_add():
        check_error(lib.infiniopAdd(descriptor, c_tensor.data, a_tensor.data, b_tensor.data, None))

    lib_add()
    torch.cuda.synchronize()
    assert torch.allclose(c, ans, atol=TOLERANCE_MAP[tensor_dtype]["atol"], rtol=TOLERANCE_MAP[tensor_dtype]["rtol"])
    # ------------------------------------------------------------------------------ #
    #                              计算pytorch算子                                     #
    # ------------------------------------------------------------------------------ #
    # Profiling workflow
    if PROFILE:
        # fmt: off
        from infiniop.libinfiniop.utils import profile_operation
        profile_operation("PyTorch", lambda: add(a, b), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_add(), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    # ------------------------------------------------------------------------------ #
    #                              计算pytorch算子                                     #
    # ------------------------------------------------------------------------------ #
    # for i in range(NUM_PRERUN if PROFILE else 1):
    #     ans = add(a, b)
    # # operatorspy/tests/add_PROFILE_ITERATIONS.py
    # if PROFILE:
    #     start_time = time.time()
    #     for i in range(NUM_ITERATIONS):
    #         _ = add(a, b)
    #     elapsed = 1000*(time.time() - start_time) / NUM_ITERATIONS
    #     print(f"pytorch time : {elapsed :6f}  ms")

    # ------------------------------------------------------------------------------ #
    #                              计算infiniop算子                                    #
    # ------------------------------------------------------------------------------ #
    # for i in range(NUM_PRERUN if PROFILE else 1):
    #     check_error(lib.infiniopAdd(descriptor, c_tensor.data, a_tensor.data, b_tensor.data, None))
    #
    # if PROFILE:  # 应该得cuda同步才型把，记录event的时间
    #         start_time = time.time()
    #         for i in range(NUM_ITERATIONS):
    #             check_error(   lib.infiniopAdd(descriptor, c_tensor.data, a_tensor.data, b_tensor.data, None) )
    #
    #         elapsed = 1000*(time.time() - start_time) / NUM_ITERATIONS
    #         print(f"    lib time: {elapsed :6f} ms")

    # ------------------------------------------------------------------------------ #
    #                                 释放资源                                         #
    # ------------------------------------------------------------------------------ #
    check_error(lib.infiniopDestroyAddDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for c_shape, a_shape, b_shape, inplace in test_cases:
        test(lib, handle, "cpu", c_shape, a_shape, b_shape, tensor_dtype=torch.float16, inplace=inplace)
        test(lib, handle, "cpu", c_shape, a_shape, b_shape, tensor_dtype=torch.float32, inplace=inplace)
    destroy_handle(lib, handle)


def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for c_shape, a_shape, b_shape, inplace in test_cases:
        c_shape = [int(val) for val in c_shape]
        a_shape = [int(val) for val in a_shape]
        b_shape = [int(val) for val in b_shape]

        test(lib, handle, "cuda", c_shape, a_shape, b_shape, tensor_dtype=torch.float16, inplace=inplace)
        test(lib, handle, "cuda", c_shape, a_shape, b_shape, tensor_dtype=torch.float32, inplace=inplace)
    destroy_handle(lib, handle)


def test_bang(lib, test_cases):
    import torch_mlu

    device = DeviceEnum.DEVICE_BANG
    handle = create_handle(lib, device)
    for c_shape, a_shape, b_shape, inplace in test_cases:
        test(lib, handle, "mlu", c_shape, a_shape, b_shape, tensor_dtype=torch.float16, inplace=inplace)
        test(lib, handle, "mlu", c_shape, a_shape, b_shape, tensor_dtype=torch.float32, inplace=inplace)
    destroy_handle(lib, handle)


def test_musa(lib, test_cases):
    import torch_musa

    device = DeviceEnum.DEVICE_MUSA
    handle = create_handle(lib, device)
    for c_shape, a_shape, b_shape, inplace in test_cases:
        test(lib, handle, "musa", c_shape, a_shape, b_shape, tensor_dtype=torch.float16, inplace=inplace)
        test(lib, handle, "musa", c_shape, a_shape, b_shape, tensor_dtype=torch.float32, inplace=inplace)
    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases11 = [
        # c_shape, a_shape, b_shape, inplace
        # ((32, 150, 512000), (32, 150, 512000), (32, 150, 512000), Inplace.OUT_OF_PLACE),
        # ((32, 150, 51200), (32, 150, 51200), (32, 150, 1), Inplace.OUT_OF_PLACE),
        # ((32, 150, 51200), (32, 150, 51200), (32, 150, 51200), Inplace.OUT_OF_PLACE),
        # ------------------------------------------------------------------------
        ((1, 3), (1, 3), (1, 3), Inplace.OUT_OF_PLACE),
        ((), (), (), Inplace.OUT_OF_PLACE),
        ((3, 3), (3, 3), (3, 3), Inplace.OUT_OF_PLACE),
        ((2, 20, 3), (2, 1, 3), (2, 20, 3), Inplace.OUT_OF_PLACE),
        ((32, 20, 512), (32, 20, 512), (32, 20, 512), Inplace.INPLACE_A),
        ((32, 20, 512), (32, 20, 512), (32, 20, 512), Inplace.INPLACE_B),
        ((32, 256, 112, 112), (32, 256, 112, 1), (32, 256, 112, 112), Inplace.OUT_OF_PLACE),
        ((32, 256, 112, 112), (32, 256, 112, 112), (32, 256, 112, 112), Inplace.OUT_OF_PLACE),
        ((2, 4, 3), (2, 1, 3), (4, 3), Inplace.OUT_OF_PLACE),
        ((2, 3, 4, 5), (2, 3, 4, 5), (5,), Inplace.OUT_OF_PLACE),
        ((3, 2, 4, 5), (4, 5), (3, 2, 1, 1), Inplace.OUT_OF_PLACE),
    ]

    test_cases = [
        # c_shape, a_shape, b_shape, inplace
        # ((32, 150, 512000), (32, 150, 512000), (32, 150, 512000), Inplace.OUT_OF_PLACE),
        # ((32, 150, 51200), (32, 150, 51200), (32, 150, 1), Inplace.OUT_OF_PLACE),
        # ((32, 150, 51200), (32, 150, 51200), (32, 150, 51200), Inplace.OUT_OF_PLACE),
        # ------------------------------------------------------------------------
        # ((1e1, 10, 10), (1e1, 10, 10), (1e1, 10, 10), Inplace.INPLACE_A),
        # ((1e2, 10, 10), (1e2, 10, 10), (1e2, 10, 10), Inplace.INPLACE_A),
        # ((1e3, 10, 10), (1e3, 10, 10), (1e3, 10, 10), Inplace.INPLACE_A),
        # ((1e4, 10, 10), (1e4, 10, 10), (1e4, 10, 10), Inplace.INPLACE_A),
        # ((1e5, 10, 10), (1e5, 10, 10), (1e5, 10, 10), Inplace.INPLACE_A),
        # ((1e6, 10, 10), (1e6, 10, 10), (1e6, 10, 10), Inplace.INPLACE_A),
        ((2e6, 10, 10), (2e6, 10, 10), (2e6, 10, 10), Inplace.INPLACE_A),
    ]

    args = get_args()
    lib = open_lib()
    lib.infiniopCreateAddDescriptor.restype = c_int32
    lib.infiniopCreateAddDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopAddDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopAdd.restype = c_int32
    lib.infiniopAdd.argtypes = [
        infiniopAddDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyAddDescriptor.restype = c_int32
    lib.infiniopDestroyAddDescriptor.argtypes = [
        infiniopAddDescriptor_t,
    ]
    PROFILE = args.profile
    if args.cpu:
        test_cpu(lib, test_cases)
    if args.cuda:
        test_cuda(lib, test_cases)
    if args.bang:
        test_bang(lib, test_cases)
    if args.musa:
        test_musa(lib, test_cases)
    if not (args.cpu or args.cuda or args.bang or args.musa):
        test_cpu(lib, test_cases)
    print("\033[92mTest passed!\033[0m")
