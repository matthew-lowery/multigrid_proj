import triton
import triton.language as tl
import torch


dtype = torch.float32
device = 'cuda:0'


@triton.jit
def conv1d_kernel(
    input_ptr, 
    kernel_ptr, 
    output_ptr,
    input_row_stride,
    kernel_row_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr
    ):

    hid = tl.program_id(0)

    output_offset = output_ptr + output_row_stride * hid
    input_offset = input_ptr + input_row_stride * hid

    accum = 0.

    for i in range(3):
        input_idx = input_offset + input_row_stride * i
        kernel_idx = kernel_ptr + kernel_row_stride * i
        input_val = tl.load(input_idx)
        kernel_val = tl.load(kernel_idx)
        accum += (input_val + kernel_val)

    tl.store(output_offset, accum)


def conv1d_triton(
    input: torch.Tensor,
    kernel: torch.Tensor,
) -> torch.Tensor:
    assert input.is_cuda and kernel.is_cuda, 'Input or kernel is not on GPU'
    input = input.contiguous()  # force memory layout
    kernel = kernel.contiguous()
    n = input.shape[0]
    d = 3
    input_pad = torch.nn.functional.pad(input, (1, 1), mode='constant', value=0)
    output = torch.empty(input.shape, device=device, dtype=dtype)
    grid = (len(input),)
    conv1d_kernel[grid](input,kernel,output, input.stride(0), kernel.stride(0),output.stride(0),d,)
    return output


@triton.jit
def conv3d_kernel(
    input_ptr, 
    kernel_ptr, 
    output_ptr,
    input_row_stride,
    input_col_stride,
    input_dep_stride,
    kernel_row_stride,
    kernel_col_stride,
    kernel_dep_stride,
    output_row_stride,
    output_col_stride,
    output_dep_stride,
    BLOCK_SIZE: tl.constexpr
    ):

    hid = tl.program_id(0)
    wid = tl.program_id(1)
    did = tl.program_id(2)


    output_offset = output_ptr + (output_row_stride * hid + output_col_stride * wid + output_dep_stride * did)

    input_offset = input_ptr + (input_row_stride * hid + input_col_stride * wid + input_dep_stride * did)

    accum = 0.

    for i in range(3):
        for j in range(3):
            for k in range(3):
                input_idx = input_offset + (input_row_stride * i + input_col_stride * j + input_dep_stride * k)
                kernel_idx = kernel_ptr + (kernel_row_stride * i + kernel_col_stride * j + kernel_dep_stride * k)
                input_val = tl.load(input_idx)
                kernel_val = tl.load(kernel_idx)
                accum += (input_val + kernel_val)
    tl.store(output_offset, accum)


def conv3d_triton(
    input: torch.Tensor,
    kernel: torch.Tensor,
) -> torch.Tensor:
    assert input.is_cuda and kernel.is_cuda, 'Input or kernel is not on GPU'

    n = input.shape[0]
    d = 3
    out_size = n - d + 1
    output = torch.empty((out_size, out_size, out_size), device=device, dtype=dtype)

    grid = (out_size, out_size, out_size)

    conv3d_kernel[grid](input,kernel,output,
                        input.stride(0), input.stride(1), input.stride(2),
                        kernel.stride(0), kernel.stride(1), kernel.stride(2),
                        output.stride(0), output.stride(1), output.stride(2), 9,)

    return output

