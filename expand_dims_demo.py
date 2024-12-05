import torch
import triton
import triton.language as tl

@triton.jit
def expand_dims_demo(x_ptr):
  row_idx = tl.arange(0, 8).expand_dims(1).broadcast_to(8, 4)
  col_idx = tl.arange(0, 4).expand_dims(0).broadcast_to(8, 4)
  matrix_idx = row_idx + col_idx
  print(matrix_idx.shape)
  print(matrix_idx.expand_dims([0, 2, 4]).shape)

def run_demo():
  print("Demo1 Output: ")
  expand_dims_demo[(1, 1, 1)](torch.ones(4, 3))

if __name__ == '__main__':
  run_demo()
