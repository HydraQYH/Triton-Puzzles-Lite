import torch

def tensor_shift(x):
  over = torch.arange(8) * 4
  print(over)
  x = x[..., None] >> over
  print(x.shape)

if __name__ == '__main__':
  tensor_shift(torch.randint(-1000000, 1000000, (16, 8), dtype=torch.int32))
