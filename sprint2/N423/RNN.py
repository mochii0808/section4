import numpy as np


# RNN class 구현

class RNN:
  def __init__(self, Wx, Wh, b):
    self.params = [Wx, Wh, b]
    self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
    self.cache = None

  def forward(self, x, h_prev):
    Wx, Wh, b = self.params

    # RNN 셀 기능 구현
    t = np.matmul(h_prev, Wh) + np.matmul(x, Wx) + b 

    h_next = np.tanh(t)

    self.cache = (x, h_prev, h_next)
    return h_next
  


