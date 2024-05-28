import time

class Timer:

  def __init__(self):
    self._start = time.time()

  def __call__(self, reset=True):
    now = time.time()
    diff = now - self._start
    if reset:
      self._start = now
    return diff


class Stat:
  def __init__(self):

    self.total = 0.0
    self.count = 0.0
    
  def __call__(self, val):
    self.total += val
    self.count += 1
    return self.total / self.count
