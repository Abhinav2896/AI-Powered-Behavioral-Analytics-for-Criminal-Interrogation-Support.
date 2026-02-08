from collections import deque

class LandmarkBuffer:
    def __init__(self, size=30):
        self.size = size
        self.buf = deque(maxlen=size)

    def add(self, item):
        self.buf.append(item)

    def window(self):
        return list(self.buf)

    def __len__(self):
        return len(self.buf)
