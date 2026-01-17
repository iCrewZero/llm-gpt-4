class ContinuousBatcher:
    def __init__(self, max_batch):
        self.max_batch = max_batch
        self.queue = []

    def add(self, req):
        self.queue.append(req)

    def next_batch(self):
        batch = self.queue[:self.max_batch]
        self.queue = self.queue[self.max_batch:]
        return batch
