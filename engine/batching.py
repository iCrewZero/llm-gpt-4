class BatchScheduler:
    def __init__(self):
        self.queue = []

    def add(self, request):
        self.queue.append(request)

    def batch(self):
        return torch.cat(self.queue, dim=0)
