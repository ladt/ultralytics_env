from ultralytics.utils.metrics import ClassifyMetrics as _ClassifyMetrics

class ClassifyMetrics(_ClassifyMetrics):

    @property
    def fitness(self):
        """Return top-1 accuracy as fitness score."""
        return self.top1
