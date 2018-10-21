from ignite import metrics
from ignite.engine import Events
from ignite.exceptions import NotComputableError


class Identity(metrics.Metric):
    """ Identity metric that updates the state og the engine every iteration, like RunningAverage,
    and not like all the others metrics. This allows to use ProgressBar for displaying the
    training loss """
    def reset(self):
        self.output = None

    def update(self, output):
        self.output = output

    def compute(self):
        return self.output

    def attach(self, engine, name):
        engine.add_event_handler(Events.EPOCH_STARTED, self.started)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.completed, name)


class Average(metrics.Metric):
    """ Average all outputs of the process method """
    def reset(self):
        self.count = 0
        self.sum = 0

    def update(self, output):
        self.count += 1
        self.sum += output

    def compute(self):
        if self.count == 0:
            raise NotComputableError('No iteration')
        return self.sum / self.count
