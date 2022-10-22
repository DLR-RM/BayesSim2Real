import collections
import pickle

import ignite
from ignite.engine import Engine, Events
import torch.utils.data as data
import torch

from src.utils.progress_bar import create_progress_bar

class RestoringScoreGuard(object):
    """RestoringScoreGuard handler can be used to stop the training if no improvement after a given number of events

    Args:
        patience (int):
            Number of events to wait if no improvement and then stop the training
        score_function (Callable):
            It should be a function taking a single argument, an `ignite.engine.Engine` object,
            and return a score `float`. An improvement is considered if the score is higher.
        trainer (Engine):
            trainer engine to stop the run if no improvement

    Examples:

    .. code-block:: python

        from ignite.engine import Engine, Events
        from ignite.handlers import EarlyStopping

        def score_function(engine):
            val_loss = engine.state.metrics['nll']
            return -val_loss

        handler = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)
        # Note: the handler is attached to an *Evaluator* (runs one epoch on validation dataset)
        evaluator.add_event_handler(Events.COMPLETED, handler)

    """

    def __init__(
        self,
        *,
        patience,
        score_function,
        out_of_patience_callback,
        training_engine: Engine,
        validation_engine: Engine,
        module: torch.nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
    ):

        if not callable(score_function):
            raise TypeError("Argument score_function should be a function")

        if patience < 1:
            raise ValueError("Argument patience should be positive integer")

        self.score_function = score_function
        self.out_of_patience_callback = out_of_patience_callback
        self.module = module
        self.optimizer = optimizer

        self.patience = patience
        self.counter = 0

        self.best_score = None
        self.best_module_state_dict = None
        self.best_optimizer_state_dict = None
        self.restore_epoch = None

        self.training_engine = training_engine
        self.validation_engine = validation_engine
        validation_engine.add_event_handler(Events.EPOCH_COMPLETED, self.on_epoch_completed)
        training_engine.add_event_handler(Events.COMPLETED, self.on_completed)

    def snapshot(self):
        if self.module is not None:
            self.best_module_state_dict = pickle.dumps(self.module.state_dict(keep_vars=False))
        if self.optimizer is not None:
            self.best_optimizer_state_dict = pickle.dumps(self.optimizer.state_dict())

    def restore_best(self):
        if self.best_module_state_dict is not None and self.module is not None:
            print(f"RestoringScoreGuard: Restoring best parameters. (Score: {self.best_score})")
            self.module.load_state_dict(pickle.loads(self.best_module_state_dict))

        if self.best_optimizer_state_dict is not None and self.optimizer is not None:
            print("RestoringScoreGuard: Restoring optimizer.")
            self.optimizer.load_state_dict(pickle.loads(self.best_optimizer_state_dict))

    def on_epoch_completed(self, _):
        score = self.score_function(self.validation_engine)

        if self.best_score is not None and score <= self.best_score:
            self.counter += 1
            print("RestoringScoreGuard: %i / %i" % (self.counter, self.patience))
            if self.counter >= self.patience:
                print("RestoringScoreGuard: Out of patience")
                self.restore_best()
                self.restore_epoch = self.training_engine.state.epoch

                # Reset the counter in case we keep training after adjusting the model.
                self.counter = 0
                self.out_of_patience_callback()
        else:
            self.best_score = score
            self.snapshot()
            self.counter = 0

    def on_completed(self, _):
        if self.restore_epoch is None or self.restore_epoch < self.training_engine.state.epoch:
            self.restore_best()
            self.restore_epoch = self.training_engine.state.epoch

class IgniteProgressBar(object):
    def __init__(self, desc, log_interval):
        self.log_interval = log_interval
        self.desc = desc
        self.progress_bar = None

    def attach(self, engine: ignite.engine.Engine):
        engine.add_event_handler(ignite.engine.Events.EPOCH_STARTED, self.on_start)
        engine.add_event_handler(ignite.engine.Events.EPOCH_COMPLETED, self.on_complete)
        engine.add_event_handler(ignite.engine.Events.ITERATION_COMPLETED, self.on_iteration_complete)

    def on_start(self, engine):
        dataloader = engine.state.dataloader
        self.progress_bar = create_progress_bar(len(dataloader) * dataloader.batch_size)

        print(self.desc(engine))
        self.progress_bar.start()

    def on_complete(self, engine):
        self.progress_bar.finish()

    def on_iteration_complete(self, engine):
        dataloader = engine.state.dataloader
        iter = (engine.state.iteration - 1) % len(dataloader) + 1

        if iter % self.log_interval == 0:
            self.progress_bar.update(self.log_interval * dataloader.batch_size)


def ignite_progress_bar(engine: ignite.engine.Engine, desc=None, log_interval=0):
    wrapper = IgniteProgressBar(desc, log_interval)
    wrapper.attach(engine)

    return wrapper


def epoch_chain(engine: ignite.engine.Engine, then_engine: ignite.engine.Engine, dataloader: data.DataLoader):
    @engine.on(ignite.engine.Events.EPOCH_COMPLETED)
    def on_complete(_):
        then_engine.run(dataloader)


def chain(engine: ignite.engine.Engine, then_engine: ignite.engine.Engine, dataloader: data.DataLoader):
    @engine.on(ignite.engine.Events.COMPLETED)
    def on_complete(_):
        then_engine.run(dataloader)

def chain_save_model(engine: ignite.engine.Engine, model: torch.nn.Module, save_path):
    @engine.on(ignite.engine.Events.COMPLETED)
    def on_complete(_):
        torch.save(model.state_dict(), save_path)

def log_epoch_results(engine: ignite.engine.Engine, name, trainer: ignite.engine.Engine):
    @engine.on(ignite.engine.Events.COMPLETED)
    def log(_):
        metrics = engine.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_nll = metrics["nll"]
        print(
            f"{name} Results - Epoch: {trainer.state.epoch if trainer.state else 0}  "
            f"Avg accuracy: {avg_accuracy*100:.2f}% Avg loss: {avg_nll:.2f}"
        )


def log_results(engine: ignite.engine.Engine, name):
    @engine.on(ignite.engine.Events.COMPLETED)
    def log(_):
        metrics = engine.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_nll = metrics["nll"]
        print(f"{name} Results  " f"Avg accuracy: {avg_accuracy*100:.2f}% Avg loss: {avg_nll:.2f}")


def store_iteration_results(engine: ignite.engine.Engine, store_object, log_interval=1000):
    @engine.on(ignite.engine.Events.ITERATION_COMPLETED)
    def log(_):
        if engine.state.iteration % log_interval == 0:
            store_object.append(engine.state.output)


def store_epoch_results(engine: ignite.engine.Engine, store_object, name=None):
    @engine.on(ignite.engine.Events.EPOCH_COMPLETED)
    def log(_):
        metrics = engine.state.metrics
        if isinstance(store_object, collections.MutableSequence):
            store_object.append(metrics)
        else:
            store_object[name] = metrics
