# copied from https://github.com/BlackHC/progress_bar/blob/master/src/blackhc/progress_bar/progress_bar.py
import abc
import time
import sys
from tqdm.auto import tqdm

use_tqdm = None

# From https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook.
def _isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def create_progress_bar(length, tqdm_args=None):
    if use_tqdm is not None:
        local_use_tqdm = use_tqdm
    else:
        local_use_tqdm = sys.stdout.isatty() or _isnotebook()

    if local_use_tqdm:
        return TQDMProgressBar(length, tqdm_args)
    else:
        return LogFriendlyProgressBar(length)


# TODO(blackhc): detect Jupyter notebooks/Ipython as use TQDM
class ProgressBarIterable:
    def __init__(
        self, iterable, length=None, length_unit=None, unit_scale=None, tqdm_args=None
    ):
        self.iterable = iterable
        self.tqdm_args = tqdm_args

        self.length = length
        if length is not None:
            self.length_unit = length_unit
        else:
            if length_unit is not None:
                raise AssertionError(
                    "Cannot specify length_unit without custom length!"
                )
            self.length_unit = 1

        self.unit_scale = unit_scale or 1

    def __iter__(self):
        if self.length is not None:
            length = self.length
        else:
            try:
                length = len(self.iterable)
            except (TypeError, AttributeError):
                raise NotImplementedError("Need a total number of iterations!")

        progress_bar = create_progress_bar(length * self.unit_scale, self.tqdm_args)
        progress_bar.start()
        for item in self.iterable:
            yield item
            progress_bar.update(self.length_unit * self.unit_scale)
        progress_bar.finish()


def with_progress_bar(
    iterable, length=None, length_unit=None, unit_scale=None, tqdm_args=None
):
    return ProgressBarIterable(
        iterable,
        length=length,
        length_unit=length_unit,
        unit_scale=unit_scale,
        tqdm_args=tqdm_args,
    )

class ProgressBar(abc.ABC):
    def __init__(self, length):
        self.length = length

    @abc.abstractmethod
    def start(self):
        pass

    @abc.abstractmethod
    def update(self, delta_processed=1):
        pass

    @abc.abstractmethod
    def finish(self):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

class TQDMProgressBar(ProgressBar):
    def __init__(self, length, tqdm_args=None):
        super().__init__(length)

        self.progress_bar = None
        self.tqdm_args = tqdm_args or {}

    def start(self):
        if self.progress_bar is not None:
            raise AssertionError("start can only be called once!")

        self.progress_bar = tqdm(total=self.length, **self.tqdm_args)

    def update(self, delta_processed=1):
        self.progress_bar.update(delta_processed)

    def finish(self):
        self.progress_bar.close()
        self.progress_bar = None

    def reset(self):
        self.progress_bar.reset()


class LogFriendlyProgressBar(ProgressBar):
    num_sections = 10
    last_flush = 0

    def __init__(self, length):
        super().__init__(length)

        self.start_time = None
        self.last_time = None
        self.num_processed = 0
        self.num_finished_sections = 0

    def start(self):
        if self.start_time is not None:
            raise AssertionError("start can only be called once!")

        self.start_time = self.get_time()
        self.last_time = self.start_time

        self.print_header(self.length)

    @staticmethod
    def get_time():
        return time.time()

    def update(self, delta_processed=1):
        self.num_processed += delta_processed

        while (
            self.num_processed
            >= self.length * (self.num_finished_sections + 1) / self.num_sections
        ):
            self.num_finished_sections += 1
            cur_time = self.get_time()
            elapsed_time = cur_time - self.start_time

            expected_time = (
                elapsed_time * self.num_sections / self.num_finished_sections
            )
            remaining_time = expected_time - elapsed_time

            self.print_section(elapsed_time, remaining_time)

            self.last_time = cur_time

        if self.num_finished_sections == self.num_sections:
            total_time = self.last_time - self.start_time + 0.000001
            ips = self.length / total_time

            self.print_finish(ips, total_time)
            self.start_time = None

    def finish(self):
        remaining_elements = self.length - self.num_processed
        if remaining_elements > 0:
            self.update(remaining_elements)

    def reset(self):
        if self.num_processed == 0:
            return

        LogFriendlyProgressBar.print()
        LogFriendlyProgressBar.print("PROGRESS BAR RESET")

        self.start_time = None
        self.last_time = None
        self.num_processed = 0
        self.num_finished_sections = 0

        self.start()

    @staticmethod
    def print_header(num_iterations):
        LogFriendlyProgressBar.print(f"{num_iterations} iterations:")

        LogFriendlyProgressBar.print(
            "|"
            + "|".join(
                f'{f"{int((index + 1) * 100 / LogFriendlyProgressBar.num_sections)}%":^11}'
                for index in range(LogFriendlyProgressBar.num_sections)
            )
            + "|"
        )
        LogFriendlyProgressBar.print("|", end="")

    @staticmethod
    def print_section(elapsed_time, remaining_time):
        elapsed_time = f"{int(elapsed_time)}s"
        remaining_time = f"{int(remaining_time)}s"
        LogFriendlyProgressBar.print(f"{elapsed_time:<5}<{remaining_time:>5}|", end="")

    @staticmethod
    def print_finish(ips, total_time):
        LogFriendlyProgressBar.print()
        if ips >= 0.1:
            LogFriendlyProgressBar.print(f"{ips:.2f}it/s total: {total_time:.2f}s")
        else:
            LogFriendlyProgressBar.print(f"total: {total_time:.2f}s {1 / ips:.2f}s/it")

    @staticmethod
    def print(text="", end="\n"):
        print(text, end=end)

        cur_time = LogFriendlyProgressBar.get_time()
        if cur_time - LogFriendlyProgressBar.last_flush > 2:
            sys.stdout.flush()
            LogFriendlyProgressBar.last_flush = cur_time
