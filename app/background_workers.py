"""The two long jobs that must not run on the GUI thread: analysing an image
and retraining the pattern model.

Both report back by signal only. Whoever starts one MUST retire it through
MainWindow._retire_worker — dropping the last reference to a still-running
QThread is a hard Qt abort, and it was the cause of three recorded crashes.
"""
from __future__ import annotations

import threading
import traceback

from PySide6 import QtCore

from analyze import analyze_image


class Worker(QtCore.QThread):
    done = QtCore.Signal(str, object)
    failed = QtCore.Signal(str, str)

    def __init__(self, path, do_class=True, do_pattern=True):
        super().__init__()
        self.path = path
        self.do_class = do_class
        self.do_pattern = do_pattern

    def run(self):
        try:
            self.done.emit(self.path, analyze_image(
                self.path, do_class=self.do_class, do_pattern=self.do_pattern))
        except Exception:
            self.failed.emit(self.path, traceback.format_exc())


class TrainWorker(QtCore.QThread):
    """Runs the in-app patternnet retraining off the GUI thread.

    Cancellable: `cancel()` (called from the GUI thread) sets a threading.Event
    that the training loop polls between epochs — for the "X" next to Train
    model that undoes an accidental click. The check only happens between
    epochs, so a cancel takes up to one epoch to land, and nothing is written
    to disk until every epoch has completed — a cancelled run cannot leave a
    half-trained model on disk.
    """
    progress = QtCore.Signal(int, int, str)
    done = QtCore.Signal(dict)
    failed = QtCore.Signal(str)
    cancelled = QtCore.Signal()

    def __init__(self, saturation=False):
        super().__init__()
        self.saturation = saturation
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def run(self):
        try:
            import model_pattern_training
            res = model_pattern_training.run_training(
                progress=lambda d, t, ph: self.progress.emit(d, t, ph),
                saturation=self.saturation,
                should_cancel=self._cancel.is_set)
            self.done.emit(res)
        except model_pattern_training.TrainCancelled:
            self.cancelled.emit()
        except Exception as e:
            self.failed.emit(str(e) or traceback.format_exc())


class SaturationWorker(QtCore.QThread):
    """The learning-curve measurement on its own, with no retraining.

    Separate from TrainWorker because the two answer different questions and
    the user should not have to overwrite a good model to ask "would more data
    help?". Same rule applies: MainWindow retires it.
    """
    progress = QtCore.Signal(float, float)
    done = QtCore.Signal(dict)
    failed = QtCore.Signal(str)

    def run(self):
        try:
            import model_pattern_curve
            self.done.emit(model_pattern_curve.run_saturation(
                progress=lambda d, t: self.progress.emit(float(d), float(t))))
        except Exception as e:
            self.failed.emit(str(e) or traceback.format_exc())
