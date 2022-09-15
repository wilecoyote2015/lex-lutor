from PySide6.QtCore import (Property, QObject, QPropertyAnimation, Signal, QThread, Slot)


class JobQueue(QObject):

    def __init__(self, cls_worker, slot_finished, args_worker_static=None, parent=None):
        super().__init__(parent)

        self.queue: [QThread, QObject, str] = []

        self.cls_worker = cls_worker
        self.slot_finished = slot_finished
        self.args_worker_static = args_worker_static if args_worker_static is not None else []

    @Slot()
    def start_next_job(self):
        if self.queue and not self.queue[-1][0].isFinished():
            self.queue = self.queue[-1:]
            self.queue[-1][0].start()
        else:
            self.queue = []

    @Slot()
    def start_job(self, *args_worker):
        thread, worker = QThread(), self.cls_worker(*self.args_worker_static, *args_worker)
        self.queue.append((thread, worker, str(id(worker))))

        worker.moveToThread(thread)
        worker.finished.connect(thread.quit)
        # worker.finished.connect(worker.deleteLater)
        worker.finished.connect(self.slot_finished)
        # thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self.start_next_job)
        thread.started.connect(worker.run)

        if len(self.queue) == 1:
            self.start_next_job()
