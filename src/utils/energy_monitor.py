import warnings

warnings.filterwarnings("ignore")
import logging

logging.getLogger("zeus").setLevel(logging.ERROR)
logging.getLogger("pynvml").setLevel(logging.ERROR)

import logging

logging.getLogger("zeus").setLevel(logging.ERROR)

from zeus.monitor import ZeusMonitor

_monitor = None


def get_monitor():
    global _monitor
    if _monitor is None:
        _monitor = ZeusMonitor(gpu_indices=[0])
    return _monitor


class EnergyMonitor:
    """
    Monitor energetico basato su Zeus.
    Misura Joules GPU durante un'operazione.
    """

    def __init__(self, device_index=0):
        self.device_index = device_index
        self._joules = 0.0
        self._monitor = get_monitor()
        self._window = f"op_{id(self)}"

    def start(self):
        self._monitor.begin_window(self._window)

    def stop(self):
        result = self._monitor.end_window(self._window)
        self._joules = result.total_energy

    def joules(self):
        return self._joules

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def measure_baseline(seconds=3):
    """Misura potenza GPU a riposo. Ritorna watt medi."""
    import time

    monitor = get_monitor()
    window = "baseline_measurement"
    monitor.begin_window(window)
    time.sleep(seconds)
    result = monitor.end_window(window)
    return result.total_energy / seconds
