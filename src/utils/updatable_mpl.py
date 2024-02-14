from collections import defaultdict
from typing import Any, List, Tuple, Dict, Iterable, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


class Memoriser:
    """
    Keep track of the added values
    """

    def __init__(self):
        self._values: List[float] = []

    def add(self, val: Any) -> List[Any]:
        self._values.append(val)
        return self.values

    @property
    def sequence(self) -> Iterable[float]:
        return range(len(self.values))

    @property
    def values(self) -> List[Any]:
        return self._values


class AxesLimit:
    def __init__(self):
        self.lower: float = None
        self.upper: float = None

    def set(self, lower, upper):
        if self.lower is None or lower < self.lower:
            self.lower = lower
        if self.upper is None or upper > self.upper:
            self.upper = upper


class UpdatablePlot:
    UniqueStoringKey = Tuple[int, str]

    def __init__(self, nclos: int = 1, nrows: int = 1, auto_update: bool = True):
        # to run GUI event loop
        plt.ion()
        self.stored_artist: Dict[UpdatablePlot.UniqueStoringKey, Line2D] = dict()
        self.figure: Figure
        self.axs: Union[Axes, List[Axes]]
        self.figure, self.axs = plt.subplots(
            figsize=(10, 8), ncols=nclos, nrows=nrows, sharex=True
        )
        self.auto_update = auto_update
        # self.ylim: Dict[str, ] = defaultdict(AxesLimit)

    @staticmethod
    def compute_ax_limit(
        values: Iterable[float], _pct: float = 0.1
    ) -> Tuple[float, float]:
        _limit = np.min(values), np.max(values)
        return (
            _limit[0] - _pct * np.abs(_limit[0]),
            _limit[1] + _pct * np.abs(_limit[1]),
        )

    def set_data(
        self,
        label: str,
        x_val: Iterable[float] = None,
        y_val: Iterable[float] = None,
        index: int = None,
        auto_xlim: bool = False,
        auto_ylim: bool = True,
        update: bool = None,
    ):
        # unique key to identify the stored variable
        key: UpdatablePlot.UniqueStoringKey = (index, label)
        plotter: Line2D = self.stored_artist.get(key, None)
        ax: Axes = self.axs if index is None else self.axs[index]

        if plotter is None:
            (plotter,) = ax.plot(x_val, y_val, label=label)
            self.stored_artist[key] = plotter
            ax.legend()
        else:
            if x_val is not None:
                plotter.set_xdata(x_val)
            if y_val is not None:
                plotter.set_ydata(y_val)
                if auto_ylim:
                    ax.relim()
                    ax.autoscale_view(True, auto_xlim, auto_ylim)

                    # ax.autoscale(True)
                    # pass
                    # # this will set the lowest/highest out of all shared series
                    # _limit = self.compute_ax_limit(y_val)
                    # self.ylim[ax].set(
                    #     lower=_limit[0], upper=_limit[1],
                    # )
                    # ax.set_ylim(self.ylim[ax].lower, self.ylim[ax].upper)
        if update is True or (update is not False and self.auto_update):
            self.update()

    def update(self):
        # drawing updated values
        self.figure.canvas.draw()

        # This will run the GUI event
        # loop until all UI events
        # currently waiting have been processed
        self.figure.canvas.flush_events()

    def clear(self):
        if isinstance(self.axs, np.ndarray):
            axs = self.axs
        else:
            axs = [self.axs]
        for ax in axs:
            ax.clear()
        # self.figure.clear()
        self.stored_artist.clear()
        # self.ylim.clear()

    def close(self):
        self.clear()
        plt.close(self.figure)


class UpdatableSequentialPlot(UpdatablePlot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stored_variables: Dict[
            UpdatablePlot.UniqueStoringKey, Memoriser
        ] = defaultdict(Memoriser)

    def add_data(self, label: str, value: float, index: int = None, **kwargs):
        var = self.stored_variables[(index, label)]
        var.add(value)
        self.set_data(
            label, x_val=var.sequence, y_val=var.values, index=index, **kwargs
        )

    def clear(self):
        super().clear()
        self.stored_variables.clear()


if __name__ == "__main__":
    plot = UpdatableSequentialPlot()
    N = 100

    plot.axs.set_xlim(0, N)
    for i in range(N):
        plot.add_data("main", np.sin(i * 0.1) * i ** 0.2)
    plot.close()

    ##############################################

    plot = UpdatableSequentialPlot(nrows=2, auto_update=False)
    N = 300

    plot.axs[0].set_xlim(0, N)
    for i in range(N):
        plot.add_data("y=sin(0.1x)", np.sin(i * 0.1), index=0)
        plot.add_data("y=cos(0.1x)", np.cos(i * 0.1), index=0)

        plot.add_data("y=0.15x", 0.15 * i, index=1)
        plot.add_data("y=x^0.5", i ** 0.5, index=1)

        if i % 5 == 0:
            plot.update()
    plot.close()
