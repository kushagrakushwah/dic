"""
Matplotlib canvas widget for embedding plots in PyQt5.
Used ONLY for final result plots — never for per-frame rendering.
"""
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=3, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()

    def clear_plot(self):
        self.axes.clear()
        self.draw()

    def plot_series(self, x, y, title="", xlabel="Frame", ylabel="Value"):
        self.axes.clear()
        self.axes.plot(x, y, marker='o', linestyle='-', markersize=3)
        self.axes.set_title(title)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.grid(True)
        self.fig.tight_layout()
        self.draw()
