from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QLabel
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import matplotlib.pyplot as plt
import numpy as np

class MyMplWidget(FigureCanvas):
    def __init__(self, pg, parent=None, figsize=(16,9), dpi=100):
        self.pg = pg
        self.fig = plt.figure(figsize=figsize, dpi=dpi)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot_data(self):
        self.ax = self.fig.add_subplot(111)
        self.ax.plot(self.pg.frequency, self.pg.power, 'k-', alpha=0.5, label='Data')
        self.ax.set_xlabel(r'Frequency ($\rm \mu Hz$)')
        self.ax.set_ylabel(r'Power ($\rm ppm^2 \, \mu Hz^{-1}$)')
        self.draw()
