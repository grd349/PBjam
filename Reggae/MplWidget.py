from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QLabel
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import matplotlib.pyplot as plt
import numpy as np

class MyMplWidget(FigureCanvas):
    def __init__(self, pg, parent=None, figsize=(16,9), dpi=100):
        self.pg = pg
        self.pg_smooth = pg.smooth(method='boxkernel', filter_width=0.1)
        self.set_nu()
        self.fig = plt.figure(figsize=figsize, dpi=dpi)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def set_nu(self, fac=4):
        minnu = self.pg.frequency.value.min()
        maxnu = self.pg.frequency.value.max()
        self.nu = np.linspace(minnu, maxnu, len(self.pg.frequency.value)*fac)
        self.nu *= 1e-6

    def plot_data(self):
        self.ax = self.fig.add_subplot(111)
        self.ax.plot(self.pg.frequency, self.pg.power, 'k-',
                     alpha=0.3, label='Data')
        self.ax.plot(self.pg_smooth.frequency, self.pg_smooth.power, 'k-',
                     alpha=0.8, label='Data')
        self.ax.set_xlabel(r'Frequency ($\rm \mu Hz$)')
        self.ax.set_ylabel(r'Power ($\rm ppm^2 \, \mu Hz^{-1}$)')
        self.ax.set_xlim([self.pg.frequency.value.min(),
                          self.pg.frequency.value.max()])
        self.ax.set_ylim([0, self.pg_smooth.power.value.max()*1.2])

    def plot_zero_two_model(self, n, dnu, eps, d02):
        self.zeros, = self.ax.plot((n + eps) * dnu,
                        np.ones(len(n))*self.pg_smooth.power.value.max()*0.5,
                        'bo')
        self.twos, = self.ax.plot((n + eps - d02) * dnu,
                        np.ones(len(n))*self.pg_smooth.power.value.max()*0.4,
                        'rs')
        self.draw()

    def replot_zero_two_model(self, n, dnu, eps, d02):
        self.zeros.set_xdata((n + eps) * dnu)
        self.twos.set_xdata((n + eps - d02) * dnu)
        self.fig.canvas.draw_idle()

    def plot_one_model(self, ng, deltaP1):
        self.ones, = self.ax.plot(1e6 / (ng * deltaP1),
                                  np.ones(len(ng))*self.pg_smooth.power.value.max()*0.45,
                                  'g^', alpha=0.2)
        self.draw()

    def replot_one_model(self, ng, deltaP1):
        self.ones.set_xdata(1e6 / (ng * deltaP1))
        self.fig.canvas.draw_idle()

    def freq_model(self, dnu, nominal_pmode, period_spacing, \
                   epsilon_g, coupling):
        lhs = np.tan(np.pi * (self.nu - nominal_pmode*1e-6) / (dnu * 1e-6))
        rhs = coupling * np.tan(np.pi/(period_spacing * self.nu) \
                                          - epsilon_g)
        mixed = np.ones(1)
        for i in range(len(self.nu)-1):
            y1 = lhs[i] - rhs[i]
            y2 = lhs[i+1] - rhs[i+1]
            if lhs[i] - rhs[i] < 0 and lhs[i+1] - rhs[i+1] > 0:
                m = (y2 - y1) / (self.nu[i+1] - self.nu[i])
                c = y2 - m * self.nu[i+1]
                intp = -c/m
                mixed = np.append(mixed, intp)
        if len(mixed) > 1:
            mixed = mixed[1:]
        return mixed * 1e6

    def get_mixed_modes(self, n, dnu, eps, period_spacing, \
                   epsilon_g, coupling, d01=0.5):
        nominal_pmode = (n[int(len(n)/2)] + eps + d01) * dnu
        mixed = self.freq_model(dnu, nominal_pmode, period_spacing, \
                                      epsilon_g, coupling)
        return mixed

    def plot_mixed_model(self, n, dnu, eps, period_spacing, \
                   epsilon_g, coupling, d01=0.5):
        mixed = self.get_mixed_modes(n, dnu, eps, period_spacing, \
                       epsilon_g, coupling, d01)
        self.mixed, = self.ax.plot(mixed,
                    np.ones(len(mixed))*self.pg_smooth.power.value.max()*0.35,
                    'gv', alpha=1.0)
        self.draw()

    def replot_mixed_model(self, n, dnu, eps, period_spacing, \
                   epsilon_g, coupling, d01):
        mixed = self.get_mixed_modes(n, dnu, eps, period_spacing, \
                       epsilon_g, coupling, d01)
        self.mixed.set_data(mixed,
                    np.ones(len(mixed))*self.pg_smooth.power.value.max()*0.35)
        self.fig.canvas.draw_idle()
