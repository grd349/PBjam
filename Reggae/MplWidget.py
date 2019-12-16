from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QLabel
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import matplotlib.pyplot as plt
import numpy as np

class MyMplWidget(FigureCanvas):
    ''' The thing that plots the data in the Geggae GUI

    Inputs
    ------

    pg: lightkurve periodogram object
        The periodogram of the data.

    '''
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
        ''' Sets the frquency array for the mixed algo '''
        minnu = self.pg.frequency.value.min()
        maxnu = self.pg.frequency.value.max()
        self.nu = np.linspace(minnu, maxnu, len(self.pg.frequency.value)*fac)
        self.nu *= 1e-6

    def plot_data(self):
        ''' Plot the periodogram data '''
        self.ax = self.fig.add_subplot(111)
        self.ax.plot(self.pg.frequency, self.pg.power, 'k-',
                     alpha=0.3, label='Data')
        self.ax.plot(self.pg_smooth.frequency, self.pg_smooth.power, 'k-',
                     alpha=0.8, label='Data')
        self.ax.set_xlabel(r'Frequency ($\rm \mu Hz$)')
        self.ax.set_ylabel(r'Power ($\rm ppm^2 \, \mu Hz^{-1}$)')
        self.ax.set_xlim([self.pg.frequency.value.min(),
                          self.pg.frequency.value.max()])
        self.ax.set_ylim([0, self.pg_smooth.power.value.max()*0.9])

    def plot_zero_two_model(self, n, dnu, eps, d01, d02, d03):
        ''' Plot the l=0, 2 modes '''
        self.zeros, = self.ax.plot((n + eps) * dnu,
                        np.ones(len(n))*self.pg_smooth.power.value.max()*0.5,
                        'bo')
        self.twos, = self.ax.plot((n + eps + d02) * dnu,
                        np.ones(len(n))*self.pg_smooth.power.value.max()*0.35,
                        'rs')
        self.threes, = self.ax.plot((n + eps + d03) * dnu,
                        np.ones(len(n))*self.pg_smooth.power.value.max()*0.15,
                        'mh')
        self.draw()

    def replot_zero_two_model(self, n, dnu, eps, d01,  d02, d03):
        ''' Update the location in frequency of the l=0, 2 modes '''
        self.zeros.set_xdata((n + eps) * dnu)
        self.twos.set_xdata((n + eps + d02) * dnu)
        self.threes.set_xdata((n + eps + d03) * dnu)
        self.fig.canvas.draw_idle()

    def freq_model(self, dnu, nominal_pmode, period_spacing, \
                   epsilon_g, coupling):
        ''' This is the algo that estimates the mixed frequencies
        given the set of the above parameters
        '''
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
        ''' This is the function that calls the mixed mode algo '''
        nominal_pmode = (n[int(len(n)/2)] + eps + d01) * dnu
        mixed = self.freq_model(dnu, nominal_pmode, period_spacing, \
                                      epsilon_g, coupling)
        return mixed

    def plot_mixed_model(self, n, dnu, eps, period_spacing, \
                   epsilon_g, coupling, d01=0.5):
        ''' This plots the mixed mode pattern '''
        self.mixed_vals = self.get_mixed_modes(n, dnu, eps, period_spacing, \
                       epsilon_g, coupling, d01)
        self.mixed, = self.ax.plot(self.mixed_vals,
                    np.ones(len(self.mixed_vals))*self.pg_smooth.power.value.max()*0.3,
                    'gv', alpha=1.0)
        self.draw()

    def replot_mixed_model(self, n, dnu, eps, period_spacing, \
                   epsilon_g, coupling, d01):
        ''' This updates the plot of the mixed mode patter '''
        self.mixed_vals = self.get_mixed_modes(n, dnu, eps, period_spacing, \
                       epsilon_g, coupling, d01)
        self.mixed.set_data(self.mixed_vals,
                    np.ones(len(self.mixed_vals))*self.pg_smooth.power.value.max()*0.3)
        self.fig.canvas.draw_idle()

    def plot_rotation_model(self, rotc):
        self.rotation_m, = self.ax.plot(self.mixed_vals - rotc,
            np.ones(len(self.mixed_vals))*self.pg_smooth.power.value.max()*0.2,
            'gv', alpha=0.3)
        self.rotation_p, = self.ax.plot(self.mixed_vals + rotc,
            np.ones(len(self.mixed_vals))*self.pg_smooth.power.value.max()*0.2,
            'gv', alpha=0.3)
        self.draw()

    def replot_rotation_model(self, rotc):
        self.rotation_m.set_data(self.mixed_vals - rotc,
            np.ones(len(self.mixed_vals))*self.pg_smooth.power.value.max()*0.2)
        self.rotation_p.set_data(self.mixed_vals + rotc,
            np.ones(len(self.mixed_vals))*self.pg_smooth.power.value.max()*0.2)
        self.fig.canvas.draw_idle()
