
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
from PyQt5.QtWidgets import QSlider, QWidget, QLabel
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QSizePolicy
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import os

from .plotting import plotting

class MyMainWindow(QMainWindow, plotting):
    ''' Main window to hold the Reggae method '''
    def __init__(self, star):
        super().__init__()
        self.star = star
        self.pg = self.star.pg

        self.dnu = 10**self.star.asy_fit.summary.loc['dnu']['mean']
        self.numax = 10**self.star.asy_fit.summary.loc['numax']['mean']
        self.epsilon = self.star.asy_fit.summary.loc['eps']['mean']
        self.d02 = -10**self.star.asy_fit.summary.loc['d02']['mean']

        self.mixed = []
        self.initUI()


    def initUI(self):
        ''' Setup main window and create central widget '''
        self.resize(1600,900)
        self.move(50,50)
        central_widget = MyCentralWidget(self)
        self.setCentralWidget(central_widget)
        self.setWindowTitle('Reggae')
        self.statusBar().showMessage('Waiting ...')

class MyCentralWidget(QWidget):
    ''' The central widget tat provides the UI '''
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.pg = self.main_window.pg
        self.dnu = main_window.dnu
        self.numax = main_window.numax
        self.eps = main_window.epsilon
        self.d02 = main_window.d02 / self.dnu
        self.n = np.arange(int(self.numax/self.dnu) - 5,
                            int(self.numax/self.dnu) + 5, 1)
        self.ng = np.arange(10, 100, 1)
        self.dnu_fac = 1000
        self.eps_fac = 1000
        self.dp1_fac = 1000
        self.q_fac = 1000
        self.epsg_fac = 100
        self.d0x_fac = 100
        self.rotc_fac = 100
        self.initUI()

    def make_slider(self, min=0, max=100, step=1, init_val=50,
                    connect=None, title=''):
        ''' Make a slider in an hbox with a label

        Inputs
        ------

        min: float or int
            The minimum value of the slider

        max: float or int
            The maximum value of the slider

        step: int
            The number of steps the slider will have.

        init_val: float or int
            The value to start the slider at.

        connect: func
            The slider connect function to call when the slider is changed.

        title: string
            The title to give the slider in the GUI.

        Returns
        -------

        hbox: QHBoxLayout
            The hbox containing the slider and the label.

        sl: QSlider
            The slider object.

        '''
        hbox = QHBoxLayout()
        sl = QSlider(Qt.Horizontal)
        sl.setMinimum(int(min))
        sl.setMaximum(int(max))
        sl.setSingleStep(step)
        sl.setValue(int(init_val))
        if connect != None:
            sl.valueChanged.connect(connect)
        lab = QLabel()
        lab.setText(title)
        hbox.addWidget(lab)
        hbox.addWidget(sl)
        return hbox, sl

    def make_dnu_slider(self, tol=1.05):
        ''' Make a Delta Nu slider '''
        minv = self.dnu / tol * self.dnu_fac
        maxv = self.dnu * tol * self.dnu_fac
        init_val = self.dnu * self.dnu_fac
        hbox, dnu_sl = self.make_slider(min=minv, max=maxv,
                                  init_val=init_val,
                                  connect=self.on_value_changed,
                                  title='Dnu')
        return hbox, dnu_sl

    def make_eps_slider(self):
        ''' Make an epsilon (p mode) slider '''
        minv = 0.5 * self.eps_fac
        maxv = 1.5 * self.eps_fac
        init_val = self.eps * self.eps_fac
        hbox, eps_sl = self.make_slider(min=minv, max=maxv,
                                  init_val=init_val,
                                  connect=self.on_value_changed,
                                  title='epsilon')
        return hbox, eps_sl

    def make_dp1_slider(self):
        ''' Make Delta P1 slider '''
        minv = 70 * self.dp1_fac
        maxv = 100 * self.dp1_fac
        init_val = 90.0 * self.dp1_fac
        hbox, dp1_sl = self.make_slider(min=minv, max=maxv,
                                  init_val=init_val,
                                  connect=self.on_value_changed,
                                  title='Delta P')
        return hbox, dp1_sl

    def make_q_slider(self):
        ''' Make the coupling slider '''
        minv = 0.0 * self.q_fac
        maxv = 0.3 * self.q_fac
        init_val = 0.12 * self.q_fac
        hbox, q_sl = self.make_slider(min=minv, max=maxv,
                                  init_val=init_val,
                                  connect=self.on_value_changed,
                                  title='Coupling')
        return hbox, q_sl

    def make_epsg_slider(self):
        ''' Make epsilon g mode slider '''
        minv = -np.pi * self.q_fac
        maxv = np.pi * self.q_fac
        init_val = 0.0
        hbox, epsg_sl = self.make_slider(min=minv, max=maxv,
                                  init_val=init_val,
                                  connect=self.on_value_changed,
                                  title='epsilon_g')
        return hbox, epsg_sl

    def make_d01_slider(self):
        ''' MAke a d01 slider '''
        minv = 0.3 * self.d0x_fac
        maxv = 0.7 * self.d0x_fac
        init_val = 0.5 * self.d0x_fac
        hbox, d01_sl = self.make_slider(min=minv, max=maxv,
                                  init_val=init_val,
                                  connect=self.on_value_changed,
                                  title='d01')
        return hbox, d01_sl

    def make_d02_slider(self):
        ''' MAke a d01 slider '''
        minv = -0.25 * self.d0x_fac
        maxv = -0.05 * self.d0x_fac
        init_val = -0.15 * self.d0x_fac
        hbox, d02_sl = self.make_slider(min=minv, max=maxv,
                                  init_val=init_val,
                                  connect=self.on_value_changed,
                                  title='d02')
        return hbox, d02_sl

    def make_d03_slider(self):
        ''' MAke a d03 slider '''
        minv = 0.1 * self.d0x_fac
        maxv = 0.45 * self.d0x_fac
        init_val = 0.3 * self.d0x_fac
        hbox, d03_sl = self.make_slider(min=minv, max=maxv,
                                  init_val=init_val,
                                  connect=self.on_value_changed,
                                  title='d03')
        return hbox, d03_sl

    def make_rotc_slider(self):
        ''' Make a slider for core rotation '''
        minv = 0.0 * self.rotc_fac
        maxv = 1.0 * self.rotc_fac
        init_val = 0.4 * self.rotc_fac
        hbox, rotc_sl = self.make_slider(min=minv, max=maxv,
                                  init_val=init_val,
                                  connect=self.on_rotation_changed,
                                  title='Core rot')
        return hbox, rotc_sl

    def initUI(self):
        ''' Build the UI '''
        fini_button = QPushButton('Finished', self)
        fini_button.clicked.connect(self.on_finished_button_clicked)
        save_button = QPushButton('Save', self)
        save_button.clicked.connect(self.on_save_button_clicked)
        #hdnu, self.dnu_slider = self.make_dnu_slider()
        #heps, self.eps_slider = self.make_eps_slider()
        hdp1, self.dp1_slider = self.make_dp1_slider()
        hq, self.q_slider = self.make_q_slider()
        hepsg, self.epsg_slider = self.make_epsg_slider()
        hd01, self.d01_slider = self.make_d01_slider()
        #hd02, self.d02_slider = self.make_d02_slider()
        hd03, self.d03_slider = self.make_d03_slider()
        hrotc, self.rotc_slider = self.make_rotc_slider()
        self.mpl_widget = MyMplWidget(self.main_window.star)
        # define label
        self.label = QLabel(self)
        # Place the buttons - HZ
        subvbox = QVBoxLayout()
        #subvbox.addLayout(hdnu)
        #subvbox.addLayout(heps)
        subvbox.addLayout(hd01)
        #subvbox.addLayout(hd02)
        subvbox.addLayout(hd03)
        subvbox.addLayout(hdp1)
        subvbox.addLayout(hq)
        subvbox.addLayout(hrotc)
        subvbox.addLayout(hepsg)
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(save_button)
        hbox.addWidget(fini_button)
        hbox.addStretch(1)
        # place hbox and label into vbox
        vbox = QVBoxLayout()
        vbox.addWidget(self.mpl_widget)
        vbox.addLayout(subvbox)
        vbox.addLayout(hbox)
        self.setLayout(vbox)
        self.mpl_widget.plot_data()
        self.mpl_widget.plot_zero_two_model(self.n, self.dnu, self.eps, 0.5, self.d02, 0.3)
        self.mpl_widget.plot_mixed_model(self.n, self.dnu, self.eps, 90.0, 0.0, 0.12, 0.5)
        self.mpl_widget.plot_rotation_model(0.4)

    def get_slider_values(self):
        ''' Gets the current values of the sliders '''
        #dnu = self.dnu_slider.value() / self.dnu_fac
        #eps = self.eps_slider.value() / self.eps_fac
        d01 = self.d01_slider.value() / self.d0x_fac
        #d02 = self.d02_slider.value() / self.d0x_fac
        d03 = self.d03_slider.value() / self.d0x_fac
        dp1 = self.dp1_slider.value() / self.dp1_fac
        q = self.q_slider.value() / self.q_fac
        epsg = self.epsg_slider.value() / self.epsg_fac
        rotc = self.rotc_slider.value() / self.rotc_fac
        return d01, d03, dp1, q, epsg, rotc

    def on_value_changed(self):
        ''' Do this when the value of a slider is changed '''
        d01, d03, dp1, q, epsg, rotc = self.get_slider_values()
        self.mpl_widget.replot_zero_two_model(self.n, self.dnu, self.eps,
                                              d01, self.d02, d03)
        self.mpl_widget.replot_mixed_model(self.n, self.dnu, self.eps,
                                           dp1, epsg, q, d01)
        self.mpl_widget.replot_rotation_model(rotc)

    def on_rotation_changed(self):
        ''' Do this when the rotc slider value is changed '''
        rotc = self.rotc_slider.value() / self.rotc_fac
        self.mpl_widget.replot_rotation_model(rotc)

    def on_finished_button_clicked(self):
        ''' Do this when the finished button is clicked '''
        self.on_save_button_clicked()
        self.main_window.statusBar().showMessage('Finished!')
        self.main_window.close()

    def on_save_button_clicked(self):
        ''' Do this when the save button is clicked '''
        d01, d03, dp1, q, epsg, rotc = self.get_slider_values()
        self.main_window.df = pd.DataFrame({
                           'd01': [d01],
                           'd03': [d03],
                           'Dp1': [dp1],
                           'q': [q],
                           'rotc': [rotc],
                           'epsg': [epsg]})
        self.main_window.df.to_csv(os.path.join(*[self.main_window.star.path,
                                    f'{self.main_window.star.ID}_reggae_output.csv']))
        mixed = self.mpl_widget.get_mixed_modes(self.n, self.dnu,
                        self.eps, dp1, epsg, q, d01)
        np.savetxt(os.path.join(*[self.main_window.star.path,
                            f'{self.main_window.star.ID}_reggae_mixed_frequencies.txt']),
                            mixed, delimiter=',')
        self.main_window.mixed = mixed
        self.main_window.statusBar().showMessage('Saved to {self.main_window.star.path}')

class MyMplWidget(FigureCanvas):
    ''' The thing that plots the data in the Geggae GUI

    Inputs
    ------

    pg: lightkurve periodogram object
        The periodogram of the data.

    '''
    def __init__(self, star, parent=None, figsize=(16,9), dpi=100):
        self.star = star
        self.pg_smooth = self.star.pg.smooth(method='boxkernel', filter_width=0.1)
        self.set_nu()
        self.fig = plt.figure(figsize=figsize, dpi=dpi)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def set_nu(self, fac=4):
        ''' Sets the frquency array for the mixed algo '''
        self.minnu, self.maxnu = self.star.asy_fit._get_freq_range()
        self.nu = np.linspace(self.minnu, self.maxnu, len(self.star.pg.frequency.value)*fac)
        self.nu *= 1e-6

    def plot_data(self):
        ''' Plot the periodogram data '''
        self.ax = self.fig.add_subplot(111)
        self.ax.plot(self.star.pg.frequency, self.star.pg.power, 'k-',
                     alpha=0.3, label='Data')
        self.ax.plot(self.pg_smooth.frequency, self.pg_smooth.power, 'k-',
                     alpha=0.8, label='Data')
        self.ax.set_xlabel(r'Frequency ($\rm \mu Hz$)')
        self.ax.set_ylabel(r'SNR')
        self.ax.set_ylim([0, self.pg_smooth.power.value.max()*1.5])
        self.fig.tight_layout()

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
        fmax = (n.max()+1 + eps) * dnu
        if fmax > self.star.pg.frequency.value.max():
            fmax = self.star.pg.frequency.value.max()
        fmin = (n.min()-1 + eps) * dnu
        if fmin < self.star.pg.frequency.value.min():
            fmin = self.star.pg.frequency.value.min()
        self.ax.set_xlim([fmin, fmax])
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
