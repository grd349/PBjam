from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
from PyQt5.QtWidgets import QSlider, QWidget, QLabel
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QSizePolicy
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from MplWidget import MyMplWidget
import numpy as np

import time

class MyCentralWidget(QWidget):

    def __init__(self, main_window, pg, dnu, numax):
        super().__init__()
        self.main_window = main_window
        self.pg = pg
        self.dnu = dnu
        self.numax = numax
        self.n = np.arange(int(numax/dnu) - 5, int(numax/dnu) + 5, 1)
        self.ng = np.arange(10, 100, 1)
        self.dnu_fac = 1000
        self.eps_fac = 1000
        self.dp1_fac = 1000
        self.initUI()

    def make_slider(self, min=0, max=100, step=1, init_val=50,
                    connect=None):
        sl = QSlider(Qt.Horizontal)
        sl.setMinimum(int(min))
        sl.setMaximum(int(max))
        sl.setSingleStep(step)
        sl.setValue(int(init_val))
        if connect != None:
            sl.valueChanged.connect(connect)
        return sl

    def make_dnu_slider(self, tol=1.05):
        minv = self.dnu / tol * self.dnu_fac
        maxv = self.dnu * tol * self.dnu_fac
        init_val = self.dnu * self.dnu_fac
        dnu_sl = self.make_slider(min=minv, max=maxv,
                                  init_val=init_val,
                                  connect=self.on_value_changed)
        return dnu_sl

    def make_eps_slider(self):
        minv = 0.5 * self.eps_fac
        maxv = 1.5 * self.eps_fac
        init_val = 1.0 * self.eps_fac
        eps_sl = self.make_slider(min=minv, max=maxv,
                                  init_val=init_val,
                                  connect=self.on_value_changed)
        return eps_sl

    def make_dp1_slider(self):
        minv = 70 * self.dp1_fac
        maxv = 90 * self.dp1_fac
        init_val = 80.0 * self.dp1_fac
        dp1_sl = self.make_slider(min=minv, max=maxv,
                                  init_val=init_val,
                                  connect=self.on_value_changed)
        return dp1_sl

    def initUI(self):
        fini_button = QPushButton('Finished', self)
        fini_button.clicked.connect(self.on_finished_button_clicked)
        self.dnu_slider = self.make_dnu_slider()
        self.eps_slider = self.make_eps_slider()
        self.dp1_slider = self.make_dp1_slider()
        self.mpl_widget = MyMplWidget(self.pg)
        # define label
        self.label = QLabel(self)
        # Place the buttons - HZ
        subvbox = QVBoxLayout()
        subvbox.addWidget(self.dnu_slider)
        subvbox.addWidget(self.eps_slider)
        subvbox.addWidget(self.dp1_slider)
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(fini_button)
        hbox.addStretch(1)
        # place hbox and label into vbox
        vbox = QVBoxLayout()
        vbox.addWidget(self.mpl_widget)
        vbox.addLayout(subvbox)
        vbox.addLayout(hbox)
        self.setLayout(vbox)
        self.mpl_widget.plot_data()
        self.mpl_widget.plot_zero_two_model(self.n, self.dnu, 1.0, 0.14)
        self.mpl_widget.plot_one_model(self.ng, 80.0)
        self.mpl_widget.plot_mixed_model(self.n, self.dnu, 1.0, 70.0, 0.0, .12)

    def on_value_changed(self):
        dnu = self.dnu_slider.value() / self.dnu_fac
        eps = self.eps_slider.value() / self.eps_fac
        dp1 = self.dp1_slider.value() / self.dp1_fac
        self.mpl_widget.replot_zero_two_model(self.n, dnu, eps, 0.14)
        self.mpl_widget.replot_one_model(self.ng, dp1)
        self.mpl_widget.replot_mixed_model(self.n, dnu, eps, dp1, 0.0, .12)

    def on_finished_button_clicked(self):
        self.main_window.statusBar().showMessage('Finished!')
        self.main_window.close()
