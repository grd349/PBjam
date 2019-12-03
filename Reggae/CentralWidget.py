from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
from PyQt5.QtWidgets import QSlider, QWidget, QLabel
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QSizePolicy
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from MplWidget import MyMplWidget
import numpy as np
import pandas as pd

class MyCentralWidget(QWidget):
    ''' The central widget tat provides the UI '''
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
        self.q_fac = 1000
        self.epsg_fac = 100
        self.d01_fac = 100
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
        init_val = 1.0 * self.eps_fac
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
        minv = 0.3 * self.d01_fac
        maxv = 0.7 * self.d01_fac
        init_val = 0.5 * self.d01_fac
        hbox, d01_sl = self.make_slider(min=minv, max=maxv,
                                  init_val=init_val,
                                  connect=self.on_value_changed,
                                  title='d01')
        return hbox, d01_sl

    def initUI(self):
        ''' Build the UI '''
        fini_button = QPushButton('Finished', self)
        fini_button.clicked.connect(self.on_finished_button_clicked)
        save_button = QPushButton('Save', self)
        save_button.clicked.connect(self.on_save_button_clicked)
        hdnu, self.dnu_slider = self.make_dnu_slider()
        heps, self.eps_slider = self.make_eps_slider()
        hdp1, self.dp1_slider = self.make_dp1_slider()
        hq, self.q_slider = self.make_q_slider()
        hepsg, self.epsg_slider = self.make_epsg_slider()
        hd01, self.d01_slider = self.make_d01_slider()
        self.mpl_widget = MyMplWidget(self.pg)
        # define label
        self.label = QLabel(self)
        # Place the buttons - HZ
        subvbox = QVBoxLayout()
        subvbox.addLayout(hdnu)
        subvbox.addLayout(heps)
        subvbox.addLayout(hdp1)
        subvbox.addLayout(hq)
        subvbox.addLayout(hepsg)
        subvbox.addLayout(hd01)
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
        self.mpl_widget.plot_zero_two_model(self.n, self.dnu, 1.0, 0.14)
        self.mpl_widget.plot_mixed_model(self.n, self.dnu, 1.0, 90.0, 0.0, .12)

    def get_slider_values(self):
        ''' Gets the current values of the sliders '''
        dnu = self.dnu_slider.value() / self.dnu_fac
        eps = self.eps_slider.value() / self.eps_fac
        dp1 = self.dp1_slider.value() / self.dp1_fac
        q = self.q_slider.value() / self.q_fac
        epsg = self.epsg_slider.value() / self.epsg_fac
        d01 = self.d01_slider.value() / self.d01_fac
        return dnu, eps, dp1, q, epsg, d01

    def on_value_changed(self):
        ''' Do this when the value of a slider is changed '''
        dnu, eps, dp1, q, epsg, d01 = self.get_slider_values()
        self.mpl_widget.replot_zero_two_model(self.n, dnu, eps, 0.14)
        self.mpl_widget.replot_mixed_model(self.n, dnu, eps, dp1, epsg, q, d01)

    def on_finished_button_clicked(self):
        ''' Do this when the finished button is clicked '''
        self.main_window.statusBar().showMessage('Finished!')
        self.main_window.close()

    def on_save_button_clicked(self):
        ''' Do this when the save button is clicked '''
        dnu, eps, dp1, q, epsg, d01 = self.get_slider_values()
        df = pd.DataFrame({'Dnu': [dnu],
                           'Eps': [eps],
                           'Dp1': [dp1],
                           'q': [q],
                           'epsg': [epsg],
                           'd01': [d01]})
        df.to_csv('reggae_output.csv')
        mixed = self.mpl_widget.get_mixed_modes(self.n, dnu,
                        eps, dp1, epsg, q, d01)
        np.savetxt('reggae_mixed_frequencies.txt', mixed, delimiter=',')
        self.main_window.statusBar().showMessage('Saved')
