from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
from PyQt5.QtWidgets import QSlider, QWidget, QLabel
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QSizePolicy
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from MplWidget import MyMplWidget
import numpy as np

import time

class MyCentralWidget(QWidget):

    def __init__(self, main_window, pg):
        super().__init__()
        self.main_window = main_window
        self.pg = pg
        self.n = np.arange(5, 20, 1)
        self.initUI()

    def make_slider(self):
        return QSlider(Qt.Horizontal)

    def initUI(self):
        fini_button = QPushButton('Finished', self)
        fini_button.clicked.connect(self.on_finished_button_clicked)
        self.eps_slider = self.make_slider()
        self.mpl_widget = MyMplWidget(self.pg)
        # define label
        self.label = QLabel(self)
        # Place the buttons - HZ
        subvbox = QVBoxLayout()
        subvbox.addWidget(self.eps_slider)
        hbox = QHBoxLayout()
        hbox.addLayout(subvbox)
        hbox.addStretch(1)
        hbox.addWidget(fini_button)
        hbox.addStretch(1)
        # place hbox and label into vbox
        vbox = QVBoxLayout()
        vbox.addWidget(self.mpl_widget)
        vbox.addLayout(hbox)
        self.setLayout(vbox)
        self.mpl_widget.plot_data()
        self.mpl_widget.plot_zero_two_model(self.n, 16.97, 0.9, 0.14)

    def on_finished_button_clicked(self):
        self.main_window.statusBar().showMessage('Finished!')
        self.main_window.close()
