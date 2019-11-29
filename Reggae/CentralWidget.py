from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QLabel
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from MplWidget import MyMplWidget

import time

class MyCentralWidget(QWidget):

    def __init__(self, main_window, pg):
        super().__init__()
        self.main_window = main_window
        self.pg = pg
        self.initUI()

    def initUI(self):
        fini_button = QPushButton('Finished', self)
        fini_button.clicked.connect(self.on_finished_button_clicked)
        self.mpl_widget = MyMplWidget(self.pg)
        # define label
        self.label = QLabel(self)
        # Place the buttons - HZ
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(fini_button)
        hbox.addStretch(1)
        # place hbox and label into vbox
        vbox = QVBoxLayout()
        vbox.addWidget(self.mpl_widget)
        vbox.addLayout(hbox)
        self.setLayout(vbox)
        self.mpl_widget.plot_data()

    def on_finished_button_clicked(self):
        self.main_window.statusBar().showMessage('Finished!')
        self.main_window.close()
