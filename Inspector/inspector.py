#!/usr/bin/env python3

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QLabel
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QSizePolicy
from PyQt5.QtGui import QIcon, QPixmap
import sys
import pandas as pd
import glob
import os

app = None

class MyMainWindow(QMainWindow):
    def __init__(self, df, image_dir):
        super().__init__()
        self.df = df
        self.image_dir = image_dir
        self.initUI()

    def initUI(self):
        self.resize(1600,900)
        self.move(50,50)
        central_widget = MyCentralWidget(self)
        self.setCentralWidget(central_widget)
        self.setWindowTitle('PBjam inspector')
        self.statusBar().showMessage('Waiting ...')

    def auto(self):
        self.central_widget.auto()

class MyCentralWidget(QWidget):

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.idx = 0
        self.initUI()


    def initUI(self):
        good_button = QPushButton('Good', self)
        good_button.clicked.connect(self.on_good_button_clicked)
        bad_button = QPushButton('Bad', self)
        bad_button.clicked.connect(self.on_bad_button_clicked)

        # define label
        self.label = QLabel(self)
        self.my_widget = MyWidget(self.label, self.main_window.df, self.main_window.image_dir)
        # Place the buttons - HZ
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(good_button)
        hbox.addWidget(bad_button)
        hbox.addStretch(1)
        # place hbox and label into vbox
        vbox = QVBoxLayout()
        vbox.addWidget(self.label)
        vbox.addLayout(hbox)
        self.setLayout(vbox)
        self.my_widget.show_image(self.idx)

    def on_good_button_clicked(self):
        self.main_window.df.at[self.idx, 'error_flag'] = 0
        self.main_window.statusBar().showMessage('Last jam was good')
        self.idx += 1
        self.my_widget.show_image(self.idx)

    def on_bad_button_clicked(self):
        self.main_window.df.at[self.idx, 'error_flag'] = 1
        print(self.main_window.df.loc[self.idx])
        self.main_window.statusBar().showMessage('Last jam was Bad')
        self.idx += 1
        self.my_widget.show_image(self.idx)

class MyWidget():
    def __init__(self, label, df, image_dir):
        self.df = df
        self.image_dir = image_dir
        self.label = label

    def show_image(self, idx):
        id = str(int(self.df.loc[idx].KIC))
        print(id)
        sfile = glob.glob(self.image_dir + os.sep + '*' + id + '*.png')
        pixmap = QPixmap(sfile[0])
        self.label.setPixmap(pixmap)

def main(df, image_dir):
    '''
    app must be defined already!!!
    '''
    global app
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    w = MyMainWindow(df, image_dir)
    w.show()
    app.exit(app.exec_())
    w.df.to_csv('checked.csv', index=False)

if __name__ == "__main__":
    if len(sys.argv) == 3:
        df = pd.read_csv(sys.argv[1])
        df['error_flag'] = [-1 for n in range(len(df))]
        file_dir = sys.argv[2]
        main(df, file_dir)
    else:
        print('Usage: inspector.py <targets.csv> <image_dir>')
