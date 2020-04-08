from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QLabel
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import sys

from CentralWidget import MyCentralWidget
app = None

class MyMainWindow(QMainWindow):
    ''' Main window to holf the Reggae method '''
    def __init__(self, pg, dnu, numax):
        super().__init__()
        self.pg = pg

        self.dnu = dnu
        self.numax = numax
        self.initUI()

    def initUI(self):
        ''' Setup main window and create central widget '''
        self.resize(1600,900)
        self.move(50,50)
        central_widget = MyCentralWidget(self, self.pg, self.dnu, self.numax)
        self.setCentralWidget(central_widget)
        self.setWindowTitle('Reggae')
        self.statusBar().showMessage('Waiting ...')

def main(pg, dnu, numax, verbose=False):
    '''
    Main code to run an instance of Reggae.
    '''
    global app
    if verbose:
        print('Setting up app')
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    if verbose:
        print('Setting up MyMainWindow')
    w = MyMainWindow(pg, dnu, numax)
    w.show()
    if verbose:
        print('Exiting')
    app.exit(app.exec_())

if __name__ == '__main__':
    import lightkurve as lk

    kic = '4448777'
    dnu = 16.97
    numax = 220.0
    lcs = lk.search_lightcurvefile(kic).download_all()
    lc = lcs.PDCSAP_FLUX.stitch().normalize().flatten(window_length=401).remove_outliers(4)
    pg = lc.to_periodogram(normalization='psd',
                           minimum_frequency=numax - dnu * 4,
                           maximum_frequency=numax + dnu * 4).flatten()
    main(pg, dnu, numax)
