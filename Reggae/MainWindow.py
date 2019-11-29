from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QLabel
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import sys

from CentralWidget import MyCentralWidget
app = None

class MyMainWindow(QMainWindow):
    def __init__(self, pg):
        super().__init__()
        self.pg = pg
        self.initUI()

    def initUI(self):
        self.resize(1600,900)
        self.move(50,50)
        central_widget = MyCentralWidget(self, self.pg)
        self.setCentralWidget(central_widget)
        self.setWindowTitle('Reggae')
        self.statusBar().showMessage('Waiting ...')

    def auto(self):
        self.central_widget.auto()

def main(pg=[], verbose=True):
    '''
    app must be defined already!!!
    '''
    global app
    if verbose:
        print('Setting up app')
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    if verbose:
        print('Setting up MyMainWindow')
    w = MyMainWindow(pg=pg)
    w.show()
    #w.auto()
    if verbose:
        print('Exiting')
    app.exit(app.exec_())

if __name__ == '__main__':
    import lightkurve as lk

    kic = '4448777'
    lcs = lk.search_lightcurvefile(kic, quarter=5).download_all()
    lc = lcs.PDCSAP_FLUX.stitch().normalize().flatten(window_length=401).remove_outliers(4)
    pg = lc.to_periodogram(normalization='psd').flatten()
    main(pg=pg)
