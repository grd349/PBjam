from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QLabel
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout
from PyQt5.QtGui import QPixmap
from argparse import ArgumentParser
import sys, glob, os
import pandas as pd

app = None

class MyMainWindow(QMainWindow):
    def __init__(self, df, dfpath, image_dir, app,
                 shuffle=False):
        super().__init__()
        if shuffle:
            self.df = df.sample(frac=1).reset_index(drop=True)
        else:
            self.df = df

        self.dfpath = dfpath
        self.image_dir = image_dir
        self.app = app
        self.initUI()

    def initUI(self):
        self.resize(1600, 900)
        self.move(50, 50)
        central_widget = MyCentralWidget(self, self.app)
        self.setCentralWidget(central_widget)
        self.setWindowTitle('PBjam inspector')
        self.statusBar().showMessage('Waiting...')

    def auto(self):
        self.central_widget.auto()

class MyCentralWidget(QWidget):

    def __init__(self, main_window, app):
        super().__init__()
        self.main_window = main_window
        self.idx = -1
        self.app = app
        self.initUI()


    def initUI(self):
        good_button = QPushButton('&Good', self)
        good_button.setShortcut('g')
        good_button.clicked.connect(self.on_good_button_clicked)
                
        bad_button = QPushButton('&Bad', self)
        bad_button.setShortcut('b')
        bad_button.clicked.connect(self.on_bad_button_clicked)
        
        skip_button = QPushButton('&Skip', self)
        skip_button.setShortcut('S')
        skip_button.clicked.connect(self.on_skip_button_clicked)
        
        # define label
        self.label = QLabel(self)
        self.my_widget = MyWidget(self.label, self.main_window.df, self.main_window.image_dir)
        
        # Place the buttons - HZ
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(good_button)
        hbox.addWidget(bad_button)
        hbox.addWidget(skip_button)
        hbox.addStretch(1)
        
        # place hbox and label into vbox
        vbox = QVBoxLayout()
        vbox.addWidget(self.label)
        vbox.addLayout(hbox)
        self.setLayout(vbox)
        self.next_image()

    def next_image(self):
        
        self.idx += 1
                
        while self.main_window.df.loc[self.idx].error_flag >= 0:

            self.idx += 1
            
            if (self.idx in self.main_window.df.index) == False:
                print('Finished going through CSV file')
                print('If any unclassified targets remain, they may not have associated png files')
                sys.exit()       
                   
        id = self.main_window.df.loc[self.idx].ID
               
        sfile = glob.glob(os.path.join(*[self.main_window.image_dir, id, f'asymptotic_fit_{id}.png']))

        if len(sfile)==0:
            self.my_widget.show_image(os.path.join(*[os.getcwd(),'failed.jpg']))
            mess = f"{id}/'asymptotic_fit_{id}.png' not found, so I skipped it"
            print(mess)
            self.write_verdict(-1, mess)
        else:
            self.my_widget.show_image(sfile[0])

            
    def on_good_button_clicked(self):
        self.write_verdict(0, 'Last star was Good')
        
    def on_bad_button_clicked(self):
        self.write_verdict(1, 'Last star was Bad')
        
    def on_skip_button_clicked(self):
          
        self.write_verdict(-1, 'Skipping image.')
    
    def write_verdict(self, err_code, mess):
        self.main_window.df.at[self.idx, 'error_flag'] = err_code
        perc = '%i / %i' % (self.idx, len(self.main_window.df))
        self.main_window.statusBar().showMessage(perc + ' - ' + mess)
        self.main_window.df.to_csv(self.main_window.dfpath, index=False)

        if self.idx < len(self.main_window.df) - 1:
            self.next_image()
        else:
            self.main_window.statusBar().showMessage('Finished')
            sys.exit()
    
class MyWidget():
    def __init__(self, label, df, image_dir):
        self.label = label

    def show_image(self, sfile):
        pixmap = QPixmap(sfile)
        self.label.setPixmap(pixmap)
        self.label.setScaledContents(True)

def main(df, dfpath, image_dir, shuffle=True):
    '''
    app must be defined already!!!
    '''
    global app
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    w = MyMainWindow(df, dfpath, image_dir, app, shuffle=shuffle)
    w.show()
    app.exit(app.exec_())

parser = ArgumentParser()
parser.add_argument('target_list', type=str)
parser.add_argument('image_dir', type=str)
parser.add_argument('--shuffle', action='store_true', dest='shuffle',
                    help="shuffle the list of targets")
parser.add_argument('--no-shuffle', action='store_false', dest='shuffle',
                    help="don't shuffle the list of targets (default)")
parser.set_defaults(feature=False)

if __name__ == "__main__":
    args = parser.parse_args()

    df = pd.read_csv(args.target_list, converters={'ID': str, 'error_code': int})
    
    if len(df) > 100:
        sys.setrecursionlimit(len(df))
    
    if not 'ID' in df.columns:
        print('CSV file must contain a column named ID')
        sys.exit()

    if not 'error_flag' in df.columns:
        df['error_flag'] = [-1 for n in range(len(df))]
    
    main(df, args.target_list, args.image_dir, shuffle=args.shuffle)
