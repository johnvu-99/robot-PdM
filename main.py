import os
import sys


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import multiprocessing


def main():
  
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    import config
    from ui.main_window import MainWindow

    
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName(config.WINDOW_TITLE)

    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
