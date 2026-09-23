"""
Entry point.

Run from the project root:

    python main.py
"""

import os
import sys

# Allow launching by absolute path from any working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

import config
from ui.main_window import MainWindow


def main():
    # High DPI attributes must be set before the application is constructed.
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName(config.WINDOW_TITLE)

    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
