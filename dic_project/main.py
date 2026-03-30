"""
DIC Analysis Tool — entry point.

Run with:
    python main.py
"""
import sys
import logging
import os

from PyQt5.QtWidgets import QApplication

from gui.dic_gui import DIC_GUI
from utils.config import LOG_LEVEL


def setup_logging():
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting DIC Analysis Tool")

    app = QApplication(sys.argv)
    app.setApplicationName("DIC Analysis Tool")

    window = DIC_GUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
