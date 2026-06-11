"""
Based on Data-driven and Physical Evolution High-Altitude Curtain Wall Robot Digital Twin Control Console.
One-click Application Entry Point (main.py)
"""

import os
import sys

# Ensure the root workspace directory is in python search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication
from src.dashboard import MainWindow, SharedEnvironment

def main():
    print("Initializing High-Altitude Curtain Wall Robot Digital Twin Console...")
    app = QApplication(sys.argv)
    
    # Initialize shared environment
    shared_env = SharedEnvironment()
    
    # Initialize main dashboard window
    win = MainWindow(shared_env)
    win.show()
    
    print("Digital Twin Console launched successfully.")
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
