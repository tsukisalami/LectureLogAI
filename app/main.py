import sys
import os
import warnings
import logging
import traceback
from pathlib import Path

# Basic console handling for critical errors
def print_error(msg):
    """Print error message to console"""
    print("\n===== ERROR =====")
    print(msg)
    print("=================\n")

print("Starting application...")

# Set up logging to file - do this first thing
try:
    log_file = "app_log.txt"
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # Add console handler for better visibility during debugging
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s: %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    logging.info("=== Application starting ===")
    print(f"Logging to {os.path.abspath(log_file)}")
except Exception as e:
    print_error(f"Error setting up logging: {e}")
    # If we can't set up logging, we'll continue without it

# Log unhandled exceptions
def exception_hook(exctype, value, traceback_obj):
    """Global function to log unhandled exceptions"""
    error_msg = ''.join(traceback.format_exception(exctype, value, traceback_obj))
    try:
        logging.critical(f"Unhandled exception: {error_msg}")
    except:
        pass  # If logging fails, we can't do much
    print_error(f"CRITICAL ERROR: {error_msg}")
    sys.__excepthook__(exctype, value, traceback_obj)

sys.excepthook = exception_hook

try:
    print("Initializing application...")
    
    # Suppress the FutureWarning from whisper about torch.load
    warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")
    
    # Add the current directory to sys.path
    current_dir = Path(__file__).resolve().parent.parent
    sys.path.append(str(current_dir))
    print(f"Added {current_dir} to system path")
    
    # Import PyQt safely - this is a common source of crashes
    try:
        print("Importing PyQt5...")
        from PyQt5.QtWidgets import QApplication, QMessageBox
        from PyQt5.QtCore import Qt
        logging.info("PyQt5 imported successfully")
    except ImportError as e:
        error_msg = f"Failed to import PyQt5: {e}"
        logging.critical(error_msg)
        print_error(error_msg)
        sys.exit(1)
    
    # Try to import the main window class
    try:
        print("Importing MainWindow...")
        from app.views.main_window import MainWindow
        logging.info("MainWindow class imported successfully")
    except ImportError as e:
        error_msg = f"Failed to import MainWindow: {e}\n{traceback.format_exc()}"
        logging.critical(error_msg)
        print_error(error_msg)
        
        # We can't show a QMessageBox if QApplication isn't initialized yet
        if 'QApplication' in globals():
            app = QApplication(sys.argv)
            QMessageBox.critical(None, "Import Error", error_msg)
        sys.exit(1)
    
    if __name__ == "__main__":
        try:
            print("Creating QApplication...")
            # Enable High DPI scaling
            QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
            QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
            
            # Create the application
            app = QApplication(sys.argv)
            app.setStyle("fusion")  # Use Fusion style for a modern look
            
            logging.info("QApplication created, creating MainWindow")
            print("Creating MainWindow...")
            
            # Create and show the main window
            window = MainWindow()
            print("Showing MainWindow...")
            window.show()
            
            logging.info("MainWindow shown, starting event loop")
            print("Starting event loop...")
            
            # Start the application event loop
            exit_code = app.exec_()
            logging.info(f"Application exiting with code {exit_code}")
            sys.exit(exit_code)
            
        except Exception as e:
            error_msg = f"Critical error in main application: {str(e)}\n\n{traceback.format_exc()}"
            logging.critical(error_msg)
            print_error(error_msg)
            
            # We can only show a message box if QApplication exists
            if 'app' in locals():
                error_dialog = QMessageBox()
                error_dialog.setIcon(QMessageBox.Critical)
                error_dialog.setText("Application Error")
                error_dialog.setInformativeText(error_msg)
                error_dialog.setWindowTitle("Critical Error")
                error_dialog.exec_()
            else:
                print_error(error_msg)
                
            sys.exit(1)
except Exception as e:
    # Last resort error handling
    error_msg = f"Fatal application error: {str(e)}\n\n{traceback.format_exc()}"
    print_error(error_msg)
    try:
        logging.critical(error_msg)
    except:
        pass
    sys.exit(1) 