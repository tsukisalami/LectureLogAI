from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QComboBox, QTabWidget,
    QListWidget, QStackedWidget, QSplitter, 
    QLineEdit, QMessageBox, QDialog, QFormLayout,
    QDateEdit, QTextEdit, QProgressDialog, QCheckBox,
    QProgressBar, QGroupBox, QRadioButton, QButtonGroup,
    QApplication, QAction, QSlider, QStyle, QToolButton,
    QShortcut
)
from PyQt5.QtCore import Qt, QDateTime, QTimer, pyqtSignal, QObject, QUrl, QSize
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtGui import QKeySequence

from ..models.database import Database
from ..utils.audio_recorder import AudioRecorder
from ..utils.ai_processor import AIProcessor
from pathlib import Path
import os
import datetime
import threading
import time
import logging
import requests
import sys
import platform
import shutil
import re

# Worker signal classes for thread-safe communication
class TranscriptionSignals(QObject):
    """Signal class for transcription worker thread"""
    finished = pyqtSignal(str, int, bool, str)  # transcript, class_id, error_occurred, error_message

class SummarizationSignals(QObject):
    """Signal class for summarization worker thread"""
    finished = pyqtSignal(str, int, bool, str)  # summary, class_id, error_occurred, error_message

class RecordDialog(QDialog):
    """Dialog for recording audio."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Record Class Audio")
        self.resize(400, 300)
        
        # Use the parent's theme if available
        if parent and hasattr(parent, 'settings') and parent.settings.get('theme') == 'dark':
            self.setStyleSheet("""
                QDialog { background-color: #2d2d2d; color: white; }
                QLabel { color: white; }
                QPushButton { background-color: #444; color: white; padding: 6px; }
                QPushButton:disabled { color: #888; }
            """)
        
        # Initialize recorder
        self.recorder = AudioRecorder()
        
        # Create UI elements
        self.layout = QVBoxLayout()
        
        # Status label
        self.status_label = QLabel("Ready to record")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.status_label)
        
        # Timer label
        self.timer_label = QLabel("00:00:00")
        self.timer_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(24)
        self.timer_label.setFont(font)
        self.layout.addWidget(self.timer_label)
        
        # Buttons layout
        self.button_layout = QHBoxLayout()
        
        # Start button
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_recording)
        self.button_layout.addWidget(self.start_button)
        
        # Pause/Resume button
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.pause_button.setEnabled(False)
        self.button_layout.addWidget(self.pause_button)
        
        # Stop button
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_recording)
        self.stop_button.setEnabled(False)
        self.button_layout.addWidget(self.stop_button)
        
        self.layout.addLayout(self.button_layout)
        
        # Set up the timer
        self.recording_seconds = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_timer)
        
        self.setLayout(self.layout)
        
        # Result path
        self.recording_path = None
    
    def update_timer(self):
        """Update the timer display."""
        self.recording_seconds += 1
        hours = self.recording_seconds // 3600
        minutes = (self.recording_seconds % 3600) // 60
        seconds = self.recording_seconds % 60
        self.timer_label.setText(f"{hours:02}:{minutes:02}:{seconds:02}")
    
    def start_recording(self):
        """Start recording audio."""
        if self.recorder.start_recording():
            self.status_label.setText("Recording...")
            self.start_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            
            # Start the timer
            self.recording_seconds = 0
            self.timer.start(1000)  # Update every second
    
    def toggle_pause(self):
        """Pause or resume recording."""
        if self.recorder.is_paused():
            if self.recorder.resume_recording():
                self.status_label.setText("Recording...")
                self.pause_button.setText("Pause")
                self.timer.start(1000)
        else:
            if self.recorder.pause_recording():
                self.status_label.setText("Paused")
                self.pause_button.setText("Resume")
                self.timer.stop()
    
    def stop_recording(self):
        """Stop recording and save the audio file."""
        self.recording_path = self.recorder.stop_recording()
        
        # Stop the timer
        self.timer.stop()
        
        if self.recording_path:
            self.status_label.setText(f"Recording saved: {Path(self.recording_path).name}")
            
            # Reset buttons
            self.start_button.setEnabled(True)
            self.pause_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.pause_button.setText("Pause")
            
            # Accept the dialog to return
            self.accept()
        else:
            self.status_label.setText("Error saving recording")
    
    def get_recording_path(self):
        """Get the path of the saved recording."""
        return self.recording_path


class AddSubjectDialog(QDialog):
    """Dialog for adding a new subject."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add New Subject")
        self.resize(300, 100)
        
        # Use the parent's theme if available
        if parent and hasattr(parent, 'settings') and parent.settings.get('theme') == 'dark':
            self.setStyleSheet("""
                QDialog { background-color: #2d2d2d; color: white; }
                QLabel { color: white; }
                QLineEdit { background-color: #3d3d3d; color: white; padding: 2px; }
                QPushButton { background-color: #444; color: white; padding: 6px; }
            """)
        
        self.layout = QFormLayout()
        
        # Subject name
        self.subject_name = QLineEdit()
        self.layout.addRow("Subject Name:", self.subject_name)
        
        # Buttons
        self.button_layout = QHBoxLayout()
        self.ok_button = QPushButton("Add")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        self.button_layout.addWidget(self.ok_button)
        self.button_layout.addWidget(self.cancel_button)
        self.layout.addRow("", self.button_layout)
        
        self.setLayout(self.layout)
    
    def get_subject_name(self):
        """Get the entered subject name."""
        return self.subject_name.text().strip()


class AddClassDialog(QDialog):
    """Dialog for adding a new class."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add New Class")
        self.resize(400, 200)
        
        # Use the parent's theme if available
        if parent and hasattr(parent, 'settings') and parent.settings.get('theme') == 'dark':
            self.setStyleSheet("""
                QDialog { background-color: #2d2d2d; color: white; }
                QLabel { color: white; }
                QLineEdit, QDateEdit { background-color: #3d3d3d; color: white; padding: 2px; }
                QPushButton { background-color: #444; color: white; padding: 6px; }
                QCalendarWidget { background-color: #3d3d3d; color: white; }
                QCalendarWidget QAbstractItemView:enabled { color: white; background-color: #444; selection-background-color: #2a82da; }
                QCalendarWidget QAbstractItemView:disabled { color: #888; }
            """)
        
        self.layout = QFormLayout()
        
        # Class name
        self.class_name = QLineEdit()
        self.layout.addRow("Class Name:", self.class_name)
        
        # Date
        self.class_date = QDateEdit()
        self.class_date.setDateTime(QDateTime.currentDateTime())
        self.class_date.setCalendarPopup(True)
        self.layout.addRow("Date:", self.class_date)
        
        # Chapter
        self.class_chapter = QLineEdit()
        self.layout.addRow("Chapter:", self.class_chapter)
        
        # Buttons
        self.button_layout = QHBoxLayout()
        self.ok_button = QPushButton("Add")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        self.button_layout.addWidget(self.ok_button)
        self.button_layout.addWidget(self.cancel_button)
        self.layout.addRow("", self.button_layout)
        
        self.setLayout(self.layout)
    
    def get_class_info(self):
        """Get the entered class information."""
        name = self.class_name.text().strip()
        date = self.class_date.date().toString(Qt.ISODate)
        chapter = self.class_chapter.text().strip()
        
        return {
            "name": name,
            "date": date,
            "chapter": chapter if chapter else None
        }


class SettingsDialog(QDialog):
    """Global application settings dialog"""
    
    def __init__(self, parent=None, database=None, ai_processor=None):
        super().__init__(parent)
        self.parent = parent
        self.database = database
        self.ai_processor = ai_processor
        self.settings = self.database.get_settings() if database else {}
        
        self.setWindowTitle("Application Settings")
        self.setMinimumWidth(500)
        
        # Apply theme if parent has it set
        if parent and hasattr(parent, 'is_dark_theme') and parent.is_dark_theme:
            self.setStyleSheet("""
                QDialog { background-color: #2D2D2D; color: white; }
                QLabel { color: white; }
                QGroupBox { color: white; }
                QPushButton { background-color: #444; color: white; padding: 6px; border: none; }
                QPushButton:hover { background-color: #666; }
                QLineEdit, QComboBox, QSpinBox { 
                    background-color: #444; 
                    color: white; 
                    padding: 6px; 
                    border: 1px solid #555; 
                }
                QTabWidget::pane { border: 1px solid #555; }
                QTabBar::tab { 
                    background-color: #444; 
                    color: white; 
                    padding: 8px 16px; 
                    border: 1px solid #555; 
                }
                QTabBar::tab:selected { background-color: #666; }
            """)
        
        layout = QVBoxLayout()
        
        # Create tabs for different settings categories
        tab_widget = QTabWidget()
        
        # Appearance tab
        appearance_tab = QWidget()
        appearance_layout = QVBoxLayout(appearance_tab)
        
        # Theme selection
        theme_group = QGroupBox("Appearance")
        theme_layout = QVBoxLayout()
        theme_label = QLabel("Application Theme:")
        
        theme_radio_layout = QHBoxLayout()
        self.light_radio = QRadioButton("Light Theme")
        self.dark_radio = QRadioButton("Dark Theme")
        
        # Set the initial theme state
        if self.settings.get('theme', 'dark') == 'light':
            self.light_radio.setChecked(True)
        else:
            self.dark_radio.setChecked(True)
        
        theme_radio_layout.addWidget(self.light_radio)
        theme_radio_layout.addWidget(self.dark_radio)
        theme_radio_layout.addStretch()
        
        theme_layout.addWidget(theme_label)
        theme_layout.addLayout(theme_radio_layout)
        theme_group.setLayout(theme_layout)
        appearance_layout.addWidget(theme_group)
        appearance_layout.addStretch()
        
        # AI Settings tab
        ai_tab = QWidget()
        ai_layout = QVBoxLayout(ai_tab)
        
        # Ollama settings
        ollama_group = QGroupBox("Ollama Settings")
        ollama_layout = QFormLayout()
        
        # Ollama host
        self.ollama_host_input = QLineEdit(self.settings.get('ollama_host', 'http://localhost:11434'))
        ollama_layout.addRow("Ollama Host:", self.ollama_host_input)
        
        # Ollama model - EDITABLE combobox
        self.ollama_model_combo = QComboBox()
        self.ollama_model_combo.setEditable(True)
        current_model = self.settings.get('ollama_model', 'mistral:latest')
        self.ollama_model_combo.addItem(current_model)
        
        # Try to load models, but don't rely on this
        refresh_button = QPushButton("Refresh Models")
        refresh_button.clicked.connect(self.load_ollama_models)
        
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.ollama_model_combo)
        model_layout.addWidget(refresh_button)
        ollama_layout.addRow("Ollama Model:", model_layout)
        
        # Add a label with instructions
        model_help = QLabel("Type or paste model names manually if automatic detection fails\n"
                          "Example: mistral:latest, llama2:13b, etc.")
        model_help.setWordWrap(True)
        ollama_layout.addRow("", model_help)
        
        # Whisper model size
        self.whisper_model_combo = QComboBox()
        whisper_models = ["tiny", "base", "small", "medium", "large"]
        self.whisper_model_combo.addItems(whisper_models)
        current_whisper = self.settings.get('whisper_model_size', 'base')
        if current_whisper in whisper_models:
            self.whisper_model_combo.setCurrentText(current_whisper)
        ollama_layout.addRow("Whisper Model Size:", self.whisper_model_combo)
        
        # Summarization presets
        self.summarization_preset_combo = QComboBox()
        summarization_presets = [
            "Reserved & concise", 
            "Reserved & developed", 
            "Outspoken & concise", 
            "Outspoken & developed"
        ]
        self.summarization_preset_combo.addItems(summarization_presets)
        current_preset = self.settings.get('summarization_preset', 'Reserved & concise')
        if current_preset in summarization_presets:
            self.summarization_preset_combo.setCurrentText(current_preset)
        
        # Help text for summarization presets
        preset_help = QLabel(
            "Reserved: Less extrapolation, sticks to transcript content\n"
            "Outspoken: More creative, may add relevant context\n"
            "Concise: Shorter summaries\n"
            "Developed: Longer, more detailed summaries"
        )
        preset_help.setWordWrap(True)
        
        ollama_layout.addRow("Summarization Style:", self.summarization_preset_combo)
        ollama_layout.addRow("", preset_help)
        
        # Safe mode checkbox
        self.safe_mode_cb = QCheckBox("Enable Safe Mode for transcription")
        if self.ai_processor:
            self.safe_mode_cb.setChecked(self.ai_processor.safe_mode)
        ollama_layout.addRow("", self.safe_mode_cb)
        
        ollama_group.setLayout(ollama_layout)
        ai_layout.addWidget(ollama_group)
        ai_layout.addStretch()
        
        # Add tabs to the tab widget
        tab_widget.addTab(appearance_tab, "Appearance")
        tab_widget.addTab(ai_tab, "AI Settings")
        
        layout.addWidget(tab_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        save_button = QPushButton("Save Settings")
        cancel_button = QPushButton("Cancel")
        
        save_button.clicked.connect(self.save_settings)
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def load_ollama_models(self):
        """Attempt to load available Ollama models"""
        if not self.ai_processor:
            return
            
        self.setCursor(Qt.WaitCursor)
        try:
            models = self.ai_processor.get_available_ollama_models()
            if models:
                current_text = self.ollama_model_combo.currentText()
                self.ollama_model_combo.clear()
                self.ollama_model_combo.addItems(models)
                
                # Try to restore the current selection if it exists
                index = self.ollama_model_combo.findText(current_text)
                if index >= 0:
                    self.ollama_model_combo.setCurrentIndex(index)
                else:
                    self.ollama_model_combo.setCurrentText(current_text)
            else:
                QMessageBox.warning(self, "Model Loading Failed", 
                                   "Could not retrieve Ollama models. Please ensure Ollama is running and try again, "
                                   "or manually enter your model name.")
        except Exception as e:
            QMessageBox.warning(self, "Model Loading Error", f"Error loading models: {str(e)}")
        finally:
            self.setCursor(Qt.ArrowCursor)
    
    def save_settings(self):
        """Save the settings and update the application"""
        try:
            # Get the theme
            theme = 'light' if self.light_radio.isChecked() else 'dark'
            
            # Get the Ollama settings
            ollama_host = self.ollama_host_input.text().strip()
            ollama_model = self.ollama_model_combo.currentText().strip()
            whisper_model_size = self.whisper_model_combo.currentText()
            summarization_preset = self.summarization_preset_combo.currentText()
            
            # Update settings
            self.settings['theme'] = theme
            self.settings['ollama_host'] = ollama_host
            self.settings['ollama_model'] = ollama_model
            self.settings['whisper_model_size'] = whisper_model_size
            self.settings['summarization_preset'] = summarization_preset
            
            # Save to database
            if self.database:
                self.database.save_settings(self.settings)
            
            # Update AI processor
            if self.ai_processor:
                self.ai_processor.update_ollama_settings(
                    host=ollama_host,
                    model=ollama_model,
                    whisper_model_size=whisper_model_size,
                    summarization_preset=summarization_preset
                )
                self.ai_processor.safe_mode = self.safe_mode_cb.isChecked()
            
            # Apply theme if parent is MainWindow
            if self.parent and hasattr(self.parent, 'apply_theme'):
                self.parent.apply_theme(theme)
            
            QMessageBox.information(self, "Settings Saved", "Settings have been saved successfully.")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save settings: {str(e)}")

class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        
        # Set window properties
        self.setWindowTitle("Class Transcriber")
        self.setMinimumSize(1000, 600)
        
        # Initialize database
        self.db = Database()
        
        # Load settings
        self.settings = self.db.get_settings()
        self.is_dark_theme = self.settings.get('theme', 'dark') == 'dark'
        
        # Apply theme
        self.apply_theme(self.settings.get('theme', 'dark'))
        
        # Create the application menu
        self.create_menu()
        
        # Initialize AI processor with settings
        self.ai_processor = AIProcessor(
            ollama_host=self.settings.get('ollama_host', 'http://localhost:11434'),
            model_name=self.settings.get('ollama_model', 'mistral:latest'),
            whisper_model_size=self.settings.get('whisper_model_size', 'base'),
            summarization_preset=self.settings.get('summarization_preset', 'Reserved & concise')
        )
        
        # Setup the recorder
        self.recorder = AudioRecorder(data_dir='app/data/recordings')
        
        # Create central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Create main layout
        self.main_layout = QHBoxLayout(self.central_widget)
        
        # Create left panel (subject/class selection)
        self.left_panel = QWidget()
        self.left_panel.setMaximumWidth(300)
        self.left_panel_layout = QVBoxLayout(self.left_panel)
        
        # Create content area (tabs for transcript, summary, etc.)
        self.content_area = QWidget()
        self.content_layout = QVBoxLayout(self.content_area)
        
        # Add a splitter to manage left panel and content area
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.left_panel)
        self.splitter.addWidget(self.content_area)
        self.splitter.setSizes([300, 700])  # Set initial sizes
        
        self.main_layout.addWidget(self.splitter)
        
        # Create UI components
        self.create_subject_ui()
        self.create_content_tabs()
        
        # Set up the current process tracking
        self.current_process = None
        self.cancel_requested = False
        
        # Create timer for progress pulse
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self._update_progress_pulse)
        
        # Create animation timer for progress indicator
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._update_progress_animation)
        
        # Load subjects
        self.load_subjects()
        
        # Check if Ollama is running
        self.check_ollama()
        
        # Initialize audio player
        self.media_player = QMediaPlayer()
        self.media_player.durationChanged.connect(self.update_duration)
        self.media_player.positionChanged.connect(self.update_position)
        self.media_player.stateChanged.connect(self.media_state_changed)
    
    def create_subject_ui(self):
        """Create UI for subject selection and management."""
        # Subject selection
        subject_layout = QHBoxLayout()
        
        # Label
        subject_label = QLabel("Subject:")
        subject_layout.addWidget(subject_label)
        
        # Dropdown for subjects
        self.subject_combo = QComboBox()
        self.subject_combo.setMinimumWidth(200)
        self.subject_combo.currentIndexChanged.connect(self.on_subject_changed)
        subject_layout.addWidget(self.subject_combo)
        
        # Add subject button
        add_subject_button = QPushButton("Add Subject")
        add_subject_button.clicked.connect(self.add_subject)
        subject_layout.addWidget(add_subject_button)
        
        # Delete subject button
        delete_subject_button = QPushButton("Delete Subject")
        delete_subject_button.clicked.connect(self.delete_subject)
        subject_layout.addWidget(delete_subject_button)
        
        # Add to panel layout
        subject_layout.addStretch()
        self.left_panel_layout.addLayout(subject_layout)
        
        # Classes list and buttons
        classes_header_layout = QHBoxLayout()
        classes_label = QLabel("Classes:")
        classes_header_layout.addWidget(classes_label)
        
        add_class_button = QPushButton("Add Class")
        add_class_button.clicked.connect(self.add_class)
        classes_header_layout.addWidget(add_class_button)
        
        delete_class_button = QPushButton("Delete Class")
        delete_class_button.clicked.connect(self.delete_class)
        classes_header_layout.addWidget(delete_class_button)
        
        classes_header_layout.addStretch()
        self.left_panel_layout.addLayout(classes_header_layout)
        
        # Classes list
        self.classes_list = QListWidget()
        self.classes_list.currentRowChanged.connect(self.on_class_selected)
        self.left_panel_layout.addWidget(self.classes_list)
    
    def create_content_tabs(self):
        """Create the tabs for content display."""
        # Class title and date display
        self.class_title_layout = QHBoxLayout()
        
        # Audio icon and class title
        self.audio_icon_label = QLabel()
        self.audio_icon = QIcon("app/static/audio_icon.png")
        self.audio_icon_pixmap = self.audio_icon.pixmap(32, 32)
        self.audio_icon_label.setPixmap(self.audio_icon_pixmap)
        self.audio_icon_label.setVisible(False)  # Hide initially
        
        self.class_title_label = QLabel("No class selected")
        self.class_title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        
        self.class_title_layout.addWidget(self.audio_icon_label)
        self.class_title_layout.addWidget(self.class_title_label)
        self.class_title_layout.addStretch()
        
        # Date display
        self.class_date_label = QLabel("")
        self.class_date_label.setStyleSheet("font-size: 14px;")
        self.class_title_layout.addWidget(self.class_date_label)
        
        self.content_layout.addLayout(self.class_title_layout)
        
        # Create tab widget
        self.content_tabs = QTabWidget()
        self.content_layout.addWidget(self.content_tabs)
        
        # Transcript tab
        self.transcript_tab = QWidget()
        self.transcript_layout = QVBoxLayout(self.transcript_tab)
        
        self.transcript_text = QTextEdit()
        self.transcript_text.setReadOnly(True)
        self.transcript_text.setPlaceholderText("Transcript will appear here")
        
        self.transcript_layout.addWidget(self.transcript_text)
        
        # Summary tab
        self.summary_tab = QWidget()
        self.summary_layout = QVBoxLayout(self.summary_tab)
        
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setPlaceholderText("Summary will appear here")
        
        self.summary_layout.addWidget(self.summary_text)
        
        # Logs tab
        self.logs_tab = QWidget()
        self.logs_layout = QVBoxLayout(self.logs_tab)
        
        self.logs_text = QTextEdit()
        self.logs_text.setReadOnly(True)
        self.logs_text.setLineWrapMode(QTextEdit.NoWrap)  # Don't wrap log lines
        self.logs_text.setFont(QFont("Courier", 9))  # Monospaced font for logs
        
        self.logs_layout.addWidget(self.logs_text)
        
        # Add tabs to the tab widget
        self.content_tabs.addTab(self.transcript_tab, "Transcript")
        self.content_tabs.addTab(self.summary_tab, "Summary")
        self.content_tabs.addTab(self.logs_tab, "Logs")
        
        # Buttons for actions
        self.action_layout = QHBoxLayout()
        
        # Record audio button
        self.record_button = QPushButton("Record Audio")
        self.record_button.clicked.connect(self.record_audio)
        
        # Transcribe button
        self.transcribe_button = QPushButton("Transcribe")
        self.transcribe_button.clicked.connect(self.transcribe_audio)
        self.transcribe_button.setEnabled(False)  # Disable until class with audio is selected
        
        # Safe mode checkbox
        self.safe_mode_cb = QCheckBox("Safe Mode")
        self.safe_mode_cb.setToolTip("Use a simpler transcription method that may be more reliable on some systems")
        self.safe_mode_cb.stateChanged.connect(self.toggle_safe_mode)
        
        # Summarize button
        self.summarize_button = QPushButton("Summarize")
        self.summarize_button.clicked.connect(self.summarize_transcript)
        self.summarize_button.setEnabled(False)  # Disable until transcript is available
        
        self.action_layout.addWidget(self.record_button)
        self.action_layout.addWidget(self.transcribe_button)
        self.action_layout.addWidget(self.safe_mode_cb)
        self.action_layout.addStretch()
        self.action_layout.addWidget(self.summarize_button)
        
        self.content_layout.addLayout(self.action_layout)
        
        # Initialize progress UI elements (hidden initially)
        self.progress_layout = QHBoxLayout()
        
        self.progress_label = QLabel("Processing...")
        
        # Use a series of dots as a simple progress indicator
        self.progress_indicator = QLabel("⬤ ⬤ ⬤ ⬤ ⬤")
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._cancel_current_process)
        
        self.progress_layout.addWidget(self.progress_label)
        self.progress_layout.addWidget(self.progress_indicator)
        self.progress_layout.addStretch()
        self.progress_layout.addWidget(self.cancel_button)
        
        # Create a widget to contain the progress UI
        self.progress_widget = QWidget()
        self.progress_widget.setLayout(self.progress_layout)
        self.progress_widget.setVisible(False)  # Hide initially
        
        self.content_layout.addWidget(self.progress_widget)
    
    def load_subjects(self):
        """Load subjects into the combo box."""
        self.subject_combo.clear()
        self.subject_combo.addItem("Select a subject...", None)
        
        subjects = self.db.get_subjects()
        for subject in subjects:
            self.subject_combo.addItem(subject['name'], subject['id'])
    
    def load_classes(self, subject_id):
        """Load classes for the selected subject."""
        self.classes_list.clear()
        
        if subject_id is None:
            return
        
        classes = self.db.get_classes(subject_id)
        
        # Sort classes by date (newest first)
        classes.sort(key=lambda c: c['date'], reverse=True)
        
        for cls in classes:
            display_text = f"{cls['name']} ({cls['date']})"
            item = self.classes_list.addItem(display_text)
            # Store the class ID as item data
            self.classes_list.item(self.classes_list.count() - 1).setData(Qt.UserRole, cls['id'])
    
    def on_subject_changed(self, index):
        """Handle subject selection change."""
        if index <= 0:
            self.classes_list.clear()
            return
        
        subject_id = self.subject_combo.currentData()
        self.load_classes(subject_id)
    
    def on_class_selected(self):
        """Handler for when a class is selected."""
        # Stop any currently playing audio
        if hasattr(self, 'media_player'):
            self.media_player.stop()
            
        row = self.classes_list.currentRow()
        if row < 0:
            # No selection, clear UI
            self.class_title_label.setText("No class selected")
            self.class_date_label.setText("")
            self.transcript_text.setText("")
            self.summary_text.setText("")
            self.transcribe_button.setEnabled(False)
            self.transcribe_button.setText("No Class Selected")
            self.summarize_button.setEnabled(False)
            self.log_to_console("No class selected")
            return
        
        # Get the class ID from the selected item
        if not self.classes_list.item(row) or not self.classes_list.item(row).data(Qt.UserRole):
            # Invalid selection
            self.log_to_console("Error: Invalid class item selected")
            return
        
        class_id = self.classes_list.item(row).data(Qt.UserRole)
        self.log_to_console(f"Class selected: ID {class_id}")
        
        # Get the class data from the database
        selected_class = self.db.get_class(class_id)
        if not selected_class:
            QMessageBox.warning(self, "Error", "Could not find the selected class in the database.")
            self.log_to_console(f"Error: Class with ID {class_id} not found in database")
            return
        
        # Update the UI with class information
        self.class_title_label.setText(selected_class['name'])
        self.class_date_label.setText(f"Date: {selected_class['date']}")
        
        # Make sure buttons are in their default state
        self.record_button.setEnabled(True)
        
        # Reset any previous "Transcribing..." state if needed
        self.transcribe_button.setText("Transcribe")
        
        # Update transcript and summary if available
        if selected_class.get('transcript'):
            # Clear and then set the transcript text to ensure it triggers UI update
            self.transcript_text.clear()
            QApplication.processEvents()  # Process events to ensure UI updates
            self.transcript_text.setText(selected_class['transcript'])
            
            # Enable the summarize button since we have a transcript
            self.summarize_button.setEnabled(True)
            
            # Log with length info for debugging
            transcript_length = len(selected_class['transcript'])
            self.log_to_console(f"Loaded existing transcript ({transcript_length} characters)")
        else:
            self.transcript_text.clear()
            self.log_to_console("No transcript available for this class")
            self.summarize_button.setEnabled(False)
        
        if selected_class.get('summary'):
            # Clear and then set the summary text, applying Markdown formatting
            self.summary_text.clear()
            QApplication.processEvents()  # Process events to ensure UI updates
            
            # Convert markdown to HTML
            formatted_summary = self.format_markdown_text(selected_class['summary'])
            self.summary_text.setHtml(formatted_summary)
            
            self.log_to_console("Summary available for this class")
        else:
            self.summary_text.clear()
            self.log_to_console("No summary available for this class")
        
        # Check for audio recording to enable/disable transcribe button
        has_audio = False
        if selected_class.get('audio_path') and os.path.exists(selected_class['audio_path']):
            # Audio file exists
            has_audio = True
            self.transcribe_button.setText("Transcribe")
            self.transcribe_button.setEnabled(True)
            
            # Show audio file information in the console
            audio_file = os.path.basename(selected_class['audio_path'])
            file_size_bytes = os.path.getsize(selected_class['audio_path'])
            file_size_mb = file_size_bytes / (1024 * 1024)
            self.log_to_console(f"Audio recording available: {audio_file} ({file_size_mb:.2f} MB)")
            
            # Create audio player UI
            self.create_audio_player_ui(selected_class['audio_path'], audio_file)
        else:
            # No audio file
            self.transcribe_button.setText("No Audio - Record First")
            self.transcribe_button.setEnabled(False)
            self.log_to_console("No audio recording available for this class. Please record audio first.")
            
            # Clear any previous audio player
            if hasattr(self, 'audio_player_widget'):
                self.audio_player_widget.setParent(None)
                self.audio_player_widget = None
    
    def create_audio_player_ui(self, audio_path, audio_filename):
        """Create a mini audio player UI for the given audio file."""
        # Create a widget to hold the audio player UI if it doesn't exist
        if hasattr(self, 'audio_player_widget'):
            self.audio_player_widget.setParent(None)
        
        self.audio_player_widget = QWidget()
        player_layout = QVBoxLayout(self.audio_player_widget)
        player_layout.setContentsMargins(10, 10, 10, 10)
        
        # Apply styling to make the player stand out
        is_dark = self.settings.get('theme') == 'dark'
        if is_dark:
            self.audio_player_widget.setStyleSheet("""
                QWidget {
                    background-color: #3a3a3a;
                    border-radius: 6px;
                    padding: 5px;
                }
                QLabel {
                    color: #e0e0e0;
                }
                QSlider::handle:horizontal {
                    background: #6c9ef8;
                    border: none;
                    width: 10px;
                    margin: -4px 0;
                    border-radius: 5px;
                }
                QSlider::groove:horizontal {
                    border: 1px solid #5c5c5c;
                    height: 4px;
                    background: #2a2a2a;
                    margin: 0px;
                    border-radius: 2px;
                }
            """)
        else:
            self.audio_player_widget.setStyleSheet("""
                QWidget {
                    background-color: #f0f0f0;
                    border: 1px solid #d0d0d0;
                    border-radius: 6px;
                    padding: 5px;
                }
                QSlider::handle:horizontal {
                    background: #0078d7;
                    border: none;
                    width: 10px;
                    margin: -4px 0;
                    border-radius: 5px;
                }
                QSlider::groove:horizontal {
                    border: 1px solid #b0b0b0;
                    height: 4px;
                    background: #e0e0e0;
                    margin: 0px;
                    border-radius: 2px;
                }
            """)
        
        # Add file info with an icon
        file_info_layout = QHBoxLayout()
        file_label = QLabel(f"🎙️ Recording: {audio_filename}")
        font = file_label.font()
        font.setBold(True)
        file_label.setFont(font)
        file_info_layout.addWidget(file_label)
        player_layout.addLayout(file_info_layout)
        
        # Add playback controls
        controls_layout = QHBoxLayout()
        
        # Play/pause button
        self.play_button = QToolButton()
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.clicked.connect(self.toggle_playback)
        self.play_button.setToolTip("Play/Pause (Spacebar)")
        self.play_button.setIconSize(QSize(24, 24))
        controls_layout.addWidget(self.play_button)
        
        # Position slider
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self.set_position)
        self.position_slider.setToolTip("Seek")
        controls_layout.addWidget(self.position_slider)
        
        # Time display
        self.time_label = QLabel("0:00 / 0:00")
        controls_layout.addWidget(self.time_label)
        
        # Volume control
        volume_button = QToolButton()
        volume_button.setIcon(self.style().standardIcon(QStyle.SP_MediaVolume))
        volume_button.setToolTip("Volume")
        controls_layout.addWidget(volume_button)
        
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(70)  # Default volume: 70%
        self.volume_slider.setMaximumWidth(100)
        self.volume_slider.valueChanged.connect(self.set_volume)
        self.volume_slider.setToolTip("Adjust Volume")
        controls_layout.addWidget(self.volume_slider)
        
        player_layout.addLayout(controls_layout)
        
        # Removed the tip text about spacebar shortcut
        
        # Insert the audio player widget at the top of the right content area
        if self.content_layout.count() > 0:
            # Insert before the first item (the tab widget)
            self.content_layout.insertWidget(0, self.audio_player_widget)
        else:
            self.content_layout.addWidget(self.audio_player_widget)
        
        # Set the media content
        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(audio_path)))
        self.set_volume(self.volume_slider.value())
        
        # Add keyboard shortcut for play/pause
        self.spacebar_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.spacebar_shortcut.activated.connect(self.toggle_playback)
        
        # Update UI
        self.update_duration(self.media_player.duration())
    
    def toggle_playback(self):
        """Toggle between play and pause states."""
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()
    
    def media_state_changed(self, state):
        """Handle media state changes to update the play/pause button."""
        if state == QMediaPlayer.PlayingState:
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
    
    def update_position(self, position):
        """Update the slider position and time display."""
        # Disable the signals to avoid a loop
        self.position_slider.blockSignals(True)
        self.position_slider.setValue(position)
        self.position_slider.blockSignals(False)
        
        # Update the time display
        duration = self.media_player.duration()
        if duration > 0:
            current_time = self.format_time(position)
            total_time = self.format_time(duration)
            self.time_label.setText(f"{current_time} / {total_time}")
    
    def update_duration(self, duration):
        """Update the slider range when media duration changes."""
        self.position_slider.setRange(0, duration)
        
        # Update the time display
        if duration > 0:
            current_time = self.format_time(self.media_player.position())
            total_time = self.format_time(duration)
            self.time_label.setText(f"{current_time} / {total_time}")
    
    def set_position(self, position):
        """Set the playback position."""
        self.media_player.setPosition(position)
    
    def set_volume(self, volume):
        """Set the audio volume."""
        self.media_player.setVolume(volume)
    
    def format_time(self, milliseconds):
        """Format time in milliseconds to mm:ss format."""
        seconds = int(milliseconds / 1000)
        minutes = seconds // 60
        seconds %= 60
        return f"{minutes}:{seconds:02d}"
    
    def add_subject(self):
        """Add a new subject."""
        dialog = AddSubjectDialog(self)
        if dialog.exec_():
            subject_name = dialog.get_subject_name()
            if subject_name:
                subject_id = self.db.add_subject(subject_name)
                if subject_id:
                    self.load_subjects()
                    
                    # Select the new subject
                    for i in range(self.subject_combo.count()):
                        if self.subject_combo.itemData(i) == subject_id:
                            self.subject_combo.setCurrentIndex(i)
                            break
                else:
                    QMessageBox.warning(self, "Error", f"Subject '{subject_name}' already exists.")
    
    def delete_subject(self):
        """Delete the selected subject."""
        index = self.subject_combo.currentIndex()
        if index <= 0:
            QMessageBox.warning(self, "Warning", "Please select a subject to delete.")
            return
        
        subject_id = self.subject_combo.currentData()
        subject_name = self.subject_combo.currentText()
        
        reply = QMessageBox.question(
            self, 
            "Confirm Deletion",
            f"Are you sure you want to delete the subject '{subject_name}' and all its classes?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            num_deleted = self.db.delete_subject(subject_id)
            self.load_subjects()
            QMessageBox.information(self, "Subject Deleted", f"Subject '{subject_name}' and its classes have been deleted.")
    
    def add_class(self):
        """Add a new class to the selected subject."""
        index = self.subject_combo.currentIndex()
        if index <= 0:
            QMessageBox.warning(self, "Warning", "Please select a subject first.")
            return
        
        subject_id = self.subject_combo.currentData()
        
        dialog = AddClassDialog(self)
        if dialog.exec_():
            class_info = dialog.get_class_info()
            if class_info['name']:
                class_id = self.db.add_class(
                    subject_id,
                    class_info['name'],
                    class_info['date'],
                    class_info['chapter']
                )
                
                self.load_classes(subject_id)
                
                # Select the new class
                for row in range(self.classes_list.count()):
                    if self.classes_list.item(row).data(Qt.UserRole) == class_id:
                        self.classes_list.setCurrentRow(row)
                        break
            else:
                QMessageBox.warning(self, "Error", "Class name cannot be empty.")
    
    def delete_class(self):
        """Delete the selected class."""
        row = self.classes_list.currentRow()
        if row < 0:
            QMessageBox.warning(self, "Warning", "Please select a class to delete.")
            return
        
        class_id = self.classes_list.item(row).data(Qt.UserRole)
        class_name = self.classes_list.item(row).text()
        
        reply = QMessageBox.question(
            self, 
            "Confirm Deletion",
            f"Are you sure you want to delete the class '{class_name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            num_deleted = self.db.delete_class(class_id)
            subject_id = self.subject_combo.currentData()
            self.load_classes(subject_id)
            QMessageBox.information(self, "Class Deleted", f"Class '{class_name}' has been deleted.")
    
    def record_audio(self):
        """Record audio for the selected class."""
        # Validate class selection
        row = self.classes_list.currentRow()
        if row < 0:
            QMessageBox.warning(self, "Error", "Please select a class first.")
            return
        
        # Get class ID
        if not self.classes_list.item(row) or not self.classes_list.item(row).data(Qt.UserRole):
            QMessageBox.warning(self, "Error", "Invalid class selection. Please try selecting the class again.")
            return
        
        class_id = self.classes_list.item(row).data(Qt.UserRole)
        
        # Show recording dialog
        dialog = RecordDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            # Log that recording was successful
            self.log_to_console("Recording saved successfully")
            
            # Get the recorded audio path
            recording_path = dialog.get_recording_path()
            
            # Verify the recording file exists
            if not os.path.exists(recording_path):
                QMessageBox.warning(self, "Error", f"Recording file not found at: {recording_path}")
                self.log_to_console(f"Error: Recording file not found: {recording_path}")
                return
            
            # Convert to absolute path to avoid any path issues
            recording_path = os.path.abspath(recording_path)
            
            # Save to database
            self.log_to_console(f"Saving recording path to database: {recording_path}")
            if not self.db.update_class(class_id, audio_path=recording_path):
                QMessageBox.warning(self, "Database Error", 
                                   "Failed to update the database with the recording path.\n"
                                   "Please try recording again.")
                self.log_to_console(f"Error: Failed to update database with recording path")
                return
            
            # Get the updated class info to verify
            updated_class = self.db.get_class(class_id)
            if not updated_class or not updated_class.get('audio_path'):
                QMessageBox.warning(self, "Database Error", 
                                   "Failed to update the database with the recording path.\n"
                                   "Please try recording again.")
                self.log_to_console(f"Error: Failed to update database with recording path")
                return
            
            # Verify the paths match
            if updated_class['audio_path'] != recording_path:
                self.log_to_console(f"Warning: Database path ({updated_class['audio_path']}) doesn't match recording path ({recording_path})")
            
            # Enable transcribe button and update its text
            self.transcribe_button.setText("Transcribe")
            self.transcribe_button.setEnabled(True)
            
            # Don't just call refresh_selected_class, update the UI directly
            try:
                # Update UI directly if needed
                self.log_to_console("Updating UI after recording")
                
                # Update class title and info
                self.class_title_label.setText(updated_class['name'])
                class_date = updated_class.get('date', '')
                if class_date:
                    try:
                        # Try to format the date nicely
                        date_obj = datetime.datetime.strptime(class_date, "%Y-%m-%d")
                        formatted_date = date_obj.strftime("%B %d, %Y")
                        self.class_date_label.setText(f"Date: {formatted_date}")
                    except:
                        self.class_date_label.setText(f"Date: {class_date}")
                
                # Force UI update to reflect recording state
                QApplication.processEvents()
                
                # Show confirmation message
                QMessageBox.information(self, "Recording Saved", 
                                       "Audio recording saved successfully.\n\n"
                                       "You can now click 'Transcribe' to convert the audio to text.")
            except Exception as ui_ex:
                logging.error(f"Error updating UI after recording: {str(ui_ex)}", exc_info=True)
                # Fall back to refresh_selected_class if direct update fails
                self.refresh_selected_class()
                
                # Show confirmation message
                QMessageBox.information(self, "Recording Saved", 
                                       "Audio recording saved successfully.\n\n"
                                       "You can now click 'Transcribe' to convert the audio to text.")
    
    def transcribe_audio(self):
        """Transcribe the recorded audio."""
        try:
            # Safety check 1: Make sure a class is selected
            row = self.classes_list.currentRow()
            if row < 0:
                QMessageBox.warning(self, "Error", "Please select a class first.")
                return
            
            # Safety check 2: Get class ID properly
            if not self.classes_list.item(row) or not self.classes_list.item(row).data(Qt.UserRole):
                QMessageBox.warning(self, "Error", "Invalid class selection. Please try selecting the class again.")
                return
            
            class_id = self.classes_list.item(row).data(Qt.UserRole)
            
            # Safety check 3: Validate class exists in database
            all_classes = self.db.get_classes()
            if not all_classes:
                QMessageBox.warning(self, "Error", "No classes found in database.")
                return
            
            selected_class = next((c for c in all_classes if c['id'] == class_id), None)
            
            if not selected_class:
                QMessageBox.warning(self, "Error", "Could not find the selected class in the database.")
                return
            
            # Safety check 4: Check audio path exists in database
            if not selected_class.get('audio_path'):
                QMessageBox.warning(self, "No Audio Recording", 
                                   "This class doesn't have an audio recording.\n\n"
                                   "Please click 'Record Audio' first to create a recording for this class.")
                
                # Update UI to clearly show that recording is needed
                self.transcribe_button.setText("No Audio - Record First")
                self.transcribe_button.setEnabled(False)
                return
            
            # Safety check 5: Validate audio file exists on disk
            audio_path = selected_class['audio_path']
            if not os.path.exists(audio_path):
                QMessageBox.warning(self, "Error", 
                                   f"Audio file not found: {audio_path}\n\n"
                                   "The recording may have been moved or deleted. Please record a new audio file.")
                
                # Update database to clear the invalid audio path
                self.db.update_class(class_id, audio_path=None)
                
                # Update UI
                self.transcribe_button.setText("No Audio - Record First")
                self.transcribe_button.setEnabled(False)
                return

            # Update UI to show transcription is in progress
            self.transcribe_button.setText("Transcribing...")
            self.transcribe_button.setEnabled(False)
            
            # Show progress UI
            self._show_progress(
                active=True, 
                message="Transcribing audio...", 
                process_type="transcription"
            )
            
            # Set up signals for thread-safe communication
            self.transcription_signals = TranscriptionSignals()
            self.transcription_signals.finished.connect(self._handle_transcription_result)
            
            # Reset cancellation flag
            self.transcription_cancelled = False
            
            # Transcription worker thread
            def transcription_worker():
                transcript = None
                error_occurred = False
                error_message = ""
                
                try:
                    # Check for cancellation before starting
                    if hasattr(self, 'transcription_cancelled') and self.transcription_cancelled:
                        logging.info("Transcription was cancelled before it started")
                        self.transcription_signals.finished.emit("", class_id, True, "Cancelled by user")
                        return
                    
                    # Perform transcription
                    transcript = self.ai_processor.transcribe_audio(audio_path)
                    
                    # Check for cancellation after transcription
                    if hasattr(self, 'transcription_cancelled') and self.transcription_cancelled:
                        logging.info("Transcription was cancelled after completion")
                        self.transcription_signals.finished.emit("", class_id, True, "Cancelled by user")
                        return
                    
                    # Update database if successful
                    if transcript:
                        self.db.update_class(class_id, transcript=transcript)
                        logging.info(f"Transcription succeeded, updating database for class ID {class_id}")
                    else:
                        error_occurred = True
                        error_message = "Transcription produced no result"
                except Exception as e:
                    error_occurred = True
                    error_message = str(e)
                    logging.error(f"Transcription error: {str(e)}", exc_info=True)
                
                # Emit signal with result
                self.transcription_signals.finished.emit(transcript, class_id, error_occurred, error_message)
            
            # Start worker thread
            self.transcription_thread = threading.Thread(target=transcription_worker, daemon=True)
            self.transcription_thread.start()
            
        except Exception as e:
            # Global error handler
            error_msg = f"Error starting transcription: {str(e)}"
            logging.error(error_msg, exc_info=True)
            QMessageBox.critical(self, "Error", error_msg)
            
            # Reset UI state
            self.transcribe_button.setText("Transcribe")
            self.transcribe_button.setEnabled(True)
            self._show_progress(active=False, process_type="transcription")
    
    def _handle_transcription_result(self, transcript, class_id, error_occurred, error_message):
        """Handle the result of transcription (called on main thread via signal)"""
        logging.info(f"Handling transcription result on main thread: class_id={class_id}, error={error_occurred}")
        
        # DEBUG: Print the current thread ID and state
        import threading
        logging.info(f"Thread ID: {threading.get_ident()}, Main thread: {threading.main_thread().ident}")
        
        # First, update UI state
        self.transcribe_button.setText("Transcribe")
        self.transcribe_button.setEnabled(True)
        self._show_progress(active=False, process_type="transcription")
        logging.info("Progress UI hidden after transcription")
        
        # Handle cancellation
        if hasattr(self, 'transcription_cancelled') and self.transcription_cancelled:
            self.log_to_console("Transcription was cancelled by user")
            return
        
        # Handle errors
        if error_occurred:
            self.log_to_console(f"Transcription failed: {error_message}")
            QMessageBox.critical(self, "Transcription Error", 
                              f"An error occurred during transcription: {error_message}")
            return
        
        # Handle empty transcript
        if not transcript:
            self.log_to_console("Transcription produced no results")
            QMessageBox.warning(self, "Transcription Warning", 
                             "The transcription process completed but produced no text.")
            return
        
        # Get fresh class data from database
        logging.info(f"Getting fresh class data for ID {class_id}")
        class_data = self.db.get_class(class_id)
        if not class_data:
            logging.error(f"Could not find class {class_id} in database after transcription")
            QMessageBox.warning(self, "Database Error", 
                             "Could not retrieve updated class data after transcription.")
            return
        
        # Log success
        self.log_to_console("Transcription successful, updating UI")
        logging.info(f"Transcript length: {len(transcript)} characters")
        
        # Check if the user changed the selected class during transcription
        current_row = self.classes_list.currentRow()
        current_class_id = -1
        if current_row >= 0 and self.classes_list.item(current_row):
            current_class_id = self.classes_list.item(current_row).data(Qt.UserRole)
        
        logging.info(f"Currently selected class ID: {current_class_id}, Transcribed class ID: {class_id}")
        
        # If the current class selection matches the transcribed class, update the UI
        if current_class_id == class_id:
            # Update transcript text area
            logging.info("Updating transcript display for current selection")
            self.transcript_text.clear()
            self.transcript_text.setText(transcript)
            
            # Process events to ensure UI updates
            QApplication.processEvents()
            
            # Verify the text was actually updated
            current_text = self.transcript_text.toPlainText()
            logging.info(f"Current transcript text length: {len(current_text)}")
            
            # Enable summarize button
            self.summarize_button.setEnabled(True)
            
            # Switch to transcript tab
            self.content_tabs.setCurrentIndex(0)
        else:
            # If the user selected a different class, find the item for our class and select it
            logging.info(f"User changed class selection during transcription. Switching back to class {class_id}")
            
            # Find the item for our class
            for i in range(self.classes_list.count()):
                item_class_id = self.classes_list.item(i).data(Qt.UserRole)
                if item_class_id == class_id:
                    # Select this item which will trigger on_class_selected
                    logging.info(f"Found class at row {i}, selecting it")
                    self.classes_list.setCurrentRow(i)
                    break
        
        # Show success message
        QMessageBox.information(self, "Transcription Complete", 
                             "Audio has been successfully transcribed.")

    def summarize_transcript(self):
        """Summarize the transcript using Ollama."""
        try:
            # Safety check 1: Make sure a class is selected
            row = self.classes_list.currentRow()
            if row < 0:
                QMessageBox.warning(self, "Error", "Please select a class first.")
                return
            
            # Safety check 2: Get class ID properly
            if not self.classes_list.item(row) or not self.classes_list.item(row).data(Qt.UserRole):
                QMessageBox.warning(self, "Error", "Invalid class selection. Please try selecting the class again.")
                return
            
            class_id = self.classes_list.item(row).data(Qt.UserRole)
            
            # Get class data
            class_data = self.db.get_class(class_id)
            if not class_data:
                QMessageBox.warning(self, "Error", "Could not find the selected class in the database.")
                return
            
            # Check if transcript exists
            if not class_data.get('transcript'):
                QMessageBox.warning(self, "Error", "No transcript found for this class.")
                return
            
            # Store transcript text for processing
            transcript_text = class_data['transcript']
            
            # Update UI state
            self.summarize_button.setText("Summarizing...")
            self.summarize_button.setEnabled(False)
            
            # Show the integrated progress UI 
            self._show_progress(
                active=True, 
                message="Generating summary from transcript...",
                process_type="summarization"
            )
            
            # Reset cancellation flag
            self.summarization_cancelled = False
            
            # Set up signals for thread-safe communication
            self.summarization_signals = SummarizationSignals()
            self.summarization_signals.finished.connect(self._handle_summarization_result)
            
            # Summarization worker thread
            def summarization_worker():
                summary = None
                error_occurred = False
                error_message = ""
                
                try:
                    # Check for cancellation before starting
                    if hasattr(self, 'summarization_cancelled') and self.summarization_cancelled:
                        logging.info("Summarization was cancelled before it started")
                        self.summarization_signals.finished.emit("", class_id, True, "Cancelled by user")
                        return
                    
                    # Log start time
                    start_time = time.time()
                    
                    # Perform summarization
                    summary = self.ai_processor.summarize_text(transcript_text)
                    
                    # Check for cancellation after summarization
                    if hasattr(self, 'summarization_cancelled') and self.summarization_cancelled:
                        logging.info("Summarization was cancelled after completion")
                        self.summarization_signals.finished.emit("", class_id, True, "Cancelled by user")
                        return
                    
                    # Log completion time
                    elapsed_time = time.time() - start_time
                    logging.info(f"Summarization completed in {elapsed_time:.2f} seconds")
                    
                    # Update database if successful
                    if summary:
                        self.db.update_class(class_id, summary=summary)
                    else:
                        error_occurred = True
                        error_message = "Summarization produced no result"
                except Exception as e:
                    error_occurred = True
                    error_message = str(e)
                    logging.error(f"Summarization error: {str(e)}", exc_info=True)
                
                # Emit signal with result
                self.summarization_signals.finished.emit(summary, class_id, error_occurred, error_message)
            
            # Start worker thread
            self.summarization_thread = threading.Thread(target=summarization_worker, daemon=True)
            self.summarization_thread.start()
            
        except Exception as e:
            # Global error handler
            error_msg = f"Error starting summarization: {str(e)}"
            logging.error(error_msg, exc_info=True)
            QMessageBox.critical(self, "Error", error_msg)
            
            # Reset UI state
            self.summarize_button.setText("Summarize")
            self.summarize_button.setEnabled(True)
            self._show_progress(active=False, process_type="summarization")
    
    def _handle_summarization_result(self, summary, class_id, error_occurred, error_message):
        """Handle the result of summarization (called on main thread via signal)"""
        logging.info(f"Handling summarization result on main thread: class_id={class_id}, error={error_occurred}")
        
        # First, update UI state
        self.summarize_button.setText("Summarize")
        self.summarize_button.setEnabled(True)
        self._show_progress(active=False, process_type="summarization")
        logging.info("Progress UI hidden after summarization")
        
        # Handle cancellation
        if hasattr(self, 'summarization_cancelled') and self.summarization_cancelled:
            self.log_to_console("Summarization was cancelled by user")
            return
        
        # Handle errors
        if error_occurred:
            self.log_to_console(f"Summarization failed: {error_message}")
            QMessageBox.critical(self, "Summarization Error", 
                              f"An error occurred during summarization: {error_message}")
            return
        
        # Handle empty summary
        if not summary:
            self.log_to_console("Summarization produced no results")
            QMessageBox.warning(self, "Summarization Warning", 
                             "The summarization process completed but produced no text.")
            return
        
        # Get fresh class data from database
        logging.info(f"Getting fresh class data for ID {class_id}")
        class_data = self.db.get_class(class_id)
        if not class_data:
            logging.error(f"Could not find class {class_id} in database after summarization")
            QMessageBox.warning(self, "Database Error", 
                             "Could not retrieve updated class data after summarization.")
            return
        
        # Log success
        self.log_to_console("Summarization successful, updating UI")
        logging.info(f"Summary length: {len(summary)} characters")
        
        # Format the summary with Markdown-to-HTML conversion
        formatted_summary = self.format_markdown_text(summary)
        
        # Check if the user changed the selected class during summarization
        current_row = self.classes_list.currentRow()
        current_class_id = -1
        if current_row >= 0 and self.classes_list.item(current_row):
            current_class_id = self.classes_list.item(current_row).data(Qt.UserRole)
        
        logging.info(f"Currently selected class ID: {current_class_id}, Summarized class ID: {class_id}")
        
        # If the current class selection matches the summarized class, update the UI
        if current_class_id == class_id:
            # Update summary text area
            logging.info("Updating summary display for current selection")
            self.summary_text.clear()
            # Set as HTML to enable formatting
            self.summary_text.setHtml(formatted_summary)
            
            # Process events to ensure UI updates
            QApplication.processEvents()
            
            # Verify the text was actually updated
            current_text = self.summary_text.toPlainText()
            logging.info(f"Current summary text length: {len(current_text)}")
            
            # Switch to summary tab
            self.content_tabs.setCurrentIndex(1)
        else:
            # If the user selected a different class, find the item for our class and select it
            logging.info(f"User changed class selection during summarization. Switching back to class {class_id}")
            
            # Find the item for our class
            for i in range(self.classes_list.count()):
                item_class_id = self.classes_list.item(i).data(Qt.UserRole)
                if item_class_id == class_id:
                    # Select this item which will trigger on_class_selected
                    logging.info(f"Found class at row {i}, selecting it")
                    self.classes_list.setCurrentRow(i)
                    
                    # After selection, switch to summary tab
                    QTimer.singleShot(300, lambda: self.content_tabs.setCurrentIndex(1))
                    break
        
        # Show success message
        QMessageBox.information(self, "Summarization Complete", 
                             "Transcript has been successfully summarized.")

    def format_markdown_text(self, text):
        """Convert Markdown syntax to HTML for display"""
        if not text:
            return ""
        
        # Replace **text** with <b>text</b> for bold
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        
        # Replace *text* with <i>text</i> for italic
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        
        # Replace `text` with <code>text</code> for inline code
        text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
        
        # Replace ~~text~~ with <s>text</s> for strikethrough
        text = re.sub(r'~~(.*?)~~', r'<s>\1</s>', text)
        
        # Replace newlines with <br> tags
        text = text.replace('\n', '<br>')
        
        return text
    
    def check_ollama(self):
        """Check if Ollama is available and the required model is installed."""
        # Run in a separate thread to avoid blocking the UI
        def check_thread():
            availability, message = self.ai_processor.check_ollama_model()
            
            # Update UI on the main thread
            def show_message():
                if not availability:
                    # If the model is not available, update settings to the default model
                    if "model not found" in message.lower():
                        self.log_to_console(f"Model '{self.settings.get('ollama_model')}' not found in Ollama.")
                        # Update UI on settings tab if it exists
                        if hasattr(self, 'ollama_model_combo'):
                            # Schedule another load of models to update the combobox
                            QTimer.singleShot(1000, self.load_ollama_models)
                    
                    QMessageBox.warning(
                        self, 
                        "Ollama Model Not Available", 
                        f"{message}\n\nPlease install Ollama and the required model to use all features."
                    )
            
            # Schedule message on the main thread if needed
            if not availability:
                QTimer.singleShot(0, show_message)
        
        # Start the thread
        threading.Thread(target=check_thread).start()
    
    def _cancel_current_process(self):
        """Cancel the current process (transcription or summarization)."""
        if hasattr(self, 'transcription_active') and self.transcription_active:
            self.transcription_cancelled = True
            self.log_to_console("Canceling transcription... This may take a moment.")
            self.progress_label.setText("Canceling transcription...")
            
            # Force cleanup after a timeout if the thread doesn't respond
            QTimer.singleShot(5000, lambda: self._show_progress(active=False, process_type="transcription") 
                             if hasattr(self, 'transcription_active') and self.transcription_active else None)
            
        elif hasattr(self, 'summarization_active') and self.summarization_active:
            self.summarization_cancelled = True
            self.log_to_console("Canceling summarization... This may take a moment.")
            self.progress_label.setText("Canceling summarization...")
            
            # Force cleanup after a timeout if the thread doesn't respond
            QTimer.singleShot(5000, lambda: self._show_progress(active=False, process_type="summarization")
                             if hasattr(self, 'summarization_active') and self.summarization_active else None)
    
    def toggle_safe_mode(self, state):
        """Toggle safe mode for transcription."""
        if hasattr(self, 'ai_processor'):
            self.ai_processor.safe_mode = state == Qt.Checked
            mode = "enabled" if self.ai_processor.safe_mode else "disabled"
            self.log_to_console(f"Safe mode {mode} for transcription")
            logging.info(f"Safe mode {mode} for transcription")

    def log_to_console(self, message):
        """Log a message to the console tab."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        # Log to console tab
        if hasattr(self, 'logs_text'):
            # Use QTimer to ensure we update on the main thread
            QTimer.singleShot(0, lambda: self.logs_text.append(formatted_message))
        
        # Also log to file
        logging.info(message)
        
        # Check if we should switch to logs tab
        if "error" in message.lower() or "failed" in message.lower() or "warning" in message.lower():
            # Switch to logs tab to show the error
            QTimer.singleShot(0, lambda: self.content_tabs.setCurrentIndex(2))  # Assuming logs tab is at index 2

    def refresh_selected_class(self):
        """Refresh the currently selected class information."""
        row = self.classes_list.currentRow()
        if row >= 0:
            # Re-trigger the selection event to refresh all UI elements
            self.on_class_selected()

    def _update_progress_pulse(self):
        """Update the progress dialog with a pulse effect to show ongoing activity."""
        if hasattr(self, 'progress_dialog') and self.progress_dialog is not None:
            # Create a pulsing effect by cycling between 0 and 100
            current_value = self.progress_dialog.value()
            if current_value >= 90:
                self.progress_dialog.setValue(0)
            else:
                self.progress_dialog.setValue(current_value + 10)
            
            # Update the label text
            if hasattr(self, 'summarization_cancelled') and self.summarization_cancelled:
                self.progress_dialog.setLabelText("Cancelling summarization...")
            else:
                self.progress_dialog.setLabelText(f"Generating summary... (This may take a while)")

    def _show_progress(self, active=True, message="Processing", process_type=""):
        """Show or hide the progress UI."""
        if active:
            # Show progress UI
            self.progress_widget.setVisible(True)
            self.progress_label.setText(message)
            
            # Set process flag
            if process_type == "transcription":
                self.transcription_active = True
                self.transcription_cancelled = False
            elif process_type == "summarization":
                self.summarization_active = True
                self.summarization_cancelled = False
                
            # Start progress timer if not already running
            if not hasattr(self, 'progress_update_timer') or not self.progress_update_timer.isActive():
                self.progress_update_timer = QTimer()
                self.progress_update_timer.timeout.connect(self._update_progress_animation)
                self.progress_update_timer.start(100)  # Update every 100ms
        else:
            # Hide progress UI
            self.progress_widget.setVisible(False)
            
            # Reset process flags
            if process_type == "transcription":
                self.transcription_active = False
            elif process_type == "summarization":
                self.summarization_active = False
                
            # Stop timer if no active processes
            if (not hasattr(self, 'transcription_active') or not self.transcription_active) and \
               (not hasattr(self, 'summarization_active') or not self.summarization_active):
                if hasattr(self, 'progress_update_timer') and self.progress_update_timer.isActive():
                    self.progress_update_timer.stop()
    
    def _update_progress_animation(self):
        """Update progress bar with an animation effect to show ongoing activity."""
        if self.progress_indicator.isVisible():
            # Update icon animation (we don't need to update the progress bar as it's in indeterminate mode)
            icons = ["⬤", "⬤", "⬤", "⬤", "⬤"]
            if hasattr(self, 'icon_index'):
                self.icon_index = (self.icon_index + 1) % len(icons)
            else:
                self.icon_index = 0
                
            # Don't update icon if cancellation is in progress
            if (hasattr(self, 'transcription_cancelled') and self.transcription_cancelled) or \
               (hasattr(self, 'summarization_cancelled') and self.summarization_cancelled):
                self.progress_indicator.setText("⚠️")
            else:
                self.progress_indicator.setText(icons[self.icon_index])

    def create_menu(self):
        """Create the application menu."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        # New subject action
        new_subject_action = QAction("New Subject", self)
        new_subject_action.triggered.connect(self.add_subject)
        file_menu.addAction(new_subject_action)
        
        # New class action
        new_class_action = QAction("New Class", self)
        new_class_action.triggered.connect(self.add_class)
        file_menu.addAction(new_class_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Recording menu
        recording_menu = menubar.addMenu("Recording")
        
        # Record audio action
        record_action = QAction("Record Audio", self)
        record_action.triggered.connect(self.record_audio)
        recording_menu.addAction(record_action)
        
        # Transcribe action
        transcribe_action = QAction("Transcribe", self)
        transcribe_action.triggered.connect(self.transcribe_audio)
        recording_menu.addAction(transcribe_action)
        
        # Summarize action
        summarize_action = QAction("Summarize", self)
        summarize_action.triggered.connect(self.summarize_transcript)
        recording_menu.addAction(summarize_action)
        
        # Settings menu
        settings_menu = menubar.addMenu("Settings")
        
        # Theme submenu
        theme_menu = settings_menu.addMenu("Theme")
        
        # Light theme action
        light_theme_action = QAction("Light Theme", self)
        light_theme_action.triggered.connect(lambda: self.on_theme_changed("light"))
        theme_menu.addAction(light_theme_action)
        
        # Dark theme action
        dark_theme_action = QAction("Dark Theme", self)
        dark_theme_action.triggered.connect(lambda: self.on_theme_changed("dark"))
        theme_menu.addAction(dark_theme_action)
        
        settings_menu.addSeparator()
        
        # Settings action
        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self.show_settings_dialog)
        settings_menu.addAction(settings_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        # About action
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        # Check Ollama action
        check_ollama_action = QAction("Check Ollama", self)
        check_ollama_action.triggered.connect(self.check_ollama)
        help_menu.addAction(check_ollama_action)
        
        return menubar

    def show_about(self):
        """Show the about dialog."""
        QMessageBox.about(
            self,
            "About Class Transcriber",
            """<h1>Class Transcriber</h1>
            <p>A tool for recording, transcribing, and summarizing class lectures.</p>
            <p>Uses Whisper for transcription and Ollama for summarization.</p>
            <p>Version 1.0</p>"""
        )

    def show_settings_dialog(self):
        """Show the global settings dialog"""
        dialog = SettingsDialog(self, self.db, self.ai_processor)
        if dialog.exec_() == QDialog.Accepted:
            # Settings were saved in the dialog, refresh the UI
            self.settings = self.db.get_settings()
            self.log_to_console("Settings updated")

    def on_theme_changed(self, theme):
        """Handle theme change."""
        if theme != self.settings.get('theme'):
            # Update settings
            self.db.update_setting('theme', theme)
            self.settings['theme'] = theme
            
            # Apply the theme
            self.apply_theme(theme)
            
            # Log the theme change
            self.log_to_console(f"Changed theme to {theme}")
    
    def apply_theme(self, theme):
        """Apply the selected theme to the application."""
        app = QApplication.instance()
        
        if theme == 'dark':
            # Apply dark theme
            app.setStyle("Fusion")
            
            # Define dark palette
            dark_palette = QPalette()
            
            # Colors for dark theme
            dark_color = QColor(45, 45, 45)
            disabled_color = QColor(70, 70, 70)
            highlight_color = QColor(42, 130, 218)
            
            # Set colors based on color group and role
            dark_palette.setColor(QPalette.Window, dark_color)
            dark_palette.setColor(QPalette.WindowText, Qt.white)
            dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
            dark_palette.setColor(QPalette.AlternateBase, dark_color)
            dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
            dark_palette.setColor(QPalette.ToolTipText, Qt.white)
            dark_palette.setColor(QPalette.Text, Qt.white)
            dark_palette.setColor(QPalette.Button, dark_color)
            dark_palette.setColor(QPalette.ButtonText, Qt.white)
            dark_palette.setColor(QPalette.BrightText, Qt.red)
            dark_palette.setColor(QPalette.Link, highlight_color)
            dark_palette.setColor(QPalette.Highlight, highlight_color)
            dark_palette.setColor(QPalette.HighlightedText, Qt.black)
            
            # Disabled colors
            dark_palette.setColor(QPalette.Disabled, QPalette.Text, QColor(128, 128, 128))
            dark_palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(128, 128, 128))
            dark_palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(128, 128, 128))
            
            # Apply the palette
            app.setPalette(dark_palette)
            
            # Style sheets for specific controls
            app.setStyleSheet("""
                QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }
                QTabWidget::pane { border: 1px solid #444; }
                QTabBar::tab { background: #333; color: #fff; padding: 5px; }
                QTabBar::tab:selected { background: #444; }
                QHeaderView::section { background-color: #333; color: white; }
            """)
            
            # Log the theme change
            logging.info("Applied dark theme")
        else:
            # Apply light theme (default)
            app.setStyle("Fusion")
            app.setPalette(app.style().standardPalette())
            app.setStyleSheet("")
            
            # Log the theme change
            logging.info("Applied light theme")
    
    def load_subjects(self):
        """Load subjects into the combo box."""
        self.subject_combo.clear()
        self.subject_combo.addItem("Select a subject...", None)
        
        subjects = self.db.get_subjects()
        for subject in subjects:
            self.subject_combo.addItem(subject['name'], subject['id'])
    
    def load_classes(self, subject_id):
        """Load classes for the selected subject."""
        self.classes_list.clear()
        
        if subject_id is None:
            return
        
        classes = self.db.get_classes(subject_id)
        
        # Sort classes by date (newest first)
        classes.sort(key=lambda c: c['date'], reverse=True)
        
        for cls in classes:
            display_text = f"{cls['name']} ({cls['date']})"
            item = self.classes_list.addItem(display_text)
            # Store the class ID as item data
            self.classes_list.item(self.classes_list.count() - 1).setData(Qt.UserRole, cls['id'])
    
    def closeEvent(self, event):
        """Handle the window close event."""
        # Stop any playing audio
        if hasattr(self, 'media_player'):
            self.media_player.stop()
        event.accept() 