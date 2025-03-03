import json
import os
import datetime
from pathlib import Path

class Database:
    """A simple JSON-based database to store and retrieve application data."""
    
    def __init__(self, data_dir='app/data'):
        """Initialize the database with the data directory."""
        self.data_dir = Path(data_dir)
        self.subjects_file = self.data_dir / 'subjects.json'
        self.classes_file = self.data_dir / 'classes.json'
        self.settings_file = self.data_dir / 'settings.json'
        
        # Ensure the data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize data files if they don't exist
        if not self.subjects_file.exists():
            with open(self.subjects_file, 'w') as f:
                json.dump([], f)
                
        if not self.classes_file.exists():
            with open(self.classes_file, 'w') as f:
                json.dump([], f)
                
        # Initialize settings file with defaults if it doesn't exist
        if not self.settings_file.exists():
            default_settings = {
                'theme': 'dark',  # Default to dark theme
                'ollama_model': 'mistral:latest',
                'ollama_host': 'http://localhost:11434',
                'whisper_model_size': 'base'
            }
            with open(self.settings_file, 'w') as f:
                json.dump(default_settings, f, indent=4)
    
    def get_subjects(self):
        """Get the list of all subjects."""
        with open(self.subjects_file, 'r') as f:
            return json.load(f)
    
    def add_subject(self, name):
        """Add a new subject to the database."""
        subjects = self.get_subjects()
        
        # Check if subject already exists
        for subject in subjects:
            if subject['name'] == name:
                return False  # Subject already exists
        
        # Create a new subject with a unique ID by finding the maximum ID and adding 1
        subject_id = 1
        if subjects:
            subject_id = max(subj["id"] for subj in subjects) + 1
        
        new_subject = {
            'id': subject_id,
            'name': name,
            'created_at': datetime.datetime.now().isoformat()
        }
        
        subjects.append(new_subject)
        
        # Save the updated subjects list
        with open(self.subjects_file, 'w') as f:
            json.dump(subjects, f, indent=4)
            
        return subject_id
    
    def delete_subject(self, subject_id):
        """Delete a subject and all its classes."""
        subjects = self.get_subjects()
        classes = self.get_classes()
        
        # Filter out the subject to delete
        updated_subjects = [s for s in subjects if s['id'] != subject_id]
        
        # Filter out classes associated with the subject
        updated_classes = [c for c in classes if c['subject_id'] != subject_id]
        
        # Save the updated data
        with open(self.subjects_file, 'w') as f:
            json.dump(updated_subjects, f, indent=4)
            
        with open(self.classes_file, 'w') as f:
            json.dump(updated_classes, f, indent=4)
        
        return len(subjects) - len(updated_subjects)  # Number of deleted subjects
    
    def get_classes(self, subject_id=None):
        """Get classes, optionally filtered by subject."""
        with open(self.classes_file, 'r') as f:
            classes = json.load(f)
        
        if subject_id is not None:
            return [c for c in classes if c['subject_id'] == subject_id]
        else:
            return classes
    
    def get_class(self, class_id):
        """Get a single class by ID."""
        with open(self.classes_file, 'r') as f:
            classes = json.load(f)
        
        # Find the class with the matching ID
        for cls in classes:
            if cls['id'] == class_id:
                return cls
                
        return None  # Class not found
    
    def add_class(self, subject_id, name, date, chapter=None):
        """Add a new class to the database."""
        classes = self.get_classes()
        
        # Generate a truly unique ID by finding the maximum ID and adding 1
        class_id = 1
        if classes:
            class_id = max(cls["id"] for cls in classes) + 1
        
        new_class = {
            'id': class_id,
            'subject_id': subject_id,
            'name': name,
            'date': date,
            'chapter': chapter,
            'created_at': datetime.datetime.now().isoformat(),
            'audio_path': None,
            'transcript': None,
            'summary': None,
            'language': None,  # Store the detected language of the transcript
            'flashcards': []
        }
        
        classes.append(new_class)
        
        # Save the updated classes list
        with open(self.classes_file, 'w') as f:
            json.dump(classes, f, indent=4)
            
        return class_id
    
    def update_class(self, class_id, **updates):
        """Update class information."""
        classes = self.get_classes()
        
        for i, cls in enumerate(classes):
            if cls['id'] == class_id:
                # Update the class with the provided values
                for key, value in updates.items():
                    if key in cls:
                        cls[key] = value
                
                # Save the updated classes list
                with open(self.classes_file, 'w') as f:
                    json.dump(classes, f, indent=4)
                
                return True
                
        return False  # Class not found
    
    def delete_class(self, class_id):
        """Delete a class."""
        classes = self.get_classes()
        
        # Filter out the class to delete
        updated_classes = [c for c in classes if c['id'] != class_id]
        
        # Save the updated classes list
        with open(self.classes_file, 'w') as f:
            json.dump(updated_classes, f, indent=4)
        
        return len(classes) - len(updated_classes)  # Number of deleted classes
    
    def add_flashcard(self, class_id, question, answer):
        """Add a flashcard to a class."""
        classes = self.get_classes()
        
        for i, cls in enumerate(classes):
            if cls['id'] == class_id:
                # Ensure the flashcards list exists
                if 'flashcards' not in cls:
                    cls['flashcards'] = []
                
                # Create a new flashcard
                new_flashcard = {
                    'id': len(cls['flashcards']) + 1,
                    'question': question,
                    'answer': answer,
                    'created_at': datetime.datetime.now().isoformat()
                }
                
                cls['flashcards'].append(new_flashcard)
                
                # Save the updated classes list
                with open(self.classes_file, 'w') as f:
                    json.dump(classes, f, indent=4)
                
                return new_flashcard['id']
        
        return None  # Class not found
    
    def get_settings(self):
        """Get the application settings."""
        try:
            with open(self.settings_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # If the file is missing or corrupted, create default settings
            default_settings = {
                'theme': 'dark',  # Default to dark theme
                'ollama_model': 'mistral:latest',
                'ollama_host': 'http://localhost:11434',
                'whisper_model_size': 'base'
            }
            self.save_settings(default_settings)
            return default_settings
    
    def save_settings(self, settings):
        """Save the application settings."""
        with open(self.settings_file, 'w') as f:
            json.dump(settings, f, indent=4)
        return True
    
    def update_setting(self, key, value):
        """Update a single setting."""
        settings = self.get_settings()
        settings[key] = value
        return self.save_settings(settings) 