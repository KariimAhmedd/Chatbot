from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QTextEdit, QPushButton, QLineEdit,
                             QLabel, QFileDialog, QScrollArea)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtGui import QFont, QPixmap
from PIL import Image, ImageQt
from datetime import datetime
import json
import torch
import threading
import queue
import os
import traceback
import sys

class MessageThread(QThread):
    message_processed = pyqtSignal(str, str)
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.queue = queue.Queue()
        self.running = True
    
    def run(self):
        while self.running:
            try:
                message = self.queue.get()
                if message is None:
                    break
                    
                # Get model response directly from GPT-2
                response = self.model.generate_response(message)
                
                self.message_processed.emit("KimozChatBot", response)
                
            except Exception as e:
                self.message_processed.emit("System", f"Error processing message: {str(e)}")
    
    def stop(self):
        self.running = False
        self.queue.put(None)
        self.wait()

class ModernChatGUI(QMainWindow):
    def __init__(self, model, image_processor):
        super().__init__()
        self.model = model
        self.image_processor = image_processor
        
        # Set window properties
        self.setWindowTitle("KimozChatBot AI Assistant")
        self.setMinimumSize(800, 600)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont('Arial', 12))
        layout.addWidget(self.chat_display)
        
        # Control panel
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        
        # Image button
        self.img_btn = QPushButton("Upload Image")
        self.img_btn.clicked.connect(self.upload_image)
        self.img_btn.setMinimumWidth(100)
        control_layout.addWidget(self.img_btn)
        
        # Message entry
        self.message_entry = QLineEdit()
        self.message_entry.setFont(QFont('Arial', 12))
        self.message_entry.returnPressed.connect(self.send_message)
        control_layout.addWidget(self.message_entry)
        
        # Send button
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send_message)
        self.send_btn.setMinimumWidth(100)
        control_layout.addWidget(self.send_btn)
        
        layout.addWidget(control_panel)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
        # Start message processing thread
        self.message_thread = MessageThread(model)
        self.message_thread.message_processed.connect(self.on_message_processed)
        self.message_thread.start()
    
    def display_message(self, sender, message, is_image=False, image_path=None):
        try:
            # Add timestamp and sender
            timestamp = datetime.now().strftime("%H:%M")
            if sender == "KimozChatBot":
                prefix = f"[{timestamp}] 🤖 Bot: "
            elif sender == "System":
                prefix = f"[{timestamp}] ⚙️ System: "
            else:
                prefix = f"[{timestamp}] 👤 You: "
            
            # Add text message
            self.chat_display.append(prefix + message)
            
            # Handle image if present
            if is_image and image_path:
                try:
                    pixmap = QPixmap(image_path)
                    if not pixmap.isNull():
                        # Scale image if needed
                        if pixmap.width() > 200:
                            pixmap = pixmap.scaledToWidth(200, Qt.SmoothTransformation)
                        self.chat_display.append("")
                        # Use HTML img tag directly with file path
                        self.chat_display.append(f'<img src="{image_path}" width="200"/>')
                        self.chat_display.append("")
                except Exception as e:
                    self.chat_display.append(f"[Error displaying image: {str(e)}]\n")
            
            # Scroll to bottom
            self.chat_display.verticalScrollBar().setValue(
                self.chat_display.verticalScrollBar().maximum()
            )
            
        except Exception as e:
            print(f"Error displaying message: {str(e)}")
    
    def send_message(self):
        message = self.message_entry.text().strip()
        if message:
            self.message_entry.clear()
            self.display_message("You", message)
            self.message_thread.queue.put(message)
            self.statusBar().showMessage("Processing...")
    
    def on_message_processed(self, sender, message):
        self.display_message(sender, message)
        self.statusBar().showMessage("Ready")
    
    def upload_image(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Image",
                "",
                "Image Files (*.png *.jpg *.jpeg *.gif *.bmp)"
            )
            if file_path:
                self.display_message("You", "Uploaded an image", is_image=True, image_path=file_path)
                self.process_image(file_path)
        except Exception as e:
            self.display_message("System", f"Error uploading image: {str(e)}")
    
    def process_image(self, image_path):
        try:
            image = Image.open(image_path)
            description = self.image_processor.process_image(image)
            
            # Generate a contextual response using GPT-2 with the image description
            context = f"I'm looking at an image. {description}"
            response = self.model.generate_response(context)
            
            self.display_message("KimozChatBot", response)
            
        except Exception as e:
            self.display_message("System", f"Error processing image: {str(e)}")
    
    def closeEvent(self, event):
        self.message_thread.stop()
        super().closeEvent(event)

def start_gui(model, image_processor):
    try:
        app = QApplication(sys.argv)
        
        # Set application-wide style
        app.setStyle('Fusion')
        
        # Create and show the main window
        window = ModernChatGUI(model, image_processor)
        window.show()
        
        # Start the event loop
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Error starting GUI: {str(e)}")
        traceback.print_exc()
        raise 