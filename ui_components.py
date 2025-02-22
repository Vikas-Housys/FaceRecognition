# ui_components.py
import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import threading
import time

class FaceRecognitionUI:
    def __init__(self, master, face_processor):
        self.master = master
        self.face_processor = face_processor
        self.capture = None
        self.is_capturing = False
        self.registration_mode = False
        
        self.setup_ui()

    def setup_ui(self):
        # Create main frame
        self.main_frame = ctk.CTkFrame(self.master)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Video display
        self.video_label = ctk.CTkLabel(self.main_frame, text="")
        self.video_label.pack(pady=20)

        # Buttons frame
        self.button_frame = ctk.CTkFrame(self.main_frame)
        self.button_frame.pack(pady=20)

        # Register button
        self.register_btn = ctk.CTkButton(
            self.button_frame,
            text="Register Face",
            command=self.start_registration
        )
        self.register_btn.pack(side="left", padx=10)

        # Recognize button
        self.recognize_btn = ctk.CTkButton(
            self.button_frame,
            text="Recognize Face",
            command=self.toggle_recognition
        )
        self.recognize_btn.pack(side="left", padx=10)

        # Exit button
        self.exit_btn = ctk.CTkButton(
            self.button_frame,
            text="Exit",
            command=self.exit_program
        )
        self.exit_btn.pack(side="left", padx=10)

        # Capture button (hidden by default)
        self.capture_btn = ctk.CTkButton(
            self.main_frame,
            text="Capture Photo",
            command=self.capture_photo
        )
        
        # Status label
        self.status_label = ctk.CTkLabel(self.main_frame, text="")
        self.status_label.pack(pady=10)

    def start_registration(self):
        """Start face registration process."""
        # Create registration dialog
        dialog = ctk.CTkInputDialog(text="Enter person name:", title="Register Face")
        name = dialog.get_input()
        
        if name:
            self.capture = cv2.VideoCapture(0)
            self.remaining_photos = 5
            self.current_name = name
            self.registration_mode = True
            self.status_label.configure(text=f"Taking 5 photos. Click 'Capture Photo' button.")
            
            # Show capture button
            self.capture_btn.pack(pady=10)
            
            # Start video feed
            self.update_video_feed()

    def update_video_feed(self):
        """Update video feed on UI."""
        if self.capture is None:
            return
            
        ret, frame = self.capture.read()
        if ret:
            # Convert frame for display
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            img = img.resize((640, 480))  # Resize for consistent display
            photo = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 480))
            self.video_label.configure(image=photo)
            
            if self.registration_mode or self.is_capturing:
                self.master.after(10, self.update_video_feed)

    def capture_photo(self):
        """Capture photo when button is clicked."""
        if self.capture is None:
            return
            
        ret, frame = self.capture.read()
        if ret:
            success, message = self.face_processor.register_face(self.current_name, frame)
            if success:
                self.remaining_photos -= 1
                self.status_label.configure(
                    text=f"Photo captured! {self.remaining_photos} remaining."
                )
                
                if self.remaining_photos == 0:
                    self.registration_mode = False
                    self.capture.release()
                    self.capture = None
                    self.capture_btn.pack_forget()  # Hide capture button
                    self.status_label.configure(text="Registration complete!")
                    self.video_label.configure(image=None, text="")
            else:
                self.status_label.configure(text=f"Failed to capture: {message}")

    
    def toggle_recognition(self):
        """Toggle real-time face recognition."""
        if not self.is_capturing:
            self.capture = cv2.VideoCapture(0)
            self.is_capturing = True
            self.recognize_btn.configure(text="Stop Recognition")
            self.recognition_loop()
        else:
            self.is_capturing = False
            self.recognize_btn.configure(text="Recognize Face")
            if self.capture:
                self.capture.release()
                self.capture = None
                self.video_label.configure(image=None, text="")

    def recognition_loop(self):
        """Main recognition loop."""
        if not self.is_capturing:
            return
            
        ret, frame = self.capture.read()
        if ret:
            # Process frame
            name, message = self.face_processor.recognize_face(frame)
            
            # Draw feedback on frame
            status_text = f"{name if name else message}"
            cv2.putText(
                frame,
                status_text,
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0) if name else (0, 0, 255),
                2
            )
            
            # Convert frame for display
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            img = img.resize((640, 480))  # Resize for consistent display
            photo = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 480))
            self.video_label.configure(image=photo)
            
            self.master.after(10, self.recognition_loop)

    def exit_program(self):
        """Clean up and exit the program."""
        if self.capture:
            self.capture.release()
        self.master.quit()


