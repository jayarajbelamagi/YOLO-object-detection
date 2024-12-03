import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import speech_recognition as sr

# Set threshold for detection confidence
thres = 0.50  # Detection confidence threshold
is_detection_running = False  # Flag to check if detection is running

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # You can replace 'yolov8n.pt' with your trained YOLOv8 model

# Open the video capture with camera index 0
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Create the main GUI window
root = tk.Tk()
root.title("YOLOv8 Object Tracking")

# Create a label to display the video stream
video_label = ttk.Label(root)
video_label.pack(padx=10, pady=10)

# Speech recognizer initialization
recognizer = sr.Recognizer()

# Function to start YOLO detection
def start_detection():
    global is_detection_running
    is_detection_running = True
    update_frame()

# Function to stop YOLO detection
def stop_detection():
    global is_detection_running
    is_detection_running = False

# Start Detection button
start_button = ttk.Button(root, text="Start Detection", command=start_detection)
start_button.pack(side=tk.LEFT, padx=10)

# Stop Detection button
stop_button = ttk.Button(root, text="Stop Detection", command=stop_detection)
stop_button.pack(side=tk.LEFT, padx=10)

# Function to handle keyboard input
def on_key_press(event):
    if event.char == 'q':  # Close the window if 'q' is pressed
        root.quit()
        stop_detection()

root.bind('<KeyPress>', on_key_press)

# Update function for object detection and video stream
def update_frame():
    global is_detection_running

    success, frame = cap.read()

    # Check if the frame is captured successfully
    if not success:
        print("Error: Unable to capture video feed.")
        root.quit()
        return

    frame = cv2.resize(frame, (700, 500))  # Resize frame for display

    if is_detection_running:
        # Perform YOLOv8 inference with tracking
        results = model.track(frame, persist=True, stream=True)

        # Draw bounding boxes directly from YOLO output
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract coordinates and class index
                cls = int(box.cls[0])  # Class index
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates

                # Draw simple bounding box from YOLO output
                conf = round(float(box.conf[0]), 2)  # Convert Tensor to float and round
                class_label = model.names[cls]

                # Draw a rectangle (simple bounding box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Display the class label and confidence score
                cv2.putText(frame, f'{class_label} {conf}', (x1, max(35, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Convert the frame to RGB and update the tkinter label
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tk = ImageTk.PhotoImage(Image.fromarray(frame_rgb))

    video_label.img_tk = frame_tk  # Store reference to avoid garbage collection
    video_label.config(image=frame_tk)

    if is_detection_running:
        root.after(10, update_frame)  # Call update again after 10 milliseconds

# Process voice commands
def process_command(command):
    command = command.lower()
    if "start" in command:
        start_detection()
    elif "stop" in command:
        stop_detection()
    else:
        print("Command not recognized")

# Listen for commands using the microphone
def listen_commands():
    with sr.Microphone() as source:
        print("Listening for commands...")
        while True:
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                command = recognizer.recognize_google(audio)
                print(f"Received Command: {command}")
                process_command(command)
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")

# Start listening in a separate thread for voice commands
listener_thread = threading.Thread(target=listen_commands, daemon=True)
listener_thread.start()

# Start the GUI main loop
root.mainloop()

# Release the video capture and close OpenCV windows properly
cap.release()
cv2.destroyAllWindows()
