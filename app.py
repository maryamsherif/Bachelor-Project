import tkinter as tk
from tkinter import *
import os
import subprocess
import threading
import sys
from PIL import Image, ImageTk

# Global variables to store the thread running the ImageControl.py script and the process ID
image_control_thread = None
image_control_process = None


def start_button_clicked():
    # Call your Python code here
    # This function will be executed when the Start button is clicked
    print("Start button clicked")
    start_image_control()


def start_image_control():
    global image_control_thread, image_control_process
    if not image_control_thread or not image_control_thread.is_alive():
        # Create a new thread to run the ImageControl.py script
        image_control_thread = threading.Thread(target=run_image_control)
        image_control_thread.start()
    else:
        print("ImageControl.py is already running.")


def run_image_control():
    # Call the ImageControl.py script
    global image_control_process
    python_interpreter = sys.executable
    image_control_process = subprocess.Popen([python_interpreter, "ImageControl.py"])

    # Update the label text to indicate the user should wait
    wait_label.config(text="Please wait until Camera starts")

    image_control_process.wait()  # Wait for the process to finish

    # Update the label text to indicate the process has finished
    wait_label.config(text="Video has finished.")


def upload_button_clicked():
    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename()

    # Move the selected image to the images directory in your app
    image_directory = "images"
    os.makedirs(image_directory, exist_ok=True)
    new_file_path = os.path.join(image_directory, os.path.basename(file_path))
    os.rename(file_path, new_file_path)

    print("Uploaded image:", new_file_path)


def stop_button_clicked():
    # Terminate the ImageControl.py script using the stored process ID
    global image_control_process
    if image_control_process:
        image_control_process.terminate()
        print("Video recording stopped")


# Create the main application window
window = tk.Tk()
window.title("Image Control App")


# Load the background image
bg_image = Image.open("background_image.jpg")

# Resize the image to fit the window
window_width = window.winfo_screenwidth()
window_height = window.winfo_screenheight()
bg_image = bg_image.resize((window_width, window_height), Image.ANTIALIAS)

# Convert the image to Tkinter format
bg_photo = ImageTk.PhotoImage(bg_image)

# Create a label with the background image
bg_label = Label(window, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# Create a frame for the content
content_frame = tk.Frame(window)
content_frame.pack(pady=20)

# Maximize the window
window.state('zoomed')

# Create the welcome label
welcome_label = tk.Label(window, text="Welcome to Image Control App", font=("Arial", 18))
welcome_label.pack(pady=10)

# Create the app description label
description_label = tk.Label(window, text="This app allows you to control your design using your right hand movements.",
                             font=("Arial", 14))
description_label.pack(pady=10)

# Create the app description label
description1_label = tk.Label(window, text="You can move the image, zoom it and rotate it.", font=("Arial", 14))
description1_label.pack(pady=10)

# Create the instructions label
instructions_label = tk.Label(window, text="To start the image capturing, click the 'Start' button.",
                              font=("Arial", 12))
instructions_label.pack(pady=10)

# Create the label for the wait message
wait_label = tk.Label(window, text="", font=("Arial", 12))
wait_label.pack(pady=10)

# Create the Start button
start_button = tk.Button(window, text="Start", font=("Arial", 14), command=start_button_clicked)
start_button.pack(pady=10)

# Create the Stop button
stop_button = tk.Button(window, text="Stop", font=("Arial", 14), command=stop_button_clicked)
stop_button.pack(pady=10)

# Create the Upload Picture button
upload_button = tk.Button(window, text="Upload Picture", font=("Arial", 14), command=upload_button_clicked)
upload_button.pack(pady=10)


# Start the main event loop
window.mainloop()
