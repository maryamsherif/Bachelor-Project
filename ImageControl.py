import os
import time
import tensorflow as tf
import numpy as np
import mediapipe as mp
import cv2


# Image Control Variables
pictures = os.listdir("images")
img1 = cv2.imread(os.path.join("images", pictures[1]), cv2.IMREAD_UNCHANGED)

# Resize image
scale_factor = 1.0
# Calculate the maximum allowed scale factor
img1 = cv2.resize(img1, (250, 250), fx=scale_factor, fy=scale_factor)
img1 = img1[:, :, :3]
startDist = None
angle = 0
# Define initial position of image
x_pos = 100
y_pos = 100
display_width, display_height = 640, 480

# Load the model
model = tf.keras.models.load_model("cnn_model.h5")

# Mediapipe model
mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


# Function to detect the hand
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
    )  # Draw right hand connections


def draw_styled_landmarks(image, results):
    # Draw right hand connections
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
    )


def extract_keypoints(results):
    rh = (
        np.array(
            [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
        ).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )
    return np.concatenate([rh])


actions = [
    "Move Up",
    "Move Down",
    "Move Left",
    "Move Right",
    "Rotate Right",
    "Rotate Left",
    "Zoom In",
    "Zoom Out",
]
threshold = 0.7

colors = [
    (245, 117, 16),
    (117, 245, 16),
    (16, 0, 245),
    (255, 0, 0),
    (160, 117, 245),
    (16, 170, 245),
    (250, 117, 245),
    (16, 117, 0),
]

# Loading the model
model.load_weights("cnn_model.h5")

sequence = []
sentence = []
threshold = 0.7

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Set mediapipe model
with mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as holistic:
    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()
        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        # Draw landmarks
        draw_styled_landmarks(image, results)

        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        res = np.zeros(len(actions))  # define a default value for res

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]

            if res[np.argmax(res)] > threshold:
                if actions[np.argmax(res)] == "Move Up":
                    y_pos -= 10
                elif actions[np.argmax(res)] == "Move Down":
                    y_pos += 10
                elif actions[np.argmax(res)] == "Move Left":
                    x_pos -= 10
                elif actions[np.argmax(res)] == "Move Right":
                    x_pos += 10
                elif actions[np.argmax(res)] == "Rotate Right":
                    # angle += 10
                    # img1 = cvzone.rotateImage(img1, angle)
                    print("beb")
                elif actions[np.argmax(res)] == "Rotate Left":
                    # angle -= 10
                    # img1 = cvzone.rotateImage(img1, angle)
                    print("beb")
                elif actions[np.argmax(res)] == "Zoom In":
                    scale_factor += 0.1
                   
                elif actions[np.argmax(res)] == "Zoom Out":
                    scale_factor -= 0.1
                    if scale_factor < 0.1:
                        scale_factor = 0.1
                time.sleep(0.5)

        word = ""

        if np.argmax(res) < len(actions):
            word = actions[np.argmax(res)]

        # Get the current size of the window and img1
        win_height, win_width = image.shape[:2]
        img_height, img_width = img1.shape[:2]

        # Check if the image is going out of bounds in the x-direction (Move left or right)
        if x_pos < 0:
            x_pos = 0
        elif x_pos + img_width > win_width:
            x_pos = win_width - img_width

        # Check if the image is going out of bounds in the y-direction (Move up or down)
        if y_pos < 0:
            y_pos = 0
        elif y_pos + img_height > win_height:
            y_pos = win_height - img_height

        # Handle zooming in and out
        max_scale_factor = min(win_width / img_width, win_height / img_height)

        if scale_factor > max_scale_factor:
            scale_factor = min(scale_factor, max_scale_factor)
        if scale_factor < 0.1:
            scale_factor = 0.1

        if img_height > win_height or img_width > win_width:
            # If so, resize the image to fit the frame
            img1 = cv2.resize(img1, (win_width, win_height))
        else:
            img1 = cv2.resize(img1, (0, 0), fx=scale_factor, fy=scale_factor)

        img_height, img_width = img1.shape[:2]
        print("image: ", image.shape)
        print("img1: ", img1.shape)

        # Handle rotating the image (Rotate left or right)

        # Add the image to the frame
        image[y_pos : y_pos + img_height, x_pos : x_pos + img_width] = img1

        # Draw header
        cv2.rectangle(image, (0, 0), (1280, 40), (255, 255, 255), -1)
        cv2.putText(
            image,
            word,
            (3, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

        # Show to screen
        cv2.imshow("OpenCV Feed", image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
