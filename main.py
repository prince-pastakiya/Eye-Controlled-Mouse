import cv2
import mediapipe as mp
import pyautogui

# Initialize the camera
cam = cv2.VideoCapture(0)

# Create a FaceMesh object to detect facial landmarks
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Get the screen dimensions
screen_w, screen_h = pyautogui.size()

while True:
    # Read a frame from the camera
    _, frame = cam.read()
    # Flip the frame horizontally (since the camera is mirrored)
    frame = cv2.flip(frame, 1)

    # Convert the frame from BGR to RGB (required for FaceMesh)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the frame using FaceMesh
    output = face_mesh.process(rgb_frame)
    # Get the facial landmarks
    landmark_points = output.multi_face_landmarks

    # Get the frame dimensions
    frame_h, frame_w, _ = frame.shape

    # If facial landmarks are detected
    if landmark_points:
        # Get the landmarks for the first face
        landmarks = landmark_points[0].landmark

        # Draw circles on the frame for the right eye landmarks (474-478)
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))

            # If this is the second landmark (the one on the right side of the eye)
            if id == 1:
                # Calculate the screen coordinates based on the eye position
                screen_x = screen_w / frame_w * x
                screen_y = screen_h / frame_h * y
                # Move the mouse cursor to the calculated position
                pyautogui.moveTo(screen_x, screen_y)

        # Get the landmarks for the left eye
        left = [landmarks[145], landmarks[159]]

        # Draw circles on the frame for the left eye landmarks
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))

        # If the left eye is closed (i.e., the distance between the two landmarks is small)
        if (left[0].y - left[1].y) < 0.004:
            # Click the mouse button
            pyautogui.click()
            # Wait for 1 second to avoid multiple clicks
            pyautogui.sleep(1)

    # Display the output frame
    cv2.imshow('Eye Controlled Mouse', frame)
    # Wait for a key press (required to update the window)
    cv2.waitKey(1)