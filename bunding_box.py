import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)

# Set the initial window size
cv2.namedWindow("Hand Tracking with Bounding Box", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Hand Tracking with Bounding Box", 1280, 860)  # Adjust the width and height as needed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
     # Process the frame with MediaPipe Hand
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:

                # Calculate bounding box coordinates
                x_min, y_min, x_max, y_max = 10000, 10000, 0, 0
                for landmark in landmarks.landmark:
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    if x < x_min:
                        x_min = x
                    if x > x_max:
                        x_max = x
                    if y < y_min:
                        y_min = y
                    if y > y_max:
                        y_max = y

                offset = 30

                x_min -= offset
                y_min -= offset
                x_max += offset
                y_max += offset

                 # Draw landmarks (optional)
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cv2.imshow("Hand Tracking with Bounding Box", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
