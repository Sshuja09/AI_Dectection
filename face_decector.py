import cv2


def webcam_face_detector():
    # Pre-trained face classifier data / Download from Github
    face_trainer_data = cv2.CascadeClassifier("frontalFace_data.xml")

    # Capture the video from the default cam
    webcam = cv2.VideoCapture(0)

    while True:
        # Read the frame from the webcam video and if it was read successfully or not
        (successful_frame, frame) = webcam.read()

        # If the frame was successfully read
        if successful_frame:
            # Convert the frame color to gray scale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            break

        # Get the face coordinates of the face in that frame with the training data
        face_coordinates = face_trainer_data.detectMultiScale(gray_frame)

        # Go through the top left of the face/s coordinates and draw a rectangle depending on the width and height
        for (x, y, w, h) in face_coordinates:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 6)

        # Show the image with the rectangle 
        cv2.imshow("SELAM AI", frame)

        # Wait for a key press or for the giving seconds / Returns the ascii num of the key pressed
        key = cv2.waitKey(1)

        # If the key press one of the mentioned ones break
        if key==113 or key==81:
            break


    webcam.release()


def pic_face_detector():
    # Pre-trained face classifier data / Download from Github
    face_trainer_data = cv2.CascadeClassifier("frontalFace_data.xml")
    
    # Choose a png file to detect face in
    img = cv2.imread("multiFace.png")

    # Change the img to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Train the algorithm to detect faces
    face_coordinates = face_trainer_data.detectMultiScale(gray_img)

    # Draw the rectangles around the face coordinates
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)


    # Shows the chosen image file and wait for a key press
    cv2.imshow("HELLO ai", img)
    cv2.waitKey()
 

def car_detector():
    car_detector_data = cv2.CascadeClassifier("cars.xml")

    img = cv2.imread("hway.jpg")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    """
    # For video

    vid = cv2.VideoCapture(FILE_NAME.mp4)

    while True:
        # Read the frame from the webcam video and if it was read successfully or not
        (successful_frame, frame) = vid.read()

        # If the frame was successfully read
        if successful_frame:
            # Convert the frame color to gray scale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            break

        # The rest of the code will be inside the while loop
    """

    car_coordinates = car_detector_data.detectMultiScale(gray_img)

    for (x, y, w, h) in car_coordinates:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Car detector", img)
    cv2.waitKey()

# pic_face_detector()
# car_detector()
webcam_face_detector()
print("Code completed")
