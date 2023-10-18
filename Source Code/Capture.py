import cv2
import numpy as np
import os

# Define constants for image dimensions and paths
IMAGE_X, IMAGE_Y = 64, 64
TRAINING_SET_PATH = './Dataset/training_set/'
TEST_SET_PATH = './Dataset/test_set/'

def create_folder(folder_name):
    # Create the training and test set folders if they don't exist
    for folder_path in [TRAINING_SET_PATH, TEST_SET_PATH]:
        folder_path = os.path.join(folder_path, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

def capture_images(ges_name):
    create_folder(ges_name)

    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")

    img_counter = 0
    t_counter = 1
    training_set_image_name = 1
    test_set_image_name = 1
    listImage = [1, 2, 3, 4, 5]

    cv2.namedWindow("Trackbars")
    
    # Define the trackbars
    trackbars = {
    "L - H": (0, 179),  # Adjust the Hue range as needed
    "L - S": (20, 255),  # Increase the Saturation range
    "L - V": (50, 255)  # Increase the Value range
    "U - H": (179, 179),
    "U - S": (255, 255),
    "U - V": (255, 255)
}


    for name, (initial_min, initial_max) in trackbars.items():
        cv2.createTrackbar(name, "Trackbars", initial_min, initial_max, nothing)

    while test_set_image_name <= 250:
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)

        # Get the values of the trackbars
        trackbar_values = {name: cv2.getTrackbarPos(name, "Trackbars") for name in trackbars}

        img = cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), thickness=2, lineType=8, shift=0)
        
        lower_blue = np.array([trackbar_values["L - H"], trackbar_values["L - S"], trackbar_values["L - V"]])
        upper_blue = np.array([trackbar_values["U - H"], trackbar_values["U - S"], trackbar_values["U - V"]])

        imcrop = img[102:298, 427:623]
        hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        result = cv2.bitwise_and(imcrop, imcrop, mask=mask)

        # Display the image and mask
        cv2.putText(frame, str(img_counter), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
        cv2.imshow("test", frame)
        cv2.imshow("mask", mask)
        cv2.imshow("result", result)

        key = cv2.waitKey(1)
        if key == ord('c'):
            image_path = TRAINING_SET_PATH if t_counter <= 350 else TEST_SET_PATH
            img_name = f"{image_path}{ges_name}/{training_set_image_name}.png"
            save_img = cv2.resize(mask, (IMAGE_X, IMAGE_Y))
            cv2.imwrite(img_name, save_img)
            print(f"{img_name} written!")
            training_set_image_name += 1

            if t_counter > 350:
                test_set_image_name += 1
                if test_set_image_name > 250:
                    break

            t_counter += 1
            if t_counter == 401:
                t_counter = 1
            img_counter += 1

        elif key == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    ges_name = input("Enter gesture name: ")
    capture_images(ges_name)
