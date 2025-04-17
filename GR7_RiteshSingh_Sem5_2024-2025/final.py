from tkinter import *
from PIL import Image, ImageTk
import cv2
import os
from tkinter import filedialog

def main():
    win = Tk()
    app = LoginPage(win)
    win.mainloop()

class LoginPage:
    def __init__(self, win):
        self.win = win
        self.win.geometry("1550x950+0+0")
        self.win.title("Age and Gender Prediction")

        self.bg_image = Image.open(r"C:\\Users\\Ritesh Singh\\Desktop\\xyz\\pick.webp")  
        self.bg_image = self.bg_image.resize((1250, 650), Image.Resampling.LANCZOS)  
        self.bg_image = ImageTk.PhotoImage(self.bg_image)

        self.canvas = Canvas(self.win, width=1650, height=950)
        self.canvas.pack(fill=BOTH, expand=True)
        self.canvas.create_image(150, 100, image=self.bg_image, anchor=NW)

        self.title_label = Label(self.win, text="Age and Gender Prediction", font=('Arial', 35, 'bold'), bg="yellow", bd=8, relief=GROOVE)
        self.title_label.place(x=0, y=0, width=1530, height=80)

        self.add_buttons()

    def add_buttons(self):
        self.start_button = Button(self.win, text="Face Prediction", font=('Arial', 20, 'bold'), bg="purple", fg="white", command=self.face_prediction)
        self.start_button.place(x=450, y=200, width=250, height=250)

        self.voice_button = Button(self.win, text="Photo Prediction", font=('Arial', 20, 'bold'), bg="purple", fg="white", command=self.photo_prediction)
        self.voice_button.place(x=1000, y=200, width=250, height=250)

        self.exit_button = Button(self.win, text="Exit", font=('Arial', 20, 'bold'), bg="red", fg="white", command=self.quit_app)
        self.exit_button.place(x=730, y=680, width=250, height=50)

    def quit_app(self):
        """Release the video capture and close both Tkinter and OpenCV windows."""
        self.win.quit()  # Closes the Tkinter window
        cv2.destroyAllWindows()  # Closes all OpenCV windows

    def face_prediction(self):
        def faceBox(faceNet, frame):
            # Remove the print(frame) statement to avoid printing the entire matrix
            frameHeight = frame.shape[0]
            frameWidth = frame.shape[1]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB=False)
            faceNet.setInput(blob)
            detection = faceNet.forward()
            bboxs = []
            for i in range(detection.shape[2]):
                confidence = detection[0, 0, i, 2]
                if confidence > 0.7:
                    x1 = int(detection[0, 0, i, 3] * frameWidth)
                    y1 = int(detection[0, 0, i, 4] * frameHeight)
                    x2 = int(detection[0, 0, i, 5] * frameWidth)
                    y2 = int(detection[0, 0, i, 6] * frameHeight)
                    bboxs.append([x1, y1, x2, y2])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            return frame, bboxs

        # Define the base path
        base_path = "C:\\Users\\Sameer Singh\\Desktop\\age gender"
        faceProto = os.path.join(base_path, "opencv_face_detector.pbtxt")
        faceModel = os.path.join(base_path, "opencv_face_detector_uint8.pb")

        ageProto = os.path.join(base_path, "age_deploy.prototxt")
        ageModel = os.path.join(base_path, "age_net.caffemodel")

        genderProto = os.path.join(base_path, "gender_deploy.prototxt")
        genderModel = os.path.join(base_path, "gender_net.caffemodel")

        faceNet = cv2.dnn.readNet(faceModel, faceProto)
        ageNet = cv2.dnn.readNet(ageModel, ageProto)
        genderNet = cv2.dnn.readNet(genderModel, genderProto)

        MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 144.895847746)
        ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(20-25)', '(25-32)', '(32-38)', '(38-43)', '(48-53)', '(60-100)']
        genderList = ['Male', 'Female']

        video = cv2.VideoCapture(0)

        padding = 20

        while True:
            ret, frame = video.read()
            if not ret:
                print("Failed to capture image from the webcam")
                break

            frame, bboxs = faceBox(faceNet, frame)
            for bbox in bboxs:
                face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                             max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

                # Gender Prediction
                genderNet.setInput(blob)
                genderPred = genderNet.forward()
                gender = genderList[genderPred[0].argmax()]

                # Age Prediction
                ageNet.setInput(blob)
                agePred = ageNet.forward()
                age = ageList[agePred[0].argmax()]

                label = "{},{}".format(gender, age)
                cv2.rectangle(frame, (bbox[0], bbox[1] - 30), (bbox[2], bbox[1]), (0, 255, 0), -1)
                cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Age-Gender", frame)
            
            # If 'q' is pressed or the Tkinter window is closed, exit the loop
            k = cv2.waitKey(1)
            if k == ord('q') or not self.win.winfo_exists():
                print("Exiting loop")
                break

        video.release()
        cv2.destroyAllWindows()

    def photo_prediction(self):
        # Select an image file
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return  # Exit if no file is selected

        # Load the image using OpenCV
        image = cv2.imread(file_path)

        def faceBox(faceNet, frame):
            frameHeight = frame.shape[0]
            frameWidth = frame.shape[1]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB=False)
            faceNet.setInput(blob)
            detection = faceNet.forward()
            bboxs = []
            for i in range(detection.shape[2]):
                confidence = detection[0, 0, i, 2]
                if confidence > 0.7:
                    x1 = int(detection[0, 0, i, 3] * frameWidth)
                    y1 = int(detection[0, 0, i, 4] * frameHeight)
                    x2 = int(detection[0, 0, i, 5] * frameWidth)
                    y2 = int(detection[0, 0, i, 6] * frameHeight)
                    bboxs.append([x1, y1, x2, y2])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            return frame, bboxs

    
        base_path = "C:\\Users\\Sameer Singh\\Desktop\\age gender"

        faceProto = os.path.join(base_path, "opencv_face_detector.pbtxt")
        faceModel = os.path.join(base_path, "opencv_face_detector_uint8.pb")

        ageProto = os.path.join(base_path, "age_deploy.prototxt")
        ageModel = os.path.join(base_path, "age_net.caffemodel")

        genderProto = os.path.join(base_path, "gender_deploy.prototxt")
        genderModel = os.path.join(base_path, "gender_net.caffemodel")

        faceNet = cv2.dnn.readNet(faceModel, faceProto)
        ageNet = cv2.dnn.readNet(ageModel, ageProto)
        genderNet = cv2.dnn.readNet(genderModel, genderProto)

        MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 144.895847746)
        ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(20-25)', '(25-32)', '(32-38)', '(38-43)', '(48-53)', '(60-100)']
        genderList = ['Male', 'Female']

        padding = 20

        # Perform face detection and age/gender prediction on the image
        frame, bboxs = faceBox(faceNet, image)
        for bbox in bboxs:
            face = image[max(0, bbox[1] - padding):min(bbox[3] + padding, image.shape[0] - 1),
                         max(0, bbox[0] - padding):min(bbox[2] + padding, image.shape[1] - 1)]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            # Gender Prediction
            genderNet.setInput(blob)
            genderPred = genderNet.forward()
            gender = genderList[genderPred[0].argmax()]

            # Age Prediction
            ageNet.setInput(blob)
            agePred = ageNet.forward()
            age = ageList[agePred[0].argmax()]

            label = "{},{}".format(gender, age)
            cv2.rectangle(image, (bbox[0], bbox[1] - 30), (bbox[2], bbox[1]), (0, 255, 0), -1)
            cv2.putText(image, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the result image
        cv2.imshow("Age-Gender Photo Prediction", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
