import cv2 as cv
import os
import json
import uuid
import face_recognition
import dlib
import matplotlib.pyplot as plt

# Making necessary directories

face_dir = "Faces"
user_dir = "UserData"
log_dir = "Logins"
os.makedirs(face_dir, exist_ok=True)
os.makedirs(user_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

print("SIGN UP")

# Asking user to input their information

while True:
    name = input("Enter your name: ")
    if name:
        break
    print("Invalid name. Please try again.")

while True:
    age = input("Enter your age: ")
    if age.isdigit():
        break
    print("Age must be a number. Please try again.")

while True:
    id = input("Enter your ID: ")
    if id.isdigit():
        break
    print("ID must be a number. Please try again.")



# Saving user's image

cap = cv.VideoCapture(0)  
if not cap.isOpened():
    print("Cannot open camera.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Cannot receive frame.")
        break

    cv.imshow("Press 's' to save the image and 'q' to quit.", frame)

    key = cv.waitKey(1) & 0xFF
    if key == ord('s') or key == ord('S'):
        img_name = os.path.join(face_dir, f"{name}_{age}_{uuid.uuid4().hex}.jpg")
        cv.imwrite(img_name, frame)
        print("Image saved as:", img_name)

    if key == ord('q') or key == ord('Q'):
        break

cap.release()
cv.destroyAllWindows()

# Detecting face in the image

image = cv.imread(img_name)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

detector = dlib.get_frontal_face_detector()
faces = detector(gray)

for face in faces:
    x, y, w, h = (face.left(), face.top(), face.width(), face.height())
    cv.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)
    cv.putText(image, name, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

load = face_recognition.load_image_file(img_name)
encoding = face_recognition.face_encodings(load, num_jitters=50, model = 'large')[0]

# Storing user's information in a file

info = {"Name": name, "Age": age, "ID": id, "Encoding": encoding[0]}

user_info_path = os.path.join(user_dir, f"{name}_{id}.json")
with open(user_info_path, 'w') as file:
    json.dump(info, file, indent=4)

l = input("Enter '1' to Login and '0' to exit: ")

if (l == 1):
    print("LOGIN")
    cam = cv.VideoCapture(0)  
    if not cam.isOpened():
        print("Cannot open camera.")
        exit()

    while True:
        ret, log_frame = cam.read()

        if not ret:
            print("Cannot receive frame.")
            break
        cv.imshow("Press 'e' to enter and 'q' to quit.", frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord('e') or key == ord('E'):
            log_name = os.path.join(log_dir, f"{name}.jpg")
            cv.imwrite(log_name, log_frame)

        if key == ord('q') or key == ord('Q'):
            break

    

    log_image = cv.imread(log_name)
    log_gray = cv.cvtColor(log_image, cv.COLOR_BGR2GRAY)

    log_detector = dlib.get_frontal_face_detector()
    logins = log_detector(log_gray)

    for face in logins:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        cv.rectangle(log_image, (x,y), (x+w, y+h), (0,0,255), 2)

    try:
        live_face_encoding = face_recognition.face_encodings(log_image, num_jitters = 23, model = 'large')[0]
        result = face_recognition.compare_faces(encoding, live_face_encoding)

        if result:
            log_img = cv.cvtColor(log_frame, cv.COLOR_BGR2RGB)
            log_img = cv.putText(log_img, name, (30, 55), cv.FONT_HERSHEY_SIMPLEX, 1,
                    (255,0,0), 2, cv.LINE_AA)
            print(f'{name} Enter....')
            plt.imshow(log_img)
            plt.show()

    except:
        log_img = cv.putText(log_frame, f'Not {name}', (30,55), cv.FONT_HERSHEY_SIMPLEX, 1, 
                (255,0,0), 2, cv.LINE_AA)
        
        cv.imshow("Login Frame", log_img)
        
    cam.release()
    cv.destroyAllWindows()

if(l == 0):
    print("Exiting program.....")
