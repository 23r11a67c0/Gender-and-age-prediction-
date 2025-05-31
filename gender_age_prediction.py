import cv2

# Load pre-trained models
age_model = cv2.dnn.readNetFromCaffe(
    'deploy_age.prototxt',
    'age_net.caffemodel'
)

gender_model = cv2.dnn.readNetFromCaffe(
    'deploy_gender.prototxt',
    'gender_net.caffemodel'
)

# Age and gender labels
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
            '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load image
image_path = 'test.jpg'  # Replace with your image
image = cv2.imread(image_path)
if image is None:
    raise Exception("Image not found. Please ensure 'test.jpg' exists in the same directory.")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Process each detected face
for (x, y, w, h) in faces:
    face_img = image[y:y+h, x:x+w].copy()
    blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), 
                                 (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

    # Predict gender
    gender_model.setInput(blob)
    gender_preds = gender_model.forward()
    gender = GENDER_LIST[gender_preds[0].argmax()]

    # Predict age
    age_model.setInput(blob)
    age_preds = age_model.forward()
    age = AGE_LIST[age_preds[0].argmax()]

    label = f"{gender}, {age}"
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

# Display result
cv2.imshow("Age and Gender Prediction", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
