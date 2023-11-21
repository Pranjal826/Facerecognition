import cv2
import face_recognition

def find_face_and_encoding(img):
    # Convert BGR to RGB (face_recognition uses RGB)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Find all face locations and face encodings in the image
    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

    return face_locations, face_encodings

def compare_faces(suspect_encoding, unknown_encoding):
    # Compare the face encodings and return True if they match
    results = face_recognition.compare_faces([suspect_encoding], unknown_encoding)
    return results[0]

# Load the suspect image
suspect_image = face_recognition.load_image_file("Myimg.jpeg")
suspect_encoding = face_recognition.face_encodings(suspect_image)[0]

# Open a video capture
cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()

    # Find faces and face encodings in the current frame
    face_locations, face_encodings = find_face_and_encoding(img)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the current face encoding with the suspect encoding
        match = compare_faces(suspect_encoding, face_encoding)

        # Draw a rectangle around the face and label the result
        color = (0, 255, 0) if match else (0, 0, 255)
        cv2.rectangle(img, (left, top), (right, bottom), color, 2)
        label = "Match" if match else "No Match"
        cv2.putText(img, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('Face Recognition', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()