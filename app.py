from flask import Flask, render_template, Response, request
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Load the Haar Cascade classifier
face_classifier = cv2.CascadeClassifier("C:/Users/aditi/Downloads/pythonProject (2)/pythonProject/haarcascade_frontalface_default.xml")

# Load the Keras model
classifier = load_model("C:/Users/aditi/Downloads/pythonProject (2)/pythonProject/model.h5")

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

def detect_emotion(frame):
    insights = {
        "Happy": "You seem happy! Keep up the positivity. Try gratitude journaling or celebrating small wins today.",
        "Neutral": "You seem calm. This is a good time to practice mindfulness or do a relaxing activity.",
        "Sad": "It seems you're feeling down. Consider reaching out to a loved one or taking a break for self-care.",
        "Angry": "You might be feeling frustrated. Try deep breathing or a calming activity to ease your mind.",
        "Fear": "You seem anxious. Ground yourself by focusing on your surroundings or trying a breathing exercise.",
        "Disgust": "You might feel discomfort or disapproval. Reflect on what’s causing these feelings.",
        "Surprise": "You seem surprised! Embrace curiosity and explore what's piqued your interest."
    }

    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]

        # Resize to model input size
        roi_gray_resized = cv2.resize(roi_gray, (48, 48))
        if np.sum([roi_gray_resized]) != 0:
            roi = roi_gray_resized.astype('float') / 255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)

            # Predictions
            predictions = classifier.predict(roi)[0]
            max_index = np.argmax(predictions)
            label = emotion_labels[max_index]
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Add Insights with dynamic wrapping
            if label in insights:
                insight_text = insights[label]
                max_line_width = 50  # Adjust for wrapping (in characters)
                wrapped_text = [
                    insight_text[i:i + max_line_width]
                    for i in range(0, len(insight_text), max_line_width)
                ]

                # Display insights below the detected face
                y0 = y + h + 20  # Start position below the face
                for i, line in enumerate(wrapped_text):
                    line_y = y0 + i * 20
                    cv2.putText(frame, line, (x, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        else:
            cv2.putText(frame, 'No Faces Detected', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame


def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = detect_emotion(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotion_detection')
def emotion_detection():
    return render_template('hii.html')

@app.route('/analysis')
def analysis():
    # Perform dataset analysis here
    # For example, count the occurrences of each emotion in your dataset
    # Replace the dummy data in the table with the actual analysis results
    return render_template('dataset.html')

@app.route('/dataset')
def dataset():
    return render_template('dataset.html')

@app.route('/us')
def us():
    return render_template('us.html')

@app.route('/upload_image')
def upload_image():
    return render_template('upload_image.html')

# Add this route for handling image upload and prediction
@app.route('/predict_image', methods=['POST'])
def predict_image():
    # Get the uploaded image file
    file = request.files['image']

    # Save the file to a temporary location
    file.save('temp.jpg')

    # Load the saved image using OpenCV
    img = cv2.imread('temp.jpg')

    # Perform emotion detection on the image
    result_img = detect_emotion(img)

    # Convert the result image to base64 format for rendering
    _, buffer = cv2.imencode('.jpg', result_img)
    result_img_base64 = base64.b64encode(buffer).decode('utf-8')

    # Render the result image in the template
    return render_template('result_image.html', result_img=result_img_base64)

@app.route('/booking')
def booking():
    return render_template('booking.html')

@app.route('/confirm_booking', methods=['POST'])
def confirm_booking():
    therapist = request.form['therapist']
    date = request.form['date']
    time = request.form['time']
    # You could store booking data in a database here
    return f"Booking confirmed with {therapist} on {date} at {time}"



if __name__ == '__main__':
    app.run(debug=True)
