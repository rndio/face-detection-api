from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

# Load DNN Face Detector (lebih akurat)
modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

@app.route("/detect_face", methods=["POST"])
def detect_face():
    if "image" not in request.files:
        return jsonify({"error": "image file is required"}), 400

    file = request.files["image"]
    img_bytes = file.read()

    # decoding image
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "invalid image"}), 400

    (h, w) = img.shape[:2]

    # preprocessing input for DNN
    blob = cv2.dnn.blobFromImage(
        img, 
        1.0, 
        (300, 300), 
        (104.0, 177.0, 123.0),
        swapRB=False, 
        crop=False
    )

    net.setInput(blob)
    detections = net.forward()

    face_count = 0

    # iterate every detection
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # threshold untuk mengurangi false positive
        if confidence > 0.6:
            face_count += 1

    return jsonify({
        "has_face": face_count > 0,
        "face_count": face_count
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
