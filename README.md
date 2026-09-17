# Detect Face API

A lightweight face detection API built with Flask and OpenCV's DNN face detector. It accepts an image upload and reports whether a face was detected and how many faces were found.

## Features

- REST API for face detection
- OpenCV DNN SSD face detection model
- Docker and Docker Compose support
- JSON responses with face presence and count

## Requirements

- Python 3.12 or newer
- Docker and Docker Compose (optional)

The required model files are included in the `models/` directory.

## Run with Docker Compose

```bash
docker compose up --build
```

The API will be available at:

```text
http://localhost:5000
```

To stop the service:

```bash
docker compose down
```

## Run locally

Create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start the API:

```bash
python app.py
```

The server listens on `0.0.0.0:5000`.

## API

### `POST /detect_face`

Upload an image using the `image` form field.

```bash
curl -X POST \
  -F "image=@path/to/image.jpg" \
  http://localhost:5000/detect_face
```

Successful response:

```json
{
  "has_face": true,
  "face_count": 1
}
```

The detector accepts detections with confidence greater than `0.6`.

### Error responses

Missing image field:

```json
{
  "error": "image file is required"
}
```

Invalid image data:

```json
{
  "error": "invalid image"
}
```

Both errors return HTTP status `400`.

## Project structure

```text
.
├── app.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── models/
    ├── deploy.prototxt
    └── res10_300x300_ssd_iter_140000.caffemodel
```

## License

Add your preferred license before publishing this repository.
