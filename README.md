# Attention Detection system using Landmark detection

## Pre-requisites

### pytesseract Library

#### Windows Installation:

* Download binary from https://github.com/UB-Mannheim/tesseract/wiki.

* For Windows 32 bit, add pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)
  \\Tesseract-OCR\\tesseract.exe' to your script. (replace path of tesseract binary if necessary)

* For Windows 64 bit, add pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe

## Workflow

The model makes use of the Face Alignment library to detect faces and genrerate the landmarks on them using the s3fd
detector.

The video is processed frame by frame, the frame is cropped, chopped and then processed to compute the attention scores
and classes

## How to run?

Open cmd in where your venv is installed

Activate the venv
>venv\Scripts\activate

Navigate to modules directory where the "flask-app.py" is located
>cd change\location\to\the\directory\modules\where\flask-app.py

set the app to run
>set FLASK_APP = flask-app

run the app
>flask run
