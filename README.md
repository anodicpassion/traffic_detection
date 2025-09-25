# YOLOv8 Traffic Detection

## Introduction

This project demonstrates a simple yet powerful application of the YOLOv8 (You Only Look Once) object detection model for identifying various traffic-related objects. The script uses the ultralytics Python library to analyze a given image and detect objects such as cars, buses, motorcycles, traffic lights, and more. It then draws bounding boxes around the detected objects and displays the results. This is a great starting point for anyone interested in real-time object detection for computer vision applications.

---
## Installation

To run this project, you need to have Python installed. The core functionality relies on the ultralytics library, which can be easily installed using pip.
Install Python: Ensure you have Python 3.8 or a newer version installed on your system.
Install Required Libraries: Open your terminal or command prompt and run the following command:

'''pip install ultralytics'''


This command will automatically install all necessary dependencies, including torch and opencv, which are required for the model and image processing.

---

## Usage
Save the Script: Save the provided Python code as traffic_detection.py in a directory of your choice.
Run from the Command Line: Navigate to the directory where you saved the file and run the script using the Python interpreter.
'''python traffic_detection.py'''


The script is configured to use a test image hosted online by default. You can modify the test_image_url variable in the if '''__name__ == "__main__":'''  block to a different URL or a local image path.

## Results
After running the script, a new window will pop up displaying the input image with bounding boxes drawn around the detected traffic objects. The objects will be labeled with their class (e.g., Car, Bus) and a confidence score. An example of the output is shown below.

<div align="center">
  <img width="406" height="569" alt="Screenshot 2025-09-25 at 8 18 22 PM" src="https://github.com/user-attachments/assets/34a9ec8b-7af5-4058-b17d-636b94e482d0" />
</div>

## License
This project is licensed under the Apache 2.0 License. See the LICENSE file for more details.

