import cv2
from ultralytics import YOLO

def detect_traffic_objects(image_path: str, model_path: str = 'yolov8n.pt'):
    """
    Performs object detection on a given image using a YOLOv8 model.

    Args:
        image_path (str): The URL or local path to the input image.
        model_path (str): The path to the YOLOv8 model weights. 
                          'yolov8n.pt' is a pre-trained model.
    """
    # Load the pre-trained YOLOv8 model. The model will be downloaded automatically if not present.
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # A list of the classes we are interested in. These correspond to the COCO dataset's class IDs.
    # The full list of COCO classes is available in the Ultralytics documentation.
    # 
    traffic_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'traffic light', 'stop sign'
    ]
    
    # Run inference on the image. The source can be a local file or a URL.
    print(f"Running detection on {image_path}...")
    try:
        results = model(image_path)
    except Exception as e:
        print(f"Error running inference: {e}")
        return

    # Process and display the results
    for result in results:
        # Get the names of the detected objects and their confidence scores
        boxes = result.boxes
        class_names = [model.names[int(cls)] for cls in boxes.cls]
        confidences = [round(float(conf), 2) for conf in boxes.conf]
        
        # Print detected objects
        print("\nDetected Objects:")
        for name, conf in zip(class_names, confidences):
            if name in traffic_classes:
                print(f"  - {name.capitalize()} (Confidence: {conf})")

        # Load the image using OpenCV to draw bounding boxes
        img = cv2.imread(result.path)

        # Iterate through each detected box and draw it on the image
        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            
            # Check if the detected object is a traffic-related class
            if label in traffic_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                # Draw the bounding box
                color = (0, 255, 0) # Green BGR color
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Create the label text with confidence score
                label_text = f"{label.capitalize()}: {confidence:.2f}"
                
                # Put the label text above the bounding box
                cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the image with detections
        cv2.imshow("Traffic Detection Result", img)
        cv2.waitKey(0) # Wait until a key is pressed to close the window
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Test image hosted on a publicly available service
    # You can replace this URL with your own image URL or a local path like 'path/to/your/image.jpg'
    test_image_url = "https://ultralytics.com/images/bus.jpg"
    
    detect_traffic_objects(test_image_url)
