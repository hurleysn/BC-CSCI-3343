# (Matt) TODO => Clean up, abstract code and write documentation
from cv2 import cv2
import argparse
import numpy as np


class YOLO:
    def __init__(self, img, config, weights, classes, confidence_limit=.5, nms_threshold=.4, scale=.00392):
        self.img = None
        self.config = config
        self.weights = weights
        self.classes = classes  # Maybe unneeded?
        self.confidence_limit = confidence_limit
        self.nms_threshold = nms_threshold
        self.scale = scale
        self.net = None

        self.build_model(img)

    def build_model(self, img):
        self.read_image(img)
        self.read_model(self.weights, self.config)
        self.set_input_blob()

        out = self.gather_prediction_output()

        class_ids, confidences, boxes = self.scan_through_detections(out)

        valid_detections = self.remove_overlapping_boxes(boxes, confidences)
        self.final_scan(valid_detections, boxes, class_ids, confidences)
        self.show_image()

    def read_image(self, img):
        try:
            self.img = cv2.imread(img)
        except:
            raise Exception("Unable to safely parse the image.")

    def read_model(self, weights, config):
        try:
            self.net = cv2.dnn.readNet(weights, config)
        except:
            raise Exception(
                "Error when reading pretrained model with configuration")

    def create_input_blob(self):
        return cv2.dnn.blobFromImage(self.img, self.scale, (416, 416), (0, 0, 0), True, crop=False)

    def set_input_blob(self):
        try:
            blob = self.create_input_blob()
            self.net.setInput(blob)
        except:
            raise Exception("Unable to correctly set input blob")

    def get_output_layers(self):
        try:
            layers = self.net.getLayerNames()
            return [layers[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        except:
            raise Exception("Unable to determine output layer names")

    def draw_bounding_box(self, class_id, x, y, w, h):
        try:
            label = str(self.classes[class_id])
            color = np.random.uniform(0, 255, 3)
            cv2.rectangle(self.img, (x, y), (w, h), color, 2)
            cv2.putText(self.img, label, (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except:
            raise Exception("Unable to draw bounding boxes")

    def gather_prediction_output(self):
        try:
            return self.net.forward(self.get_output_layers())
        except:
            raise Exception("Error when running inference")

    def scan_through_detections(self, output_layers):
        # Define width/height for ease of use later
        width = self.img.shape[1]
        height = self.img.shape[0]

        # Used to store detected objects, their confidence and class
        class_ids = []
        confidences = []
        boxes = []

        for output_layer in output_layers:
            for detection in output_layer:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.confidence_limit:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        return class_ids, confidences, boxes

    def remove_overlapping_boxes(self, boxes, confidences):
        try:
            return cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_limit, self.nms_threshold)
        except:
            raise Exception("Error when applying non-max suppression")

    def final_scan(self, indices, boxes, class_id, confidences):
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]

            self.draw_bounding_box(class_id[i], round(x), round(y), round(x + w),
                                   round(y + h))

    def show_image(self):
        cv2.imshow("YOLO Object Detection", self.img)
        cv2.waitKey()
        cv2.imwrite("yolo-object-detection.jpg", self.img)
        cv2.destroyAllWindows()
