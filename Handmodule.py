import cv2

import numpy as np

import tensorflow as tf

class handDetector():

    def __init__(self, model_file, input_size=256, confidence_threshold=0.5):

        self.input_size = input_size

        self.confidence_threshold = confidence_threshold

        

        self.interpreter = tf.lite.Interpreter(model_path=model_file)

        self.interpreter.allocate_tensors()

        

        self.input_details = self.interpreter.get_input_details()

        self.output_details = self.interpreter.get_output_details()

        

    def preprocess(self, frame):

        # Resize and normalize the input image

        resized = cv2.resize(frame, (self.input_size, self.input_size))

        normalized = resized / 255.0

        

        # Add a batch dimension to the input

        input_data = np.expand_dims(normalized, axis=0)

        

        return input_data

        

    def postprocess(self, outputs, frame):

        # Get the hand detections and landmarks from the output tensor

        detections = outputs[0]

        landmarks = outputs[1:]

        

        hands = []

        for i in range(detections.shape[1]):

            confidence = detections[0, i, 2]

            if confidence >= self.confidence_threshold:

                x1 = int(detections[0, i, 3] * frame.shape[1])

                y1 = int(detections[0, i, 4] * frame.shape[0])

                x2 = int(detections[0, i, 5] * frame.shape[1])

                y2 = int(detections[0, i, 6] * frame.shape[0])

                

                landmarks_ = landmarks[i].reshape(-1, 2)

                landmarks_ = landmarks_ * np.array([frame.shape[1], frame.shape[0]])

                landmarks_ = landmarks_.astype(np.int32)

                

                hands.append({

                    "bbox": [x1, y1, x2, y2],

                    "landmarks": landmarks_.tolist(),

                    "confidence": float(confidence)

                })

                

        return hands

        

    def findHands(self, frame, draw=True):

        # Preprocess the input frame

        input_data = self.preprocess(frame)

        

        # Run the inference

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        self.interpreter.invoke()

        

        # Get the output tensors

        outputs = []

        for detail in self.output_details:

            outputs.append(self.interpreter.get_tensor(detail['index']))

        

        # Postprocess the outputs

        hands = self.postprocess(outputs, frame)

        

        if draw:

            for hand in hands:

                # Draw the bounding box and landmarks

                x1, y1, x2, y2 = hand["bbox"]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                for x, y in hand["landmarks"]:

                    cv2.circle(frame, (x, y), 5, (0, 0, 255), cv2.FILLED)

        

        return frame

    

    def findPosition(self, frame, handNo=0, draw=True):

        hands = self.findHands(frame, draw=False)

        

        if len(hands) > handNo:

            hand = hands[handNo]

            landmarks = hand["landmarks"]

            xList = [lm[0] for lm in landmarks]

            yList = [lm[1] for lm in landmarks]

            bbox = hand["bbox"]
        xMin, yMin = min(xList), min(yList)

        xMax, yMax = max(xList), max(yList)

        return (xMin, yMin, xMax, yMax), landmarks

    else:

        return None, None

def findDistance(self, p1, p2, frame, draw=True):

    x1, y1 = p1

    x2, y2 = p2

    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    if draw:

        cv2.circle(frame, (x1, y1), 10, (255, 0, 255), cv2.FILLED)

        cv2.circle(frame, (x2, y2), 10, (255, 0, 255), cv2.FILLED)

        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

        cv2.circle(frame, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

    length = math.hypot(x2 - x1, y2 - y1)

    return length, frame

def findVolume(self, frame, volBar, volPer):

    pass


