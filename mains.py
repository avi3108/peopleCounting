import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import Tracker
import time


class PersonCounter:
    def __init__(self, video_path, output_path, model_path='models/yolov8m.pt', target_fps=30):
        self.video_path = video_path
        self.output_path = output_path
        self.model = YOLO(model_path)
        self.area1 = [(140, 285), (166, 292), (109, 334), (86, 318)]
        self.area2 = [(47, 300), (109, 268), (131, 279), (77, 313)]
        self.class_list = self.load_class_list('models/coco.txt')
        self.tracker = Tracker()
        self.entering = set()
        self.exiting = set()
        self.people_enter = {}
        self.people_exiting = {}
        self.target_fps = target_fps
        self.frame_counter = 0
        self.start_time = time.time()
    
    def load_class_list(self, file_path):
        with open(file_path, "r") as file:
            data = file.read()
            return data.split("\n")

    def process_frame(self, frame,cap):
        frame = cv2.resize(frame, (1020, 500))
        results = self.model.predict(frame)
        a = results[0].boxes.boxes
        px = pd.DataFrame(a).astype("float")

        object_list = []

        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            c = self.class_list[d]
            if 'person' in c:
                object_list.append([x1, y1, x2, y2])

        bbox_id = self.tracker.update(object_list)

        for bbox in bbox_id:
            x3, y3, x4, y4, id = bbox
            results = cv2.pointPolygonTest(np.array(self.area2, np.int32), ((x4, y4)), False)
            if results >= 0:
                self.people_enter[id] = (x4, y4) 
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)

            if id in self.people_enter:
                results1 = cv2.pointPolygonTest(np.array(self.area1, np.int32), ((x4, y4)), False)
                if results1 >= 0:
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                    cv2.circle(frame, (x4, y4), 4, (255, 0, 255), -1)
                    cv2.putText(frame, str(id), (x3, 3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    self.entering.add(id)

            results2 = cv2.pointPolygonTest(np.array(self.area1, np.int32), ((x4, y4)), False)
            if results2 >= 0:
                self.people_exiting[id] = (x4, y4) 
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)

            if id in self.people_exiting:
                results3 = cv2.pointPolygonTest(np.array(self.area2, np.int32), ((x4, y4)), False)
                if results3 >= 0:
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
                    cv2.circle(frame, (x4, y4), 4, (255, 0, 255), -1)
                    cv2.putText(frame, str(id), (x3, 3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    self.exiting.add(id)
        cv2.polylines(frame, [np.array(self.area1, np.int32)], True, (255, 0, 0), 2)
        cv2.putText(frame, '1', (504, 471), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

        cv2.polylines(frame, [np.array(self.area2, np.int32)], True, (255, 0, 0), 2)
        cv2.putText(frame, '2', (466, 485), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

        enter_text = f"Enter: {len(self.entering)}"
        cv2.putText(frame, enter_text, (800, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        exit_text = f"Exit: {len(self.exiting)}"
        cv2.putText(frame, exit_text, (800, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Calculate and display adjusted frame rate
        self.frame_counter += 1
        elapsed_time = time.time() - self.start_time
        adjusted_fps = self.frame_counter / elapsed_time
        cv2.putText(frame, f"FPS: {adjusted_fps:.2f}", (800, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        return frame
    

    def run(self):
        #to write the video please uncommit below commented code
        #fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #out = cv2.VideoWriter(self.output_path, fourcc, 30.0, (1020, 500))

        cap = cv2.VideoCapture(self.video_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame
            processed_frame = self.process_frame(frame,cap)
           # out.write(processed_frame)
            cv2.imshow("RGB", processed_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        #out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import time

    input_video_path = 'input_videos/exit1.mp4'
    output_video_path = 'output_videos/exit_example.avi'
    # model_path='models/yolov8m.pt'
    target_fps = 30  # Adjust the target frame rate as needed
    people_counter = PersonCounter(input_video_path, output_video_path, target_fps=target_fps)
    people_counter.run()