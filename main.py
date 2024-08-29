import cv2
import NVision as Vision


#Vision.start_multiprocessing()
_, frame = Vision.cap.read()

while (True):
    rover_objects, frame, drawn_frame = Vision.get_rover_objects(processed_frame=cv2.flip(frame,0), get_new_frame=True,
                                                                 draw_cv=True)
    #navigation(rover_objects)
    if cv2.waitKey(1) == 27:
        break

Vision.cap.release()
cv2.destroyAllWindows()
