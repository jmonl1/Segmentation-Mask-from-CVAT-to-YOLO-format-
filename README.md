# Segmentation-Mask-from-CVAT-to-YOLO-format-
Modified the Computer Vision Engineer's mask_to_poly.py code from: https://www.youtube.com/watch?v=aVKGjzAUHz0&amp;t=785s. To work with multiple classes instead of just 1. 

# Steps:
1. At around 10 minutes into the video he shows you how to export your data in Segmentation mask form. Do that.
2. Once Downloaded Extract File from zip. (Pretty Simple)
3. Put MaskToPoly.py (from here) into the same file as the files SegmentationClass and labelmap.txt
4. Then make a file called labels. Or whatever you want and then change the line that says: output_dir = './labels' to output_dir = './Whatever You Want To Call The File'
5. Save any changes then Run MaskToPoly.py
6. Once it is done running check the file called labels and the text files should look like 1 0.7808333333333334 0.805094130675526 0.7808333333333334 0.8073089700996677 0.7816666666666666. But Longer obviously.
7. These are your labels for training.
8. So do what he does in the video with the labels and copy and paste into labels->train(or val).
9. For config it should look like this 

   path: /Users/Jacks/Downloads/detection/datasets
   train: images/train 
   val: images/train  
   nc: 6
   names: ['Fence','FloorObstacle', 'Rock', 'Tree','Car','Person']

The main difference to his is changing the nc to the number of classes you have and listing out all the classes in the same order as it is in labelmap.txt, which is in the segmentation folder you downloaded.
11. Train your model on epoch = 1 to see if it worked. To check look at the confusion matrix, batch, and val images.
12. That's it. Hopefully, it works for you. If not. Oops sorry :(

# Also here is the segmentation code for running the model on a video. Not mine. This is from ultralytics. Just thought I would save you the time :) 
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import os
import numpy as np
model_path = os.path.join('.', 'runs', 'segment', 'train10', 'weights', 'last.pt') # This is the path to last.pt. He shows you how to do it in the video I believe. 

model = YOLO(model_path)
names = model.model.names
video_path = os.path.join('.', 'Record_2024-01-12-13-07-36.mp4') # this is the path to the video I used. Yours will probably be different. 
cap = cv2.VideoCapture(video_path)
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

out = cv2.VideoWriter('instance-segmentation.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))

while True:
    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    results = model.predict(im0)
    annotator = Annotator(im0, line_width=2)

    if results[0].masks is not None:
        clss = results[0].boxes.cls.cpu().tolist()
        masks = results[0].masks.xy
        for mask, cls in zip(masks, clss):
            annotator.seg_bbox(mask=mask,
                               mask_color=colors(int(cls), True),
                               det_label=names[int(cls)])

    out.write(im0)
    cv2.imshow("instance-segmentation", im0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()
