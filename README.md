![image](https://github.com/jmonl1/Segmentation-Mask-from-CVAT-to-YOLO-format-/assets/47769476/1320bc26-7199-4321-b1a1-75a93f18546a)# Segmentation-Mask-from-CVAT-to-YOLO-format-
Modified the Computer Vision Engineer's mask_to_poly.py code from: https://www.youtube.com/watch?v=aVKGjzAUHz0&amp;t=785s. To work with multiple classes instead of just 1. 

# Steps:
1. At around 10 minutes into the video he shows you how to export your data in Segmentation mask form. Do that.
2. Once Downloaded. Extract the File from the zip. (Pretty Simple)
3. Put MaskToPoly.py (from here) into the same file as the files SegmentationClass and labelmap.txt
![image](https://github.com/jmonl1/Segmentation-Mask-from-CVAT-to-YOLO-format-/assets/47769476/3dbdb2d7-9757-44de-8182-2094c1f457f2)
4. Then make a file called labels. Or whatever you want and then change the line that says: output_dir = './labels' to output_dir = './Whatever You Want To Call The File'
5. Save any changes then Run MaskToPoly.py
6. Once it is done running check the file called labels and the text files should look like 1.txt can be longer or short though depending on the segment mask. 
![image](https://github.com/jmonl1/Segmentation-Mask-from-CVAT-to-YOLO-format-/assets/47769476/38e4bd25-109a-4e48-975c-678b4cc0d579)
7. These are your labels for training.
8. So do what he does in the video with the labels and copy and paste into labels->train(or val).
9. View the config I have added or the picture under this. The main difference to his is changing the nc to the number of classes you have and listing out all the classes in the same order as it is in labelmap.txt, which is in the segmentation folder you downloaded.
![image](https://github.com/jmonl1/Segmentation-Mask-from-CVAT-to-YOLO-format-/assets/47769476/35cd8425-b5a5-4f82-bcde-06b61fc9b1e3)
11. Train your model on epoch = 1 to see if it worked. To check look at the confusion matrix, batch, and val images.
![image](https://github.com/jmonl1/Segmentation-Mask-from-CVAT-to-YOLO-format-/assets/47769476/1373edc3-727f-4c00-96b5-a4f879c691c3)
12. That's it. Hopefully, it works for you. If not. Oops sorry :(

# Also here is the segmentation code link from ultralytics https://docs.ultralytics.com/guides/instance-segmentation-and-tracking/#what-is-instance-segmentation :)
If you are confused by it the code I used is segmentPredict. Which is the same pretty much. 
