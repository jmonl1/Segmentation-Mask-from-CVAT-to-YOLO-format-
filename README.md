# Segmentation-Mask-from-CVAT-to-YOLO-format-
Modified the Computer Vision Engineer's mask_to_poly.py code from: https://www.youtube.com/watch?v=aVKGjzAUHz0&amp;t=785s. To work with multiple classes instead of just 1. 

# Steps:
1. At around 10 minutes into the video he shows you how to export your data in Segmentation mask form. Do that.
2. Once Downloaded. Extract the File from the zip. (Pretty Simple)
3. Put MaskToPoly.py (from here) into the same file as the files SegmentationClass and labelmap.txt
4. Then make a file called labels. Or whatever you want and then change the line that says: output_dir = './labels' to output_dir = './Whatever You Want To Call The File'
5. Save any changes then Run MaskToPoly.py
6. Once it is done running check the file called labels and the text files should look like 1.txt can be longer or short though depending on the segment mask. 
7. These are your labels for training.
8. So do what he does in the video with the labels and copy and paste into labels->train(or val).
9. View the Config I have added. The main difference to his is changing the nc to the number of classes you have and listing out all the classes in the same order as it is in labelmap.txt, which is in the segmentation folder you downloaded.
11. Train your model on epoch = 1 to see if it worked. To check look at the confusion matrix, batch, and val images.
12. That's it. Hopefully, it works for you. If not. Oops sorry :(

# Also here is the segmentation code link from ultralytics https://docs.ultralytics.com/guides/instance-segmentation-and-tracking/#what-is-instance-segmentation :)
If you are confused by it the code I used is segmentPredict. Which is the same pretty much. 
