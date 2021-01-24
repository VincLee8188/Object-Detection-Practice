"""Study note:
Object localization is finding what and where a (single) object exists in an image
Object localization is finding what and where (multiple) objects are in an image

Two common ways to define bboxes:
1. (x1, y1) is upper left corner point, (x2, y2) is bottom right corner point.
2. Two points define a corner point, and two points to define height and width.

Approaches to conduct object detection:
1. Sliding windows.

Potential problems:
1. A lot of computation!  (OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks)
2. Many bounding boxes for same object. (non-max suppression) (regional based networks, R-CNN, Fast R-CNN, Faster R-CNN)
3. Complicated 2 step process. (Yolo - You only look once)
"""
