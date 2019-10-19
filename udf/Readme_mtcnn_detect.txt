Uses network of RNet, ONet, PNet to detect face on both camera or image. Also generate BoundingBox to mark the face.

Core function: Face Detect

Small faces are detected in large picture pyramid and large faces are detected in small picture pyramid.

Stage 1:
Pass in image
Create multiple scaled copies of the image
Feed scaled images into P-Net
Gather P-Net output
Delete bounding boxes with low confidence
Convert 12 x 12 kernel coordinates to “un-scaled image” coordinates
Non-Maximum Suppression for kernels in each scaled image
Non-Maximum Suppression for all kernels
Convert bounding box coordinates to “un-scaled image” coordinates
Reshape bounding boxes to square

Stage 2:
Pad out-of-bound boxes
Feed scaled images into R-Net
Gather R-Net output
Delete bounding boxes with low confidence
Non-Maximum Suppression for all boxes
Convert bounding box coordinates to “un-scaled image” coordinates
Reshape bounding boxes to square

Stage 3:
Pad out-of-bound boxes
Feed scaled images into O-Net
Gather O-Net output
Delete bounding boxes with low confidence
Convert bounding box and facial landmark coordinates to “un-scaled image” coordinates
Non-Maximum Suppression for all boxes

Delivering Results:
Package all coordinates and confidence levels into a dictionary
Return the dictionary

https://towardsdatascience.com/how-does-a-face-detection-program-work-using-neural-networks-17896df8e6ff

https://towardsdatascience.com/face-detection-neural-network-structure-257b8f6f85d1