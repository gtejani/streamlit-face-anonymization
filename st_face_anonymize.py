import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import time

st.write("""
# Face Anonymisation App
Upload an image to blur the faces in the image.
""")

# Function to load and process the image
def load_image(image_file):
    img = Image.open(image_file)
    return img

def pil_to_cv2(image):
    image_np = np.array(image)
    # Convert RGB to BGR
    return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

def cv2_to_pil(image):
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)

# Using Gaussian blurring
def anonymize_face_gaussian(image, factor=3.0):
	# automatically determine the size of the blurring kernel based
	# on the spatial dimensions of the input image
	(h, w) = image.shape[:2]
	kW = int(w / factor)
	kH = int(h / factor)
	# ensure the width of the kernel is odd
	if kW % 2 == 0:
		kW -= 1
	# ensure the height of the kernel is odd
	if kH % 2 == 0:
		kH -= 1
	# apply a Gaussian blur to the input image using our computed
	# kernel size
	return cv2.GaussianBlur(image, (kW, kH), 0)

# Using Pixelated blurring
def anonymize_face_pixelate(image, blocks=3):
	# divide the input image into NxN blocks
	(h, w) = image.shape[:2]
	xSteps = np.linspace(0, w, blocks + 1, dtype="int")
	ySteps = np.linspace(0, h, blocks + 1, dtype="int")
	# loop over the blocks in both the x and y direction
	for i in range(1, len(ySteps)):
		for j in range(1, len(xSteps)):
			# compute the starting and ending (x, y)-coordinates
			# for the current block
			startX = xSteps[j - 1]
			startY = ySteps[i - 1]
			endX = xSteps[j]
			endY = ySteps[i]
			# extract the ROI using NumPy array slicing, compute the
			# mean of the ROI, and then draw a rectangle with the
			# mean RGB values over the ROI in the original image
			roi = image[startY:endY, startX:endX]
			(B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
			cv2.rectangle(image, (startX, startY), (endX, endY), (B, G, R), -1)
	# return the pixelated blurred image
	return image

image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if image_file is not None:
    image = load_image(image_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
	
    prototxtPath = os.path.sep.join(["models", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["models", "res10_300x300_ssd_iter_140000.caffemodel"])

    net = cv2.dnn.readNet(prototxtPath, weightsPath)
    
    image_cv2 = pil_to_cv2(image)
    orig = image_cv2.copy()
    (h, w) = image_cv2.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image_cv2, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    valid_dets = 0
    method = "gaussian"
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            try:
                # extract the face ROI
                face = image_cv2[startY:endY, startX:endX]
                # check to see if we are applying the "gaussian" face blurring method
                if method == "gaussian":
                    face = anonymize_face_gaussian(face, factor=3.0)
                else:
                    face = anonymize_face_pixelate(face, blocks=args["blocks"])
                # store the blurred face in the output image
                image_cv2[startY:endY, startX:endX] = face
                face_bool = 'Yes'
                valid_dets += 1
            except Exception as e:
                print('Bounding box error (outside image detections?)', e)
                continue 
    
    image_pil = cv2_to_pil(image_cv2)
    st.image(image_pil, caption="Blurred Face Image", use_column_width=True)
    
