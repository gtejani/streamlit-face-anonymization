import numpy as np
import argparse
import cv2
import os
import logging
import time
## Blur Functions

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

def main(args):
    # path and save_path 
    images_folders = []
    if os.path.isfile(args['image']):
        images_folders.append(args['image'])
    else:
        for dirpath, dirnames, filenames in os.walk(args['image'], followlinks=True):
            if filenames:
                images_folders.append(dirpath)

    # Configure logging
    logging.basicConfig(filename=args['logs'], level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create a logger
    logger = logging.getLogger(__name__)
    handler  = logging.StreamHandler()
    logger.addHandler(handler)

    # load our serialized face detector model from disk
    logger.info("Loading face detector model...")
    
    prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
    weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])

    net = cv2.dnn.readNet(prototxtPath, weightsPath)
    
    time_runs = []
    root_path = os.path.dirname(os.path.commonpath(images_folders))
    for folder in images_folders:
        if os.path.isfile(folder):
            images = [folder]
        else:
            images = [os.path.join(folder, i) for i in os.listdir(folder) if not os.path.isdir(os.path.join(folder, i))]
        
        if not images:
            continue
        save_dir = os.path.join(args['save_dir'], args['method'], os.path.relpath(folder, root_path))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        logger.info(f'Total number of images in {folder}: {len(images)}')
        for ind in range(len(images)):
            face_bool = 'No'
            image_path = images[ind]
            start = time.time()
            try:
                image = cv2.imread(image_path)
                orig = image.copy()
                (h, w) = image.shape[:2]
                # construct a blob from the image
                # blob = cv2.dnn.blobFromImage(image, 1.0, (h, w), (104.0, 177.0, 123.0))
                blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                # pass the blob through the network and obtain the face detections
                net.setInput(blob)
                detections = net.forward()
                # loop over the detections
                valid_dets = 0
                # print('detections: ', type(detections), detections.shape)
                for i in range(0, detections.shape[2]):
                    # extract the confidence (i.e., probability) associated with the detection
                    confidence = detections[0, 0, i, 2]
                    # filter out weak detections by ensuring the confidence is greater than the minimum confidence
                    if confidence > args["confidence"]:

                        # compute the (x, y)-coordinates of the bounding box for the object
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        try:
                            # extract the face ROI
                            face = image[startY:endY, startX:endX]
                            # check to see if we are applying the "gaussian" face blurring method
                            if args["method"] == "gaussian":
                                face = anonymize_face_gaussian(face, factor=3.0)
                            else:
                                face = anonymize_face_pixelate(face, blocks=args["blocks"])
                            # store the blurred face in the output image
                            image[startY:endY, startX:endX] = face
                            face_bool = 'Yes'
                            valid_dets += 1
                        except Exception as e:
                            print('Bounding box error (outside image detections?)', e)
                            continue    
                # display the original image and the output image with the blurred face(s) side by side
                end = time.time()
                time_runs.append(end - start)

                logger.info(f'time_taken: {round(end - start, 4)} | File_name: {image_path} | is_Face: {face_bool} | #Faces: {valid_dets}')
                # output = np.hstack([orig, image])
                cv2.imwrite(os.path.join(save_dir, os.path.basename(image_path)), image)
                # cv2.imshow("Output", output)
                # cv2.waitKey(0)
            except Exception as e:
                print('Image error: ', e)
                logger.warning(f'Corrupted/None Image File: {image_path}')
                continue
    if time_runs: avg_time = sum(time_runs) / len(time_runs)
    else: avg_time = 0.0
    logger.info(f'Average time to process 1 image: {round(avg_time, 4)} seconds')

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image")
    ap.add_argument("-s", "--save_dir", type=str, default='final_results', help="path to save face-blurred images")
    ap.add_argument("-f", "--face", default= 'models', help="path to face detector model directory")
    ap.add_argument("-m", "--method", type=str, default="gaussian", choices=["gaussian", "pixelated"], help="face blurring/anonymizing method")
    ap.add_argument("-b", "--blocks", type=int, default=20, help="# of blocks for the pixelated blurring method")
    ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
    ap.add_argument("-l", "--logs", type=str, default='run.log', help="path to save logs")

    args = vars(ap.parse_args())

    main(args)
