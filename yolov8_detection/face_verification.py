import cv2
import argparse
import numpy as np
from insightface.app import FaceAnalysis


parser = argparse.ArgumentParser(description='Face Verification')
parser.add_argument('--image1', type=str, help='Path to first image')
parser.add_argument('--image2', type=str, help='Path to second image')
args = parser.parse_args()


img1 = cv2.imread(args.image1)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img2 = cv2.imread(args.image2)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

model = FaceAnalysis(providers=['CPUExecutionProvider'], name="buffalo_s")
model.prepare(ctx_id=0 , det_size=(640,640))

embedding1 = model.get(img1)[0]["embedding"]
embedding2 = model.get(img2)[0]["embedding"]

distance = np.sqrt(np.sum((embedding1 - embedding2)** 2))

threshold = 25 
if distance < threshold:
    print('Same Person')
elif distance > threshold:
    print('Different Person')