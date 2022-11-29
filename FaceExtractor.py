from os import listdir
from os.path import isfile, join
import os;

# import the necessary packages
#from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from FaceAligner import FaceAligner
import argparse
import dlib
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-i", "--imageDir", required=False, help="path to input image directory")
ap.add_argument("-o", "--outDir", required=False, help="path to output image directory")
args = vars(ap.parse_args())


img_dir_path = args["imageDir"]
if not img_dir_path:
    img_dir_path = './Dataset/Train/img'

out_dir_path = args["outDir"]
if not out_dir_path:
    out_dir_path = './outimg'

onlyfiles = [f for f in listdir(img_dir_path) if isfile(join(img_dir_path, f)) and f.split(".")[1] == "png"]


if not os.path.exists(out_dir_path):
    os.makedirs(out_dir_path) 

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=256, desiredFaceHeight=256, desiredLeftEye=(0.33, 0.3))

for file in onlyfiles:
    print(f"Processing image {file}")
    img_file_name = f"{img_dir_path}/{file}"

    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(img_file_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # show the original input image and detect faces in the grayscale
    # image
    rects = detector(gray, 2)
    print(rects)
    if cv2.waitKey(10) == 27: # q를 누르면 종료
        break

    # loop over the face detections    
  
    face_count = 0
    for rect in rects:
        face_count += 1
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        (x, y, w, h) = rect_to_bb(rect)
        #faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
        faceAligned = fa.align(image, gray, rect)
        path, filename = os.path.split(img_file_name)
        new_file_name = join(out_dir_path, filename)
        # new_file_name = join(img_out_path, f"{img_file_name.split('.')[0]}_aligned_{face_count}.{img_file_name.split('.')[1]}")
        cv2.imwrite(new_file_name, faceAligned)
        cv2.imshow("Input", faceAligned)
        break

        
        