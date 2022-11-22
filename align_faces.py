# import the necessary packages
from FaceAligner import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils 
import dlib
import cv2

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=256)

for i in range(100):
	# load the input image, resize it, and convert it to grayscale
	image = cv2.imread("Dataset/Train/img/"+str(i)+".png")
	image = imutils.resize(image, width=800)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# show the original input image and detect faces in the grayscale
	# image
	cv2.imshow("Input", image)
	rects = detector(gray, 2)

	# loop over the face detections
	for rect in rects:
		# extract the ROI of the *original* face, then align the face
		# using facial landmarks
		(x, y, w, h) = rect_to_bb(rect)
		faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
		faceAligned = fa.align(image, gray, rect)
		# display the output images
		cv2.imshow("Original", faceOrig)
		cv2.imwrite("Original/original" + str(i) + ".png", faceOrig)
		cv2.imshow("Face Aligned", faceAligned)
		cv2.waitKey(0)