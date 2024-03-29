import numpy as np
from scipy import ndimage
# import matplotlib.pyplot as plt
import cv2
import pytesseract
import distance
from time import time
import os

# Import OCR's
import tensorflow as tf
import easyocr
import keras_ocr

# Disable Tensorflow Logger - Faulty
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Automatic brightness and contrast optimization with histogram clipping
# https://stackoverflow.com/questions/56905592/automatic-contrast-and-brightness-adjustment-of-a-color-photo-of-a-sheet-of-pape/56909036
def automatic_brightness_and_contrast(image, clip_hist_percent=1):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Calculate grayscale histogram
	hist = cv2.calcHist([gray],[0],None,[256],[0,256])
	hist_size = len(hist)

	# Calculate cumulative distribution from the histogram
	accumulator = []
	accumulator.append(float(hist[0]))
	for index in range(1, hist_size):
		accumulator.append(accumulator[index -1] + float(hist[index]))

	# Locate points to clip
	maximum = accumulator[-1]
	clip_hist_percent *= (maximum/100.0)
	clip_hist_percent /= 2.0

	# Locate left cut
	minimum_gray = 0
	while accumulator[minimum_gray] < clip_hist_percent:
		minimum_gray += 1

	# Locate right cut
	maximum_gray = hist_size -1
	while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
		maximum_gray -= 1

	# Calculate alpha and beta values
	if(maximum_gray != minimum_gray):
		alpha = 255 / (maximum_gray - minimum_gray)
	else:
		alpha = 255 / 1
	beta = -minimum_gray * alpha

	auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
	return (auto_result, alpha, beta)


def read_image(file_name):
	return cv2.imread('img/' + file_name)


# Perspective Transformation to improve readability
# https://learnopencv.com/automatic-document-scanner-using-opencv/
# https://dontrepeatyourself.org/post/learn-opencv-by-building-a-document-scanner/
def perspective_transform(img, preProParameter):
	original = img.copy()

	# Morphological closing operation to get rid of details
	morphological = img.copy()
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (preProParameter[0], preProParameter[0]))
	morphological = cv2.morphologyEx(morphological, cv2.MORPH_CLOSE, kernel, iterations= 3)

	# Remove Background --> GrabCut too expensive and unreliable, discarded
	if 0:
		mask = np.zeros(morphological.shape[:2],np.uint8)
		bgdModel = np.zeros((1,65),np.float64)
		fgdModel = np.zeros((1,65),np.float64)
		rect = (20,20,morphological.shape[1]-20,morphological.shape[0]-20)
		cv2.grabCut(morphological,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
		mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
		foreground = morphological*mask2[:,:,np.newaxis]

	# Convert to grayscale and blur
	gray = cv2.cvtColor(morphological, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (preProParameter[1], preProParameter[1]), 0)

	# Edge Detection
	canny = cv2.Canny(gray, 0, preProParameter[2])
	canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

	# Blank canvas
	con = np.zeros_like(original)

	# Finding contours for the detected edges
	contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

	# Keeping only the 5 largest detected contour
	page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
	con = cv2.drawContours(con, page, -1, (0, 255, 255), 3)

	final = []

	if len(page):
		# Blank canvas
		con = np.zeros_like(original)

		# Loop over the contours
		for c in page:
			# Approximate the contour
			epsilon = 0.02 * cv2.arcLength(c, True)
			corners = cv2.approxPolyDP(c, epsilon, True)
			# If our approximated contour has four points
			if len(corners) == 4:
				con = np.zeros_like(original)
				con = cv2.drawContours(con, c, -1, (0, 255, 255), 3)
				break

		cv2.drawContours(con, c, -1, (0, 255, 255), 3)
		# cv2.imwrite('img/ContourSansEdge.jpg', con)
		cv2.drawContours(con, corners, -1, (0, 255, 0), 10)

		if len(corners) < 4:
			return original # Return original if no four corners found

		# Sorting the corners and converting them to desired shape
		corners = sorted(np.concatenate(corners).tolist())
		corners = order_points(corners)

		# Displaying the corners
		for index, c in enumerate(corners):
			character = chr(65 + index)
			cv2.putText(con, character, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)
			cv2.putText(con, character, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

		(tl, tr, br, bl) = corners

		# Finding the maximum width
		widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
		widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
		maxWidth = max(int(widthA), int(widthB))

		# Finding the maximum height
		heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
		heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
		maxHeight = max(int(heightA), int(heightB))

		# Final destination co-ordinates
		destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]

		# Getting the homography
		M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))

		# Perspective transform using homography
		final = cv2.warpPerspective(original, M, (destination_corners[2][0], destination_corners[2][1]), flags=cv2.INTER_LINEAR)

		# Fix error with dimensions exceeding original image
		if final.shape[0] > original.shape[0] or final.shape[1] > original.shape[1]:
			# print ('Error: image dimension somehow exceeded original dimensions')
			final = final[:original.shape[0], :original.shape[1]]

		# Thresholding
		binarized = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
		binarized = cv2.GaussianBlur(binarized,(1,1),0)
		ret, binarized = cv2.threshold(binarized,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

		cv2.imwrite('img/canny.jpg', cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB))
		cv2.imwrite('img/contour.jpg', con)
		cv2.imwrite('img/transformed.jpg', final)
		cv2.imwrite('img/binarized.jpg', binarized)

	# Create collage of all Steps
	if 1:
		height = original.shape[0]
		width = original.shape[1]
		new_shape = (height*2, width*3, 3)

		collage = np.zeros((new_shape), np.uint8)

		collage[0:height, 0:width] = original
		collage[0:height, width:width*2] = morphological
		collage[0:height, width*2:width*3] = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
		collage[height:height*2, 0:width] = con
		if len(final):
			collage[height:height+final.shape[0], width:width+final.shape[1]] = final
			collage[height:height+final.shape[0], width*2:width*2+final.shape[1]] = cv2.cvtColor(binarized, cv2.COLOR_GRAY2RGB)
		
		cv2.imwrite('img/collageDetection.jpg', collage)

	if len(final):
		return final
		# return cv2.cvtColor(binarized, cv2.COLOR_GRAY2RGB) # Binarization causes illegible text under certain circumstances, discarded
	else:
		return original # Return original if no contour found


def order_points(pts):
	# List all x values
	x = [pts[0][0], pts[1][0], pts[2][0], pts[3][0]]

	# Separate coordinates into left and right
	left = [pts[np.argsort(x)[0]], pts[np.argsort(x)[1]]]
	right = [pts[np.argsort(x)[2]], pts[np.argsort(x)[3]]]

	# Separate left and right coordinates into top and bottom
	tl = [] # Top-left
	tr = [] # Top-right
	bl = [] # Bottom-left
	br = [] # Bottom-right

	# Sort Left
	if left[0][1] <= left[1][1]:
		tl = left[0]
		bl = left[1]
	else:
		tl = left[1]
		bl = left[0]

	# Sort Right
	if right[0][1] <= right[1][1]:
		tr = right[0]
		br = right[1]
	else:
		tr = right[1]
		br = right[0]

	return [tl, tr, br, bl]


def preprocessing(img):
	cv2.imwrite('img/image.jpg', img)

	# Auto Contrast
	processed_image, alpha, beta = automatic_brightness_and_contrast(img)

	cv2.imwrite('img/preprocessing.jpg', processed_image)

	return processed_image


def recognize_text_tesseract(img):
	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img_dict = pytesseract.image_to_data(img_rgb, output_type='dict')

	word_indices = []
	for i in range(len(img_dict['conf'])):
		if (img_dict['conf'][i] != -1):
			word_indices.append(i)

	dict_tesseract = []
	for i in word_indices:
		dict_tesseract.append({
			'text': img_dict['text'][i],
			'conf': img_dict['conf'][i],
			'left': int(img_dict['left'][i]),
			'top': int(img_dict['top'][i]),
			'width': int(img_dict['width'][i]),
			'height': int(img_dict['height'][i])
		})

	return dict_tesseract


def recognize_text_easyocr(img):
	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	reader = easyocr.Reader(['en'])
	result = reader.readtext(img_rgb)
	
	dict_easyocr = []
	for i in result:
		if(i[2] > 0.05):
			dict_easyocr.append({
				'text': i[1],
				'conf': i[2],
				'left': int(i[0][0][0]),
				'top': int(i[0][0][1]),
				'width': int(i[0][2][0]) - int(i[0][0][0]),
				'height': int(i[0][2][1]) - int(i[0][0][1])
			})

	# if not dict_easyocr:
	# 	print('Easy OCR could not recognize any text')
	return dict_easyocr


def recognize_text_keras_ocr(img):
	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	pipeline = keras_ocr.pipeline.Pipeline()
	prediction_groups = pipeline.recognize([keras_ocr.tools.read(img_rgb)])

	dict_keras_ocr = []
	for i in prediction_groups[0]:
		dict_keras_ocr.append({
			'text': i[0],
			'conf': '', # No confidence available
			'left': int(i[1][0][0]),
			'top': int(i[1][0][1]),
			'width': int(i[1][2][0]) - int(i[1][0][0]),
			'height': int(i[1][2][1]) - int(i[1][0][1])
		})

	return dict_keras_ocr


def print_dicts(dicts):
	names = ['Tesseract', 'EasyOCR', 'KerasOCR']
	counter = 0
	for dictionary in dicts:
		print(names[counter] + ':')
		for entry in dictionary:
			print(entry)
		print('\n')
		counter += 1


def draw_output(img, words, output_name):
	final_image = np.copy(img)

	for word in words:
		final_image = cv2.rectangle(final_image,(word['left'], word['top']), (word['left'] + word['width'], word['top'] + word['height']), (0, 255, 0), 2)
		final_image = cv2.putText(final_image, word['text'], (word['left'] + 5, word['top'] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3 , cv2.LINE_AA)		# Stroke
		final_image = cv2.putText(final_image, word['text'], (word['left'] + 5, word['top'] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1 , cv2.LINE_AA)

	cv2.imwrite('img/' + output_name + '.jpg', final_image)

	return final_image


def evaluate(dict_ocr, target_word):
	for word in dict_ocr:
		# print(target_word, word['text'])
		if 1 - (distance.levenshtein(target_word.lower(), word['text'].lower()))/max(len(target_word), len(word['text'])) >= 0.6: # Threshold for text similarity should around 0.6 or higher
			return True, word

	return False, 'Not found'


if __name__ == '__main__':
	# Full Multi-image pipeline on cropped PR2 images
	if 0:
		resultDir = 'img/PR2_Test'

		if not os.path.exists(resultDir):
			os.mkdir(resultDir)

		# 0: Tesseract, 1: EasyOCR, 2: Keras-OCR
		ocr = 1

		start_time = time()

		preProParameters = [
			[1, 3, 200],
			[1, 3, 100],
			[3, 3, 80],
			[5, 7, 50],
			[9, 15, 10]
		]

		for imageNr in range(0, 75+1): # 75 PR2 Images
			file_name = 'PR2_cropped/' + str(imageNr) + '.jpg'
			print(file_name)

			img_cv = read_image(file_name)
			img_preprocessed = preprocessing(img_cv)
			for parameter in preProParameters:
				img_transformed = perspective_transform(img_preprocessed, parameter)
				if ocr == 0:
					dict_ocr = recognize_text_tesseract(img_transformed)
				if ocr == 1:
					dict_ocr = recognize_text_easyocr(img_transformed)
				if ocr == 2:
					dict_ocr = recognize_text_keras_ocr(img_transformed)

				if len(dict_ocr) > 0:
					# Enable Rotation check
					if 0:
						for rotation in [90, 180, 270]:
							if rotation == 90:
								img_transformed = cv2.rotate(img_transformed, cv2.ROTATE_90_CLOCKWISE)
							if rotation == 180:
								img_transformed = cv2.rotate(img_transformed, cv2.ROTATE_180)
							if rotation == 270:
								img_transformed = cv2.rotate(img_transformed, cv2.ROTATE_90_COUNTERCLOCKWISE)

							if ocr == 0:
								dict_ocr = recognize_text_tesseract(img_transformed)
							if ocr == 1:
								dict_ocr = recognize_text_easyocr(img_transformed)
							if ocr == 2:
								dict_ocr = recognize_text_keras_ocr(img_transformed)

							if len(dict_ocr) > 0:
								break

				if len(dict_ocr) > 0:
					break

			if ocr == 0:
				final = draw_output(img_transformed, dict_ocr, 'tesseract')
			if ocr == 1:
				final = draw_output(img_transformed, dict_ocr, 'easyocr')
			if ocr == 2:
				final = draw_output(img_transformed, dict_ocr, 'keras_ocr')

			cv2.imwrite(resultDir + '/' + str(imageNr) + '_OCR.jpg', final)
			collage = read_image('collageDetection.jpg')
			cv2.imwrite(resultDir + '/' + str(imageNr) + '_Collage.jpg', collage)

			f = open(resultDir + '/' + str(imageNr) + '.txt', 'w')
			results = ''
			for entry in dict_ocr:
				results += 'Text: ' + entry['text']
				results += ', Confidence: ' + str(entry['conf'])
				results += ', Left: ' + str(entry['left'])
				results += ', Top: ' + str(entry['top'])
				results += ', Width: ' + str(entry['width'])
				results += ', Height: ' + str(entry['height'])
				results += '\n'

			f.write(results)

		end_time = time()
		print('Processing time: ' + str(end_time - start_time))

	# Text recognition on cropped PR2 images without object detection
	if 0:
		resultDir = 'img/PR2_Test_NoPipeline'

		if not os.path.exists(resultDir):
			os.mkdir(resultDir)

		# 0: Tesseract, 1: EasyOCR, 2: Keras-OCR
		ocr = 1

		for imageNr in range(0, 75+1): # 75 PR2 Images
			file_name = 'PR2_cropped/' + str(imageNr) + '.jpg'
			print(file_name)

			img_cv = read_image(file_name)
			if ocr == 0:
				dict_ocr = recognize_text_tesseract(img_cv)
				final = draw_output(img_cv, dict_ocr, 'tesseract')
			if ocr == 1:
				dict_ocr = recognize_text_easyocr(img_cv)
				final = draw_output(img_cv, dict_ocr, 'easyocr')
			if ocr == 2:
				dict_ocr = recognize_text_keras_ocr(img_cv)
				final = draw_output(img_cv, dict_ocr, 'keras_ocr')

			cv2.imwrite(resultDir + '/' + str(imageNr) + '_OCR.jpg', final)

			f = open(resultDir + '/' + str(imageNr) + '.txt', 'w')
			results = ''
			for entry in dict_ocr:
				results += 'Text: ' + entry['text']
				results += ', Confidence: ' + str(entry['conf'])
				results += ', Left: ' + str(entry['left'])
				results += ', Top: ' + str(entry['top'])
				results += ', Width: ' + str(entry['width'])
				results += ', Height: ' + str(entry['height'])
				results += '\n'

			f.write(results)

	# Single Image Version for image in /camera
	if 0:
		file_name = 'camera/' + 'MedicineBox_1.jpg'
		start_time = time()

		img_cv = read_image(file_name)
		img_preprocessed = preprocessing(img_cv)
		img_transformed = perspective_transform(img_preprocessed, [1, 3, 200])
		
		#dict_tesseract = recognize_text_tesseract(img_preprocessed)
		dict_easyocr = recognize_text_easyocr(img_transformed)
		#dict_keras_ocr = recognize_text_keras_ocr(img_preprocessed)

		#print_dicts([dict_tesseract, dict_easyocr, dict_keras_ocr])
		print_dicts([dict_easyocr])

		#draw_output(img_cv, dict_tesseract, 'tesseract')
		final_easyocr = draw_output(img_transformed, dict_easyocr, 'easyocr')
		#draw_output(img_cv, dict_keras_ocr, 'keras_ocr')

		end_time = time()
		cv2.imshow('Result', final_easyocr)
		cv2.waitKey(0)
		print('Processing time: ' + str(end_time - start_time))

	# Live Camera Version Full
	if 0:
		cap = cv2.VideoCapture(0)
		target_word = 'Foobar'

		while True:
			ret, frame = cap.read()
			height = 480
			dimension = (int(frame.shape[1]*(height/frame.shape[0])), height)
			frame = cv2.resize(frame, dimension, interpolation = cv2.INTER_AREA)

			# Object Detection
			img_preprocessed = preprocessing(frame)
			img_transformed = perspective_transform(img_preprocessed, [1, 3, 200])
			canny = read_image('canny.jpg')
			contour = read_image('contour.jpg')

			# Text Recognition
			dict_easyocr = recognize_text_easyocr(img_transformed)
			found, target = evaluate(dict_easyocr, target_word)
			final_easyocr = draw_output(img_transformed, dict_easyocr, 'easyocr')

			final_frame = np.zeros((frame.shape[0]*2, frame.shape[1]*3, 3), np.uint8)
			final_frame[:frame.shape[0], :frame.shape[1]] = frame
			final_frame[:frame.shape[0], frame.shape[1]:frame.shape[1]*2] = canny
			final_frame[:frame.shape[0], frame.shape[1]*2:] = contour
			final_frame[frame.shape[0]:frame.shape[0]+img_transformed.shape[0], :img_transformed.shape[1]] = img_transformed
			if not found:
				final_frame[frame.shape[0]:frame.shape[0]+final_easyocr.shape[0], frame.shape[1]:frame.shape[1]+final_easyocr.shape[1]] = final_easyocr
				final_frame[frame.shape[0]:frame.shape[0]+final_easyocr.shape[0], frame.shape[1]*2:frame.shape[1]*2+final_easyocr.shape[1]] = 128
			if found:
				final_frame[frame.shape[0]:frame.shape[0]+final_easyocr.shape[0], frame.shape[1]:frame.shape[1]+final_easyocr.shape[1]] = final_easyocr

				final_easyocr = cv2.rectangle(img_transformed,(target['left'], target['top']), (target['left'] + target['width'], target['top'] + target['height']), (0, 255, 0) ,3)
				final_easyocr = cv2.putText(final_easyocr, target['text'], (target['left'], target['top']), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3 , cv2.LINE_AA)
				final_easyocr = cv2.putText(final_easyocr, target['text'], (target['left'], target['top']), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1 , cv2.LINE_AA)
				final_easyocr = cv2.rectangle(final_easyocr, (0, 0), (final_easyocr.shape[1], final_easyocr.shape[0]), (0, 255, 0), 3)
				final_frame[frame.shape[0]:frame.shape[0]+final_easyocr.shape[0], frame.shape[1]*2:frame.shape[1]*2+final_easyocr.shape[1]] = final_easyocr

			cv2.imwrite('img/collageCamera.jpg', final_frame)
			cv2.imshow('Capture', final_frame)

			if cv2.waitKey(1) == ord('q'):
				break
		
		cap.release()
		cv2.destroyAllWindows()

	# Live Camera Version Partials
	if 0:
		cap = cv2.VideoCapture(0)

		while True:
			ret, frame = cap.read()
			height = 480
			dimension = (int(frame.shape[1]*(height/frame.shape[0])), height)
			frame = cv2.resize(frame, dimension, interpolation = cv2.INTER_AREA)

			# Object Detection
			if 0:
				img_preprocessed = preprocessing(frame)
				img_transformed = perspective_transform(img_preprocessed, [1, 3, 200])

				final_frame = np.zeros((frame.shape[0]*2, frame.shape[1], 3), np.uint8)
				final_frame[:frame.shape[0]] = frame
				final_frame[frame.shape[0]:frame.shape[0] + img_transformed.shape[0], :img_transformed.shape[1]] = img_transformed

			# Text Recognition
			if 0:
				target_word = 'Foobar'
				dict_easyocr = recognize_text_easyocr(frame)
				found, target = evaluate(dict_easyocr, target_word)
				final_easyocr = draw_output(frame, dict_easyocr, 'easyocr')

				final_frame = np.zeros((frame.shape[0]*2, frame.shape[1], 3), np.uint8)
				final_frame[:frame.shape[0]] = frame
				if not found:
					final_frame[frame.shape[0]:] = final_easyocr
				if found:
					final_easyocr = cv2.rectangle(frame,(target['left'], target['top']), (target['left'] + target['width'], target['top'] + target['height']), (0, 255, 0) ,3)
					final_easyocr = cv2.putText(final_easyocr, target['text'], (target['left'], target['top']), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1 , cv2.LINE_AA)
					final_frame[frame.shape[0]:] = final_easyocr
					final_frame[frame.shape[0]:frame.shape[0]+5] = [0, 255, 0]
					final_frame[frame.shape[0]:, :5] = [0, 255, 0]
					final_frame[final_frame.shape[0]-5:] = [0, 255, 0]
					final_frame[frame.shape[0]:, frame.shape[1]-5:] = [0, 255, 0]

			cv2.imshow('Capture', final_frame)
			

			if cv2.waitKey(1) == ord('q'):
				break
		
		cap.release()
		cv2.destroyAllWindows()

# TROUBLESHOOT to fix cv2.imshow error:
# pip install opencv-python==3.4.17.61
# pip uninstall opencv-python-headless
# pip uninstall opencv-python
# pip install opencv-python

# Show image via PLT
# plt.imshow()
# plt.show()