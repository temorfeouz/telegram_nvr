# install
# pip install cv2 imutils flask

# import the necessary packages
# from motion_detection import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2
import os
import requests
import numpy as np
import imutils

class SingleMotionDetector:
	def __init__(self, accumWeight=0.5):
		# store the accumulated weight factor
		self.accumWeight = accumWeight
		# initialize the background model
		self.bg = None
		
	def update(self, image):
		# if the background model is None, initialize it
		if self.bg is None:
			self.bg = image.copy().astype("float")
			return
		# update the background model by accumulating the weighted
		# average
		cv2.accumulateWeighted(image, self.bg, self.accumWeight)
		
	def detect(self, image, tVal=25):
		if self.bg is None:
			return None
		# compute the absolute difference between the background model
		# and the image passed in, then threshold the delta image
		delta = cv2.absdiff(self.bg.astype("uint8"), image)
		thresh = cv2.threshold(delta, tVal, 255, cv2.THRESH_BINARY)[1]
		# perform a series of erosions and dilations to remove small
		# blobs
		thresh = cv2.erode(thresh, None, iterations=2)
		thresh = cv2.dilate(thresh, None, iterations=2)
		# find contours in the thresholded image and initialize the
		# minimum and maximum bounding box regions for motion
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		(minX, minY) = (np.inf, np.inf)
		(maxX, maxY) = (-np.inf, -np.inf)
		# if no contours were found, return None
		if len(cnts) == 0:
# 			print "none((("
			return None 
		# otherwise, loop over the contours
		for c in cnts:
# 			print "find coutours"
			# compute the bounding box of the contour and use it to
			# update the minimum and maximum bounding box regions
			(x, y, w, h) = cv2.boundingRect(c)
			(minX, minY) = (min(minX, x), min(minY, y))
			(maxX, maxY) = (max(maxX, x + w), max(maxY, y + h))
		# otherwise, return a tuple of the thresholded image along
		# with bounding box
		return (thresh, (minX, minY, maxX, maxY))


# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)
# initialize the video stream and allow the camera sensor to

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")
	
def initStream():
	global vs
	vs = VideoStream(streamURL).start()
	time.sleep(.2)
	
# record and send video vars
outputFrame = None
vs = None
streamURL =""
token = ''
chat_id= ''
bright=0
minBright=0
needInitVideo=True
writing=False
outputVideo=None
wname="/tmp/video.avi"
sname="/tmp/video.mp4"
isStopped=False
video_record_lag=27*2 # ~2 sec
min_bright_that_no_move=5 # min bright when stop record video

def updateMaxBright(b):
	global bright  
	if b>bright:
		print "new max bright {}".format(b)
		bright=b
def writeVideo(frame):
	global outputVideo,writing,needInitVideo,frame_count,fname,isStopped
	(h, w) = (None, None)
	(h, w) = frame.shape[:2]
	if needInitVideo:
		print "start write video"
		needInitVideo=False
		timestamp = datetime.datetime.now()
		fourcc=cv2.cv.CV_FOURCC(*'DIVX')	
		outputVideo = cv2.VideoWriter(wname, fourcc, 5, (w, h ),True)

	outputVideo.write(frame)
	writing=True
	isStopped=False

def stopWriteVideo():
	global outputVideo,writing,needInitVideo,frame_count,wname,sname,isStopped,minBright,bright
	if writing==False or isStopped==True: return
	outputVideo.release()
	needInitVideo=True
	writing=False 
	if bright>=minBright: 
		print "SEND min bright from config {} cur bright {}".format(minBright,bright)
		try:
			cmd='ffmpeg -i '+wname+' -c:v copy -c:a copy -y '+sname
			os.system(cmd)  
			files = {'video': open(sname,'rb')}
			url='https://api.telegram.org/bot'+token+'/sendVideo?chat_id='+chat_id
			requests.post(url, files=files)
			os.remove(sname)		
		except:
			print "send file"	
    		
	else:
		print "do NOT send because min bright from config {} cur bright {}".format(minBright,bright)
	
	try:
		os.remove(wname)
		isStopped=True
		bright=0
	except:
	    print "err remove file"
	
def detect_motion(frameCount):
	# grab global references to the video stream, output frame, and
	# lock variables
	global vs, outputFrame, lock,frame_count
	# initialize the motion detector and the total number of frames
	# read thus far
	md = SingleMotionDetector(accumWeight=0.1)
	# total = 0
	novideo_count=0
	# loop over frames from the video stream
	move=False
	while True:
		# read the next frame from the video stream, resize it,
		# convert the frame to grayscale, and blur it
		frame = vs.read()
		if frame is None:
			print "empty frame, skip"
			initStream()
			continue

# 		cv2.imwrite("test.jpg", frame)
		frame = imutils.resize(frame, width=500)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (7, 7), 0)
		# grab the current timestamp and draw it on the frame
# 		timestamp = datetime.datetime.now()
# 		cv2.putText(frame, timestamp.strftime(
# 			"%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
# 			cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
		bright=np.average(frame)
		updateMaxBright(bright)
		cv2.putText(frame,"{}".format(bright) , (10, frame.shape[0] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
			
		# if the total number of frames has reached a sufficient
		# number to construct a reasonable background model, then
		# continue to process the frame
# 		writeVideo(frame)
		motion = md.detect(gray)
# 		if total > frameCount:
		# detect motion in the image
		# check to see if motion was found in the frame
		if motion is not None:
			# unpack the tuple and draw the box surrounding the
			# "motion area" on the output frame
			novideo_count=0
			move=True
		else: 
			novideo_count=novideo_count+1

		md.update(gray)
		
		if motion is not None:
			(thresh, (minX, minY, maxX, maxY)) = motion
			cv2.rectangle(frame, (minX, minY), (maxX, maxY),
						  (0, 0, 255), 2)
			
		if move and bright>min_bright_that_no_move:
			writeVideo(frame)
		
		if bright<=min_bright_that_no_move and move:
			novideo_count=video_record_lag #if we have a movement but bright is lower than light is turn off - stop record and sends
		
		if novideo_count>=video_record_lag:
			move=False
			stopWriteVideo()	
		
			
		# update the background model and increment the total number
		# of frames read thus far
		# total += 1
		# acquire the lock, set the output frame, and release the
		# lock
		with lock:
			outputFrame = frame.copy()
			
def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock
	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue
			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
			# ensure the frame was successfully encoded
			if not flag:
				continue
# 			cv2.imwrite("test.jpg", outputFrame)
			# yield the output frame in the byte format
			yield str(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
				bytearray(encodedImage) + b'\r\n')
			
@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")
		
		
# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-f", "--frame-count", type=int, default=10,
		help="# of frames used to construct the background model")
	ap.add_argument("-b", "--min-brightness", type=float, default=60,
		help="# min brightness for trigger record")
	ap.add_argument("-s", "--stream", type=str, default="0", required=True,
		help="# stream for watch")
	ap.add_argument("-tt", "--telegram-token", type=str, default="",required=True,
		help="# telegram token for send movement")	
	ap.add_argument("-tc", "--telegram-chat", type=str, default="",required=True,
		help="# telegram chat for send movement")			
	args = vars(ap.parse_args())
	
	token=args["telegram_token"]
	chat_id=args["telegram_chat"]
	streamURL=args["stream"]
	
	initStream()
	
	# start a thread that will perform motion detection
	t = threading.Thread(target=detect_motion, args=(
		args["frame_count"],))
	t.daemon = True
	t.start()
	minBright=args["min_brightness"]
	print "start with min bright {}".format(minBright)
	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=False,
		threaded=True, use_reloader=False)
# release the video stream pointer
vs.stop()











