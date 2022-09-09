# some pacages needs to be installed 

!pip install face_recognition
pip install FER
!pip install tensorflow-gpu # for the Tensorflow GPU 
!pip install tflearn    #for the tensorflow learn
#!pip install cv2
#!pip install imutils
#!import os

# Possible imports PLeasee check with functions as well

import imutils
import numpy as np
import cv2
#import face_recognition
import os
from google.colab.patches import cv2_imshow
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
import matplotlib.pyplot as plt 
%matplotlib inline
from fer import FER


#load training dataset
from google.colab import files
files.upload()

#extract to folder
from zipfile import ZipFile
file_path = "TrainingKnonw.zip"

with ZipFile(file_path, 'r') as zip:
  zip.extractall()
  print("Done")

# Load the Folders known and UNknown

#Known = Storage and application sheet. This will be the Photos and Names of the user
# We ill just load an existing known folders. Seach folder will contains photos ad a names 

#this will just be photos with names and no names 

#We can load the Image from a source

def loadImage(filename='photo.jpg', quality=0.8):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
  display(js)
  data = eval_js('takePhoto({})'.format(quality))
  binary = b64decode(data.split(',')[1])
  with open(filename, 'wb') as f:
    f.write(binary)
  return filename			# this is a path to the image
	
		
image2path = loadImage()  # we will save this path so we can use it again



# now we will do the testing
# we opted to use a function that we can call at any moment

def IdentifyFace(image2path):
    import face_recognition
    import cv2
    import os
    from google.colab.patches import cv2_imshow
        
	def read_img(image2path):
		img = cv2.imread(image2path)
		(h, w) = img.shape[:2]
		width = 500
		ratio = width / float(w)
		height = int(h * ratio)
		return cv2.resize(img, (width, height))


	known_encodings = []
	known_names = []
	known_dir = 'Training knonw'

	for file in os.listdir(known_dir):
		img = read_img(known_dir + '/' + file)
		img_enc = face_recognition.face_encodings(img)[0]
		known_encodings.append(img_enc)
		known_names.append(file.split('.')[0])

	unknown_dir = 'unknown'
	for file in os.listdir(unknown_dir):
		print("Processing", file)
		img = read_img(unknown_dir + '/' + file)
		img_enc = face_recognition.face_encodings(img)[0]

		results = face_recognition.compare_faces(known_encodings, img_enc)
		# print(face_recognition.face_distance(known_encodings, img_enc))

		for i in range(len(results)):
			if results[i]:
				name = known_names[i]
				(top, right, bottom, left) = face_recognition.face_locations(img)[0]
				cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
				cv2.putText(img, name, (left+2, bottom+20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
				cv2_imshow(img)
            
			#print(results)
	return(results) # we just want to collect the name

	
Person1Name = IdentifyFace(image2path)
print(Person1Name)


# Now we will Define the Emotion of the Person1Name

def EmotionFeel(image2path):
    test_image_one = plt.imread(image2path)
    emo_detector = FER(mtcnn=True)
    # Capture all the emotions on the image
    captured_emotions = emo_detector.detect_emotions(test_image_one)
    # Print all captured emotions with the image
    print(captured_emotions)
    plt.imshow(test_image_one)
    
    # Use the top Emotion() function to call for the dominant emotion in the image
    dominant_emotion, emotion_score = emo_detector.top_emotion(test_image_one)
    #print(dominant_emotion, emotion_score)
    retrun(dominant_emotion)

def EmotionInMotion():
    from fer import Video
    from fer import FER
    import os
    import sys
    import pandas as pd
    
    # Put in the location of the video file that has to be processed
    location_videofile = "/content/Video_One.mp4"
    
        # Build the Face detection detector
    face_detector = FER(mtcnn=True)
    # Input the video for processing
    input_video = Video(location_videofile)
    
    # The Analyze() function will run analysis on every frame of the input video. 
    # It will create a rectangular box around every image and show the emotion values next to that.
    # Finally, the method will publish a new video that will have a box around the face of the human with live emotion values.
    processing_data = input_video.analyze(face_detector, display=False)
    
    # We will now convert the analysed information into a dataframe.
    # This will help us import the data as a .CSV file to perform analysis over it later
    vid_df = input_video.to_pandas(processing_data)
    vid_df = input_video.get_first_face(vid_df)
    vid_df = input_video.get_emotions(vid_df)
    
    # Plotting the emotions against time in the video
    pltfig = vid_df.plot(figsize=(20, 8), fontsize=16).get_figure()
    
    # We will now work on the dataframe to extract which emotion was prominent in the video
    angry = sum(vid_df.angry)
    disgust = sum(vid_df.disgust)
    fear = sum(vid_df.fear)
    happy = sum(vid_df.happy)
    sad = sum(vid_df.sad)
    surprise = sum(vid_df.surprise)
    neutral = sum(vid_df.neutral)
    
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    emotions_values = [angry, disgust, fear, happy, sad, surprise, neutral]
    
    score_comparisons = pd.DataFrame(emotions, columns = ['Human Emotions'])
    score_comparisons['Emotion Value from the Video'] = emotions_values
    print(comparisons)
    