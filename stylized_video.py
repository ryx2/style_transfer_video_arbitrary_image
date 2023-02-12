import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import time

# Load content and style images (see example in the attached colab).

import cv2
cap = cv2.VideoCapture('/Volumes/antivrs.xlx/projects/2022_12_05 money means/money means.mov')
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
for i in range(1500):
	_, _ = cap.read()
	print(f'read frame {i}')

# read all style images

# determine image moods

# adjust style image over time; zigzag thru this area

# take care of the style image piece first
style_image = plt.imread('/Volumes/antivrs.xlx/projects/2023_01_19 dont tell me/AI music video pics for style transfer/epic_wave.jpeg')
style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.
style_image = style_image[:, 500:(500+256), 600:(500+256), :]
# style_image = style_image[:, 500:(500+128), 500:(500+128), :]
style_scaler = 1
style_image = tf.image.resize(style_image, (256 * style_scaler, 256 * style_scaler))
# Load image stylization module.
hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# now the content im
for i in range(1):
# content_image = plt.imread('/Users/raymondxu/Desktop/money_means_thumb.jpg')
	ret, frame = cap.read()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	plt.imshow(frame)
	plt.show()
	# style_image = plt.imread('/Volumes/antivrs.xlx/projects/2023_01_19 dont tell me/AI music video pics for style transfer/fantastic-epic-magical-forest-landscape-summer-beautiful-mystic-nature-gaming-assets-celtic_636456-3123.jpeg')
	# Convert to float32 numpy array, add batch dimension, and normalize to range [0, 1]. Example using numpy:
	content_image = np.array(frame).astype(np.float32)[np.newaxis, ...] / 255.
	# Optionally resize the images. It is recommended that the style image is about
	# 256 pixels (this size was used when training the style transfer network).
	# The content image can be any size.
	# print(style_image.shape)
	# style_image_temp = tf.image.resize(style_image, (256, 256))
	# print(style_image_temp.shape)
	content_scaler = 0.5
	content_image = tf.image.resize(content_image, (int(1080 * content_scaler), int(1920 * content_scaler)))

	# Stylize image.
	print(content_image.shape)
	start = time.time()
	outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
	print(outputs[0].shape)
	# outputs = tf.image.resize(outputs[0], (int(outputs[0].shape[1] / content_scaler), int(outputs[0].shape[2] / content_scaler)))
	print(outputs[0].shape)
	# outputs = hub_module(tf.constant(outputs), tf.constant(style_image))
	print(outputs[0].shape)
	print(f'took {time.time() - start}')
	stylized_image = outputs[0]



	plt.imshow(stylized_image[0])
	plt.show()


cap.release()