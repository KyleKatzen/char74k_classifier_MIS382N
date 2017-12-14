import preprocessing, train
import tensorflow as tf 
import numpy as np 
import cv2
from tqdm import tqdm
import argparse

def predict(image, model_name):
	img_h, img_w = 64, 64

	nn = train.Classifier('classifier', img_w, img_h, len(preprocessing.CLASSES))

	with tf.Session() as sess:
		saver = tf.train.Saver()
		saver.restore(sess, model_name)

		image = preprocessing.open_image(image, (img_w, img_h))

		classes = sess.run(nn.classes, feed_dict = {nn.input : [image]})
		predicted_label = np.argmax(classes[0])

		print(predicted_label)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', type = str, required = True, help = "Image Path")
	parser.add_argument('-m', type = str, required = True, help = "Model Name")

	opt = parser.parse_args()

	predict(opt.i, opt.m)