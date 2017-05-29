

import numpy as np
import tensorflow as tf
from scipy import ndimage
from vgg16 import VGG_ILSVRC_16_layers
from imagenet_classes import class_names


img1 = ndimage.imread("img-airplane-224x224.jpg", mode="RGB").astype(float)
img2 = ndimage.imread("img-dog-224x224.jpg", mode="RGB").astype(float)

batch1 = np.reshape(img1, (1, 224, 224, 3))
batch2 = np.reshape(img2, (1, 224, 224, 3))

images = np.concatenate((batch1, batch2), 0)

##batch1 = tf.convert_to_tensor(batch1, dtype=tf.float32)
##batch2 = tf.convert_to_tensor(batch2, dtype=tf.float32)

##batch = tf.concat([batch1, batch2], 0)

batch_size = 1
batch = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])

net = VGG_ILSVRC_16_layers({'input': batch})
prob = net.layers['prob']

with tf.Session() as sess:
    
    	sess.run( tf.global_variables_initializer() )
    	net.load('vgg16.npy', sess)
    
    	##feed = {batch: images}
	feed = {batch: batch1}

        prob_value = sess.run(prob, feed_dict=feed)[0]
        
	preds = (np.argsort(prob_value)[::-1])[0:4]
	print preds	
	for p in preds:
        		print class_names[p], prob_value[p]
	print ""
