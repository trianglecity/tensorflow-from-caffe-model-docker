
##
##  Mini Tensorflow from (VGG16) caffe-model on Docker
##


NOTICE 1: the source code is from 

		[1] (based on) https://github.com/DrSleep/tensorflow-deeplab-resnet.git

		[2] (prototxt and model  DeepLabV2(RESNET-101)) http://liangchiehchen.com/projects/DeepLabv2_resnet.html 

NOTICE 2: caffe to tensorflow is based on

		[1] https://github.com/ethereon/caffe-tensorflow/issues/114

		[2] https://stackoverflow.com/questions/38616620/caffe-to-tensorflow-kaffe-by-ethereon-typeerror-descriptors-should-not-be-c

		[3] https://github.com/ethereon/caffe-tensorflow/tree/master/examples/mnist


NOTICE 3: caffe models are from

		[1] https://github.com/BVLC/caffe/wiki/Model-Zoo#fully-convolutional-networks-for-semantic-segmentation-fcns

		[2] https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-vgg_ilsvrc_16_layers_deploy-prototxt

NOTICE 4: the usage and images are from

		[1] https://github.com/boyw165/tensorflow-vgg
##


[1] download (or git clone ) this source code folder.

[2] cd downloaded-source-code-folder

[3] sudo docker build -t tensorflow-resnet-caffemodel-dev:01 .

	wait ... wait ... wait ..

[4] sudo docker run --rm --privileged  -i --name tf-resnet -v $PWD:/home/deeplearning  -t "tensorflow-resnet-caffemodel-dev:01"  bash

[] root@8c72ad9a0fff:# cd /home/deeplearning

[7]  root@8c72ad9a0fff:/home/deeplearning# git clone https://github.com/tensorflow/tensorflow.git --branch r1.1

[8]  root@8c72ad9a0fff:/home/deeplearning# cd tensorflow/

[9]  root@8c72ad9a0fff:/home/deeplearning/tensorflow# ./configure
	
	
	Please specify the location of python. [Default is /usr/bin/python]: 
	Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]: 
	Do you wish to use jemalloc as the malloc implementation? [Y/n] y
	jemalloc enabled
	Do you wish to build TensorFlow with Google Cloud Platform support? [y/N] n
	No Google Cloud Platform support will be enabled for TensorFlow
	Do you wish to build TensorFlow with Hadoop File System support? [y/N] n
	No Hadoop File System support will be enabled for TensorFlow
	Do you wish to build TensorFlow with the XLA just-in-time compiler (experimental)? [y/N] n
	No XLA JIT support will be enabled for TensorFlow
	Do you wish to build TensorFlow with VERBS support? [y/N] y
	VERBS support will be enabled for TensorFlow
	Found possible Python library paths:
	  /usr/local/lib/python2.7/dist-packages
	  /usr/lib/python2.7/dist-packages
	Please input the desired Python library path to use.  Default is [/usr/local/lib/python2.7/dist-packages]
	
	Using python library path: /usr/local/lib/python2.7/dist-packages
	Do you wish to build TensorFlow with OpenCL support? [y/N] n
	No OpenCL support will be enabled for TensorFlow
	Do you wish to build TensorFlow with CUDA support? [y/N] n
	No CUDA support will be enabled for TensorFlow
	Extracting Bazel installation...
	...........
	INFO: Starting clean (this may take a while). Consider using --async if the clean takes more than several minutes.
	Configuration finished
	
	
[10]  root@8c72ad9a0fff:/home/deeplearning/tensorflow# gcc -v

	
	gcc version 5.4.0 20160609 (Ubuntu 5.4.0-6ubuntu1~16.04.4) 
	

[11]  root@8c72ad9a0fff:/home/deeplearning/tensorflow# bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package 

	Target //tensorflow/tools/pip_package:build_pip_package up-to-date:
  	bazel-bin/tensorflow/tools/pip_package/build_pip_package
	INFO: Elapsed time: 433.238s, Critical Path: 432.46s

[12] root@8c72ad9a0fff:/home/deeplearning/tensorflow# bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

	Output wheel file is in: /tmp/tensorflow_pkg


[13] root@8c72ad9a0fff:/home/deeplearning/tensorflow# cd /tmp/tensorflow_pkg/
[14] root@8c72ad9a0fff:/tmp/tensorflow_pkg# ls (copy the .whl file name)

		tensorflow-1.1.0-cp27-cp27mu-linux_x86_64.whl

[15] root@8c72ad9a0fff:/tmp/tensorflow_pkg# pip install ./tensorflow-1.1.0-cp27-cp27mu-linux_x86_64.whl 

[16] root@8c72ad9a0fff:/tmp/tensorflow_pkg# cd /home/deeplearning/


##

[17] root@8c72ad9a0fff:/home/deeplearning# cd caffe

[18] root@8c72ad9a0fff:/home/deeplearning/caffe# make

[19] root@8c72ad9a0fff:/home/deeplearning/caffe# cd /build/tools

[20] root@8c72ad9a0fff:/home/deeplearning/caffe/build/tools# ls

	upgrade_net_proto_text.bin


[21] root@8c72ad9a0fff:/home/deeplearning/caffe/build/tools# cd /home/deeplearning/

##


[22] root@8c72ad9a0fff:/home/deeplearning# cd /caffe-tensorflow/kaffe/caffe

[23] root@8c72ad9a0fff:/home/deeplearning/caffe-tensorflow/kaffe/caffe# ls

	__init__.py  __init__.pyc  caffepb.py  caffepb.pyc  resolver.py  resolver.pyc


[24] root@8c72ad9a0fff:/home/deeplearning/caffe-tensorflow/kaffe/caffe# rm *.pyc

[25] root@8c72ad9a0fff:/home/deeplearning/caffe-tensorflow/kaffe/caffe# mv caffepb.py caffe_pb2.py

[26] root@8c72ad9a0fff:/home/deeplearning/caffe-tensorflow/kaffe/caffe# vim resolver.py

		       			
            	from . import caffepb
            	self.caffepb = caffepb

		to

		from . import caffe_pb2
            	self.caffepb = caffe_pb2

##

[27] root@8c72ad9a0fff:/home/deeplearning/caffe-tensorflow/kaffe/caffe# cd ../../../caffe_2_tensorflow


[28] root@8c72ad9a0fff:/home/deeplearning/caffe_2_tensorflow# ../caffe/build/tools/upgrade_net_proto_text vgg_ilsvrc_16_layers_deploy.prototxt  vgg16.prototxt
	
	
	 upgrade_proto.cpp:53] Attempting to upgrade input file specified using deprecated V1LayerParameter: vgg_ilsvrc_16_layers_deploy.prototxt
	 upgrade_proto.cpp:61] Successfully upgraded file specified using deprecated V1LayerParameter
	 upgrade_proto.cpp:67] Attempting to upgrade input file specified using deprecated input fields: vgg_ilsvrc_16_layers_deploy.prototxt
	 upgrade_proto.cpp:70] Successfully upgraded file specified using deprecated input fields.
	 upgrade_proto.cpp:72] Note that future Caffe releases will only support input layers and not input fields.
	 upgrade_net_proto_text.cpp:49] Wrote upgraded NetParameter text proto to vgg16.prototxt


[29] root@8c72ad9a0fff:/home/deeplearning/caffe_2_tensorflow# ../caffe-tensorflow/convert.py  vgg16.prototxt  --code-output-path=vgg16.py
	
	
	------------------------------------------------------------
	    WARNING: PyCaffe not found!
	    Falling back to a pure protocol buffer implementation.
	    * Conversions will be drastically slower.
	    * This backend is UNTESTED!
	------------------------------------------------------------
	
	Type                 Name                                          Param               Output
	----------------------------------------------------------------------------------------------
	Input                input                                            --    (10, 3, 224, 224)
	Convolution          conv1_1                                          --   (10, 64, 224, 224)
	Convolution          conv1_2                                          --   (10, 64, 224, 224)
	Pooling              pool1                                            --   (10, 64, 112, 112)
	Convolution          conv2_1                                          --  (10, 128, 112, 112)
	Convolution          conv2_2                                          --  (10, 128, 112, 112)
	Pooling              pool2                                            --    (10, 128, 56, 56)
	Convolution          conv3_1                                          --    (10, 256, 56, 56)
	Convolution          conv3_2                                          --    (10, 256, 56, 56)
	Convolution          conv3_3                                          --    (10, 256, 56, 56)
	Pooling              pool3                                            --    (10, 256, 28, 28)
	Convolution          conv4_1                                          --    (10, 512, 28, 28)
	Convolution          conv4_2                                          --    (10, 512, 28, 28)
	Convolution          conv4_3                                          --    (10, 512, 28, 28)
	Pooling              pool4                                            --    (10, 512, 14, 14)
	Convolution          conv5_1                                          --    (10, 512, 14, 14)
	Convolution          conv5_2                                          --    (10, 512, 14, 14)
	Convolution          conv5_3                                          --    (10, 512, 14, 14)
	Pooling              pool5                                            --      (10, 512, 7, 7)
	InnerProduct         fc6                                              --     (10, 4096, 1, 1)
	InnerProduct         fc7                                              --     (10, 4096, 1, 1)
	InnerProduct         fc8                                              --     (10, 1000, 1, 1)
	Softmax              prob                                             --     (10, 1000, 1, 1)
	Converting data...
	Saving source...
	Done.


[30] root@8c72ad9a0fff:/home/deeplearning/caffe_2_tensorflow# cat vgg16.py

	from kaffe.tensorflow import Network

	class VGG_ILSVRC_16_layers(Network):
    		def setup(self):
        		(self.feed('input')
             			.conv(3, 3, 64, 1, 1, name='conv1_1')
             			.conv(3, 3, 64, 1, 1, name='conv1_2')
             			.max_pool(2, 2, 2, 2, name='pool1')
             			.conv(3, 3, 128, 1, 1, name='conv2_1')
             			.conv(3, 3, 128, 1, 1, name='conv2_2')
             			.max_pool(2, 2, 2, 2, name='pool2')
             			.conv(3, 3, 256, 1, 1, name='conv3_1')
             			.conv(3, 3, 256, 1, 1, name='conv3_2')
             			.conv(3, 3, 256, 1, 1, name='conv3_3')
             			.max_pool(2, 2, 2, 2, name='pool3')
             			.conv(3, 3, 512, 1, 1, name='conv4_1')
             			.conv(3, 3, 512, 1, 1, name='conv4_2')
             			.conv(3, 3, 512, 1, 1, name='conv4_3')
             			.max_pool(2, 2, 2, 2, name='pool4')
             			.conv(3, 3, 512, 1, 1, name='conv5_1')
             			.conv(3, 3, 512, 1, 1, name='conv5_2')
             			.conv(3, 3, 512, 1, 1, name='conv5_3')
             			.max_pool(2, 2, 2, 2, name='pool5')
             			.fc(4096, name='fc6')
             			.fc(4096, name='fc7')
             			.fc(1000, relu=False, name='fc8')
             			.softmax(name='prob'))
	

[31] root@8c72ad9a0fff:/home/deeplearning/caffe_2_tensorflow# wget http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel	

[32] root@aae1bdb022b4:/home/deeplearning/caffe_2_tensorflow# ../caffe-tensorflow/convert.py   vgg16.prototxt --caffemodel VGG_ILSVRC_16_layers.caffemodel --data-output-path=vgg16.npy
	
	
	
	------------------------------------------------------------
	    WARNING: PyCaffe not found!
	    Falling back to a pure protocol buffer implementation.
	    * Conversions will be drastically slower.
	    * This backend is UNTESTED!
	------------------------------------------------------------
	
	Type                 Name                                          Param               Output
	----------------------------------------------------------------------------------------------
	Input                input                                            --    (10, 3, 224, 224)
	Convolution          conv1_1                               (64, 3, 3, 3)   (10, 64, 224, 224)
	Convolution          conv1_2                              (64, 64, 3, 3)   (10, 64, 224, 224)
	Pooling              pool1                                            --   (10, 64, 112, 112)
	Convolution          conv2_1                             (128, 64, 3, 3)  (10, 128, 112, 112)
	Convolution          conv2_2                            (128, 128, 3, 3)  (10, 128, 112, 112)
	Pooling              pool2                                            --    (10, 128, 56, 56)
	Convolution          conv3_1                            (256, 128, 3, 3)    (10, 256, 56, 56)
	Convolution          conv3_2                            (256, 256, 3, 3)    (10, 256, 56, 56)
	Convolution          conv3_3                            (256, 256, 3, 3)    (10, 256, 56, 56)
	Pooling              pool3                                            --    (10, 256, 28, 28)
	Convolution          conv4_1                            (512, 256, 3, 3)    (10, 512, 28, 28)
	Convolution          conv4_2                            (512, 512, 3, 3)    (10, 512, 28, 28)
	Convolution          conv4_3                            (512, 512, 3, 3)    (10, 512, 28, 28)
	Pooling              pool4                                            --    (10, 512, 14, 14)
	Convolution          conv5_1                            (512, 512, 3, 3)    (10, 512, 14, 14)
	Convolution          conv5_2                            (512, 512, 3, 3)    (10, 512, 14, 14)
	Convolution          conv5_3                            (512, 512, 3, 3)    (10, 512, 14, 14)
	Pooling              pool5                                            --      (10, 512, 7, 7)
	InnerProduct         fc6                                   (4096, 25088)     (10, 4096, 1, 1)
	InnerProduct         fc7                                    (4096, 4096)     (10, 4096, 1, 1)
	InnerProduct         fc8                                    (1000, 4096)     (10, 1000, 1, 1)
	Softmax              prob                                             --     (10, 1000, 1, 1)
	Converting data...
	Saving data...
	Done.
	
	
	
[33] root@aae1bdb022b4:/home/deeplearning/caffe_2_tensorflow# ls

	VGG_ILSVRC_16_layers.caffemodel  vgg16.npy  vgg16.prototxt  vgg16.py  vgg_ilsvrc_16_layers_deploy.prototxt


[34] root@aae1bdb022b4:/home/deeplearning/caffe_2_tensorflow# cp vgg16.npy  ../example
[35] root@aae1bdb022b4:/home/deeplearning/caffe_2_tensorflow# cp vgg16.py  ../example

[36] root@aae1bdb022b4:/home/deeplearning/caffe_2_tensorflow# cd ../example

[37] root@aae1bdb022b4:/home/deeplearning/example# cp *.* ../caffe-tensorflow/

[38] root@aae1bdb022b4:/home/deeplearning/example# cd ../caffe-tensorflow

[39] root@aae1bdb022b4:/home/deeplearning/caffe-tensorflow# python ./vgg16_usage.py 

	
	[895 908 744 404]
	warplane, military plane 0.912189
	wing 0.0273883
	projectile, missile 0.0215362
	airliner 0.0173264


[40] the source code looks like this
	
	
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


	
[41 cleanup] root@aae1bdb022b4:/home/deeplearning/caffe-tensorflow# rm vgg16.npy
[42 cleanup] root@aae1bdb022b4:/home/deeplearning/caffe_2_tensorflow# rm  ./VGG_ILSVRC_16_layers.caffemodel
[43 cleanup] root@aae1bdb022b4:/home/deeplearning/caffe_2_tensorflow# rm vgg16.npy

