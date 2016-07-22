
# coding: utf-8

# # Instant Recognition with Caffe
# 
# In this example we'll classify an image with the bundled CaffeNet model based on the network architecture of Krizhevsky et al. for ImageNet. We'll compare CPU and GPU operation then reach into the model to inspect features and the output.
#
# First, import required modules, set plotting parameters, and run `./scripts/download_model_binary.py models/bvlc_reference_caffenet` to get the pretrained CaffeNet model if it hasn't already been fetched.

# In[2]:




# In[ ]:

import numpy as np
import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')

# Make sure that caffe is on the python path:
caffe_root = '/home/dz/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

#plt.rcParams['figure.figsize'] = (10, 10)
#plt.rcParams['image.interpolation'] = 'nearest'
#plt.rcParams['image.cmap'] = 'gray'

import os
# Set Caffe to CPU mode, load the net in the test phase for inference, and configure input preprocessing.

# In[3]:

caffe.set_device(0)
caffe.set_mode_gpu()
#caffe.set_mode_cpu()

net = caffe.Net('/home/dz/raintest/raintest_caffenet/deploy-3conv.prototxt',
                '/home/dz/raintest/raintest_caffenet/new2_iter_40000.caffemodel',
                caffe.TEST)
#net = caffe.Net('/home/dz/raintest/raintest_caffenet/deploy1.prototxt',
#                '/home/dz/raintest/raintest_caffenet/finetune_iter_18054.caffemodel',
#                caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load('/home/dz/raintest/test/out-14-6.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB



# Let's start with a simple classification. We'll set a batch of 50 to demonstrate batch processing, even though we'll only be classifying one image. (Note that the batch size can also be changed on-the-fly.)

# In[5]:

# set net to batch size of 50
net.blobs['data'].reshape(1,3,227,227)
f = open('prediction.csv','w')
f1 = open('pre-label.csv','w')
    # Feed in the image (with some preprocessing) and classify with a forward pass.

# In[6]:

#from os import listdir
#from os.path import isfile, join
# image = '/home/dz/raintest/test/GC/'
#onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for i in range(0,3000):
    image = '/home/dz/raintest/test/GC/'
#    j=str(i).rjust(4,'0')
    net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(image + str(i).rjust(4,'0') + '.jpg'))
    out = net.forward()
    a = out['prob']
    b = a[0]
#    f.write(str(b[0])+'\t')
#    f.write(str(b[1])+'\t')
#    f.write(format(out['prob'].argmax())+'\n')    
#    f.write(format(out['prob'])+'\n')
    f1.write(str(i).rjust(4,'0')+'.jpg,'+format(out['prob'].argmax())+'\n')
#    print("Predicted class is #{}.".format(out['prob']))

f.close()
f1.close()




# sort top k predictions from softmax output
#top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
#print labels[top_k]


