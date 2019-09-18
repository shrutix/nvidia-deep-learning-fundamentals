get_ipython().system(u'wget http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel')
get_ipython().system(u'wget https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt')

get_ipython().system(u'wget https://github.com/BVLC/caffe/blob/master/python/caffe/imagenet/ilsvrc_2012_mean.npy?raw=true')
get_ipython().system(u'mv ilsvrc_2012_mean.npy?raw=true ilsvrc_2012_mean.npy')

import caffe
import numpy as np
caffe.set_mode_gpu()
import matplotlib.pyplot as plt 

ARCHITECTURE = 'deploy.prototxt'
WEIGHTS = 'bvlc_alexnet.caffemodel'
MEAN_IMAGE = 'ilsvrc_2012_mean.npy'
TEST_IMAGE = '/dli/data/BeagleImages/louietest2.JPG'

net = caffe.Classifier(ARCHITECTURE, WEIGHTS) 

image= caffe.io.load_image(TEST_IMAGE)
plt.imshow(image)
plt.show()

mean_image = np.load(MEAN_IMAGE)
mu = mean_image.mean(1).mean(1) 

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  
transformer.set_mean('data', mu)            
transformer.set_raw_scale('data', 255)      
transformer.set_channel_swap('data', (2,1,0))  

net.blobs['data'].reshape(1,        
                          3,         
                          227, 227)  

transformed_image = transformer.preprocess('data', image)

net.blobs['data'].data[...] = transformed_image

output = net.forward()

output

output_prob = output['prob'][0]  
print 'predicted class is:', output_prob.argmax()

get_ipython().system(u'wget https://raw.githubusercontent.com/HoldenCaulfieldRye/caffe/master/data/ilsvrc12/synset_words.txt')
labels_file = 'synset_words.txt'
labels = np.loadtxt(labels_file, str, delimiter='\t')

print 'output label:', labels[output_prob.argmax()]

print ("Input image:")
plt.imshow(image)
plt.show()

print("Output label:" + labels[output_prob.argmax()])
