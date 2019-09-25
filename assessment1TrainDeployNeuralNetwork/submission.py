import caffe
import cv2
import sys

def deploy(img_path):

    caffe.set_mode_gpu()
    
    # Initialize the Caffe model using the model trained in DIGITS. Which two files constitute your trained model?
    architecture = '/dli/data/digits/20190922-225346-d93f' + '/' + 'deploy.prototxt'
    weights = '/dli/data/digits/20190922-225346-d93f' + '/' + 'snapshot_iter_270.caffemodel'
    net = caffe.Classifier(architecture, weights,
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256, 256))
                       
    # Create an input that the network expects. This is different for each project, so don't worry about the exact steps, but find the dataset job directory to show you know that whatever preprocessing is done during training must also be done during deployment.
    input_image= caffe.io.load_image(img_path)
    input_image = cv2.resize(input_image, (256,256))
    mean_image = caffe.io.load_image('/dli/data/digits/20190922-225123-167e/mean.jpg')
    input_image = input_image-mean_image

    # Make prediction. What is the function and the input to the function needed to make a prediction?
    prediction = net.predict([input_image])

    # Create an output that is useful to a user. What is the condition that should return "whale" vs. "not whale"?
    if prediction.argmax()==0:
        return "whale"
    else:
        return "not whale"

    
##Ignore this part    
if __name__ == '__main__':
    print(deploy(sys.argv[1]))

