import numpy as np
import cv2
import time 
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras import backend
from keras.models import Model
from scipy.optimize import fmin_l_bfgs_b
from matplotlib import pyplot as plt 
from pprint import pprint


config = tf.ConfigProto(device_count={'GPU': 1,'CPU':4})
sess=tf.Session(config=config)
backend.set_session(sess)


def show_image(data):
    data= np.asarray(data,dtype='uint8')
    plt.imshow(data,interpolation='nearest')
    plt.show()

iterations=10
content_weight=0.1
style_weight=1000
total_variation_weight=1.0
total_variation_loss_factor=1
imagenet_mean_rgb_value=[123.68,116.779,103.939]

content_image='/home/ananya/Documents/ananya/vision/content_image.jpeg'
style_image='/home/ananya/Documents/ananya/vision/style_image.jpg'
content_image=cv2.imread(content_image)
content_image=cv2.cvtColor(content_image,cv2.COLOR_BGR2RGB)



style_image=cv2.imread(style_image)
style_image=cv2.cvtColor(style_image,cv2.COLOR_BGR2RGB)


content_image = cv2.resize(content_image, (512, 512))
style_image = cv2.resize(style_image, (512, 512))

content_array=np.asarray(content_image,dtype='float32')
content_array=np.expand_dims(content_array,axis=0)
content_array[:,:,:,0]-=imagenet_mean_rgb_value[2]
content_array[:,:,:,1]-=imagenet_mean_rgb_value[1]
content_array[:,:,:,2]-=imagenet_mean_rgb_value[0]
content_array=content_array[:,:,:,::-1]


style_array=np.asarray(style_image,dtype='float32')
style_array=np.expand_dims(style_array,axis=0)
style_array[:,:,:,0]-=imagenet_mean_rgb_value[2]
style_array[:,:,:,1]-=imagenet_mean_rgb_value[1]
style_array[:,:,:,2]-=imagenet_mean_rgb_value[0]
style_array=style_array[:,:,:,::-1]


#model vgg16
content_image=backend.variable(content_array)
style_image=backend.variable(style_array)
output_image=backend.placeholder((1,512,512,3))

# taking content_image , style_image , output_image as our input 

input_tensor=backend.concatenate([content_image,style_image,output_image],axis=0)

model=VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)




layers=dict([layer.name,layer.output] for layer in model.layers)
 
loss=backend.variable(0.)

def content_loss(content,combination):
    return backend.sum(backend.square(content-combination))


layer_features=layers['block2_conv2']
content_image_features=layer_features[0,:,:,:]
output_image_features=layer_features[2,:,:,:]

loss+=content_weight*content_loss(content_image_features,output_image_features)

def gram_matrix(x):
    features=backend.batch_flatten(backend.permute_dimensions(x,(2,0,1)))
    gram=backend.dot(features, backend.transpose(features))
    return gram

def style_loss(style, combinations):
    s=gram_matrix(style)
    c=gram_matrix(combinations)
    channel=3
    size= 100*100
    st= backend.sum(backend.square(s-c)/4*(channel**2)*(size**2))
    return st




feature_layers= ['block1_conv2' , 'block2_conv2' , 'block3_conv3' , 'block4_conv3' , 'block5_conv3']


for l_n in feature_layers:
    layer_fatures= layers[l_n]
    style_feature=layer_features[1,:,:,:]
    output_feature=layer_features[2,:,:,:]
    sl=style_loss(style_feature,output_feature)
    loss+=(style_weight/len(feature_layers))*sl

z=512

def total_variation_loss(x):
    a=backend.square(x[:,:z-1,:z-1,:]-x[:,1:,:z-1,:])
    b=backend.square(x[:,:z-1,:z-1,:]-x[:,:z-1,1:,:])
    return backend.sum(backend.pow(a+b,1))


loss+=total_variation_weight*total_variation_loss(output_image)

outputs= [loss]
outputs+=backend.gradients(loss, output_image)



def eval_loss_and_grads(x):
    x=x.reshape((1,512,512,3))
    o=backend.function([output_image],outputs)([x])
    loss_value=o[0]
    grad_value=o[1].flatten().astype('float64')
    return loss_value, grad_value


class Evaluator:
     
      def loss(self, x):
          loss,gradients = eval_loss_and_grads(x)
          self._gradients=gradients
          return loss 
      def gradients(self, x):
          return self._gradients 


evaluator=Evaluator()


x=np.random.uniform(0,255,(1,512,512,3))-128.0




for i in range(100):
    print('start of iteration', i  )
    start_time=time.time()
    x, min_val ,info=fmin_l_bfgs_b(evaluator.loss, x.flatten(),fprime=evaluator.gradients, maxfun=20)
    end_time = time.time()
    print(end_time-start_time)
    if i==0 or (i+1) % 5 ==0 :
       y= np.copy(x).reshape(512,512,3)
       y=y[:,:,::-1]
       y[:,:,0]+=imagenet_mean_rgb_value[2]
       y[:,:,1]+=imagenet_mean_rgb_value[1]
       y[:,:,2]+=imagenet_mean_rgb_value[0]
       y=np.clip(y,0,255).astype("uint8")
       output_image1=y
     
    


cv2.imshow('image',output_image1)
cv2.waitKey(0)
cv2.destroyAllWindows()
 
               









