
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/models/fcn32s.py
# fc weights into the 1x1 convs  , get_upsampling_weight 



from keras.models import *
from keras.layers import *


import os
file_path = os.path.dirname( os.path.abspath(__file__) )

VGG_Weights_path = file_path+"/../data/vgg16_weights_th_dim_ordering_th_kernels.h5"

IMAGE_ORDERING = 'channels_last'


def FCN32( n_classes ,  input_height=416, input_width=608 , vgg_level=3):

	assert input_height%32 == 0
	assert input_width%32 == 0

	# https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5
	img_input = Input(shape=(input_height,input_width,3))

	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1' )(img_input)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2' )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool' )(x)
	f1 = x
	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1' )(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2' )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool' )(x)
	f2 = x

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1' )(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2' )(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3' )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool' )(x)
	f3 = x

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1' )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2' )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3' )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool' )(x)
	f4 = x

	# Block 5
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1' )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2' )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3' )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool' )(x)
	f5 = x

	x = Flatten(name='flatten')(x)
	x = Dense(4096, activation='relu', name='fc1')(x)
	x = Dense(4096, activation='relu', name='fc2')(x)
	x = Dense( 1000 , activation='softmax', name='predictions')(x)

	vgg  = Model(  img_input , x  )
	# vgg.load_weights(VGG_Weights_path)

	o = f5

	o = ( Conv2D( 4096 , ( 7 , 7 ) , activation='relu' , padding='same'))(o)
	o = Dropout(0.5)(o)
	o = ( Conv2D( 4096 , ( 1 , 1 ) , activation='relu' , padding='same'))(o)
	o = Dropout(0.5)(o)

	o = ( Conv2D( n_classes ,  ( 1 , 1 ) ,kernel_initializer='he_normal' ))(o)
	o = Conv2DTranspose( n_classes , kernel_size=(64,64) ,  strides=(32,32) , use_bias=False)(o)
	o_shape = Model(img_input , o ).output_shape
	
	outputHeight = o_shape[1]
	outputWidth = o_shape[2]


	o = (Reshape(( -1, outputHeight*outputWidth)))(o)
	o = (Permute((2, 1)))(o)
	o = (Activation('softmax'))(o)
	model = Model( img_input , o )
	model.outputWidth = outputWidth
	model.outputHeight = outputHeight

	return model


if __name__ == '__main__':
	m = FCN32( 101 )
	from keras.utils import plot_model
	plot_model( m , show_shapes=True , to_file='model.png')
