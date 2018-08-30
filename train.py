import keras
import Models , LoadBatches
from keras.utils import to_categorical
import os
import cv2
import numpy as np
def load_data(img_files,bg,ed):
    def process_tag(label_img, label_tag):
        for i,num in enumerate(label_tag):
            label_img[label_img==num] = i
        return np.array(label_img, np.float32)
    data = []
    label = []
    label_tag = [0,3,5,6,50,255]
    u = 0
    for img_file in img_files[bg:ed]:
        if img_file.split('.')[-1] == 'tif':
            img = cv2.imread('train/images/'+img_file)            
            label_img = cv2.imread('train/labels/'+'02-'+img_file.split('-')[-1],0)
            if sum(label_img.shape) != 1024:
                continue
            label_img = cv2.resize(label_img, (output_height, output_width),interpolation=cv2.INTER_NEAREST)
            label_img = np.reshape(label_img,(-1,1))
            label_img = process_tag(label_img,label_tag)
            label.append(label_img)
            data.append(cv2.resize(img,(256,256)))
            u+=1
    return np.array(data, np.float),np.array(label)



if __name__ == "__main__":

	batch_size = 8
	n_classes = 6
	input_height = 256
	input_width = 256
	validate = False
	epochs = 10


	optimizer_name = 'adadelta'
	model_name = 'vgg_segnet'


	print("creating model...")
	modelFns = { 'vgg_segnet':Models.VGGSegnet.VGGSegnet , 'vgg_unet':Models.VGGUnet.VGGUnet , 'vgg_unet2':Models.VGGUnet.VGGUnet2 , 'fcn8':Models.FCN8.FCN8 , 'fcn32':Models.FCN32.FCN32   }
	modelFN = modelFns[ model_name ]

	m = modelFN( n_classes , input_height=input_height, input_width=input_width)
	m.compile(loss='categorical_crossentropy', optimizer= optimizer_name, metrics=['accuracy'])
	output_height = m.outputHeight
	output_width = m.outputWidth
	
	print('loading data...')
	files = os.listdir('train/images/')
	img_files = []
	for file in files:
		if file.split('.')[-1] == 'tif':
			img_files.append(file)
	data_size = len(img_files)        
	i = 0
	data, label = load_data(img_files,0,data_size)
	label = np.array(label,np.int)
	label = to_categorical(label, num_classes=n_classes) 	
	import pdb
	pdb.set_trace()
	try:
		m.fit(data, label, batch_size=batch_size, epochs=epochs, class_weight="auto")
	finally:
		m.save_weights("model_"+model_name+'.h5')



	# G  = LoadBatches.imageSegmentationGenerator( train_images_path , train_segs_path ,  train_batch_size,  n_classes , input_height , input_width , output_height , output_width)

	# m.fit_generator( G , 512  , epochs=1 )

	# if validate:
	# 	G2  = LoadBatches.imageSegmentationGenerator( val_images_path , val_segs_path ,  val_batch_size,  n_classes , input_height , input_width , output_height , output_width   )

	# if not validate:
	# 	for ep in range( epochs ):
	# 		
	# 		m.save_weights( save_weights_path + "." + str( ep ) )
	# 		m.save( save_weights_path + ".model." + str( ep ) )
	# else:
	# 	for ep in range( epochs ):
	# 		m.fit_generator( G , 512  , validation_data=G2 , validation_steps=200 ,  epochs=1 )
	# 		m.save_weights( save_weights_path + "." + str( ep )  )
	# 		m.save( save_weights_path + ".model." + str( ep ) )


