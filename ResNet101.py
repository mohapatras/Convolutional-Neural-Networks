import sys
import warnings
import os

import tensorflow as tf
import numpy as np

print(tf.__version__)
print(sys.version)

def _image_input_shape(input_shape,
                       default_size,
                       min_size,
                       data_format,
                       require_flatten,
                       channels=1,
                       weights=None):
    
    
    if weights!= 'ipie_primary' and input_shape and len(input_shape) == 3:
        if data_format == 'channels_first':
            if input_shape[0] not in {1, 3}:
                warnings.warn(
                            'Image input channel should either be 1 or 3.'
                            'Now the data is passed with '+str(input_shape[0])+ 'input_channels.')
            default_shape = (input_shape[0], default_size, default_size)
        else:
            if input_shape[-1] not in {1, 3}:
                warnings.warn(
                            'Image input channel should either be 1 or 3.'
                            'Now the data is passed with '+str(input_shape[-1])+ 'input_channels.')
            deafult_shape = (deafult_size, default_size, default_shape[-1])
        
    else:
        if data_format == 'channels_first':
            default_shape = (channels, default_size, default_size)
        else:
            default_shape = (default_size, default_size, channels)
            
    if weights == 'ipie_primary' and require_flatten:
        if input_shape is not None:
            if input_shape!= default_shape:
                raise ValueError('When setting `include_top = True` '
                                 'and loading `ipie_primary` weights, '
                                 'input shape should be ' + str(default_shape)+'.')
            
        return default_shape
    
    if input_shape:
        if data_format == 'channels_first':
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError('`input_shape` must be a tuple of three integers.')
                
                if input_shape[0] != 1 and weights == 'ipie_primary': # 1 because IPIe images are grayscale.    
                    raise ValueError('The input must have 1 channel ; got `input_shape='+str(input_shape)+'`')
                if ((input_shape[1] is not None and input_shape[1] < min_size) or 
                    (input_shape[2] is not None and input_shape[2] < min_size)):
                    raise ValueError('Input size must be at least '+str(min_size)+'x'+str(min_size)+'; got'
                                     '`input_shape='+str(input_shape))+'`'
                    
        else:
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError('`input_shape` must be a tuple of three integers.')
                
                if input_shape[0] != 1 and weights == 'ipie_primary': # 1 because IPIe images are grayscale.    
                    raise ValueError('The input must have 1 channel ; got `input_shape='+str(input_shape)+'`')
                if ((input_shape[1] is not None and input_shape[1] < min_size) or 
                    (input_shape[2] is not None and input_shape[2] < min_size)):
                    raise ValueError('Input size must be at least '+str(min_size)+'x'+str(min_size)+'; got'
                                     '`input_shape='+str(input_shape))+'`'
                    
        
    else:
        if require_flatten:
            input_shape = default_shape
        
        else:
            if data_format=='channels_first':
                input_shape = (1, None, None) # Change 1 to 3 for making it comaptible with 3 channel images.
            else:
                input_shape = (None, None, 1) # Change 1 to 3 for making it comaptible with 3 channel images.
            
    if require_flatten:
        if None in input_shape:
            raise ValueError('If `include_top` is True, then specify a static input shape;'
                            'got `input_shape='+str(input_shape)+'`')
    
    
    return input_shape

class BatchNorm(tf.keras.layers.BatchNormalization):
	def call(self, inputs, training=None):
		return super(self.__class__,self).call(inputs, training=training)


                
def identity_block(input_tensor, kernel_size, filters, 
					stage, block, use_bias = True, train_bn=True):
	filter_1, filter_2, filter_3 = filters
	conv_name_base = 'res'+str(stage)+block+'_branch'
	bn_name_base = 'bn'+str(stage)+block+'_branch'

	'''if tf.keras.backend.image_data_format()=='channels_first':
					bn_axis = 1
				else:
					bn_axis = 3'''

	x = tf.keras.layers.Conv2D(filter_1, (1,1),name=conv_name_base+'2a', use_bias=use_bias)(input_tensor)
	x = BatchNorm( name=bn_name_base+'2a')(x, training=train_bn)
	x = tf.keras.layers.Activation('relu')(x)

	x = tf.keras.layers.Conv2D(filter_2,(kernel_size,kernel_size), padding='same', name=conv_name_base+'2b', use_bias=use_bias)(x)
	x = BatchNorm(name=bn_name_base+'2b')(x, training=train_bn)
	x = tf.keras.layers.Activation('relu')(x)

	x = tf.keras.layers.Conv2D(filter_3,(1,1), name=conv_name_base+'2c', use_bias=use_bias)(x)
	x = BatchNorm(name=bn_name_base+'2c')(x, training=train_bn)

	x = tf.keras.layers.add([x, input_tensor])
	x = tf.keras.layers.Activation('relu', name='res'+str(stage)+block+'_out')(x)

	return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2,2), use_bias=True,train_bn=True):
	filter_1, filter_2, filter_3 = filters
	conv_name_base = 'res'+str(stage)+block+'_branch'
	bn_name_base = 'bn'+str(stage)+block+'_branch'

	'''if tf.keras.backend.image_data_format() == 'channels_first':
					bn_axis = 1
				else:
					bn_axis = 3'''

	x = tf.keras.layers.Conv2D(filter_1, (1,1), name= conv_name_base+'2a', use_bias=use_bias)(input_tensor)
	x = BatchNorm(name=bn_name_base+'2a')(x,training=train_bn)
	x = tf.keras.layers.Activation('relu')(x)

	x = tf.keras.layers.Conv2D(filter_2, (kernel_size,kernel_size), name=conv_name_base+'2b', use_bias=use_bias)(x)
	x = BatchNorm(name=bn_name_base+'2b')(x,training=train_bn)
	x = tf.keras.layers.Activation('relu')(x)

	x = tf.keras.layers.Conv2D(filter_3, (1,1), name=conv_name_base+'2c', use_bias=use_bias)(x)
	x = BatchNorm(name=bn_name_base+'2c')(x, training=train_bn)

	shortcut = tf.keras.layers.Conv2D(filter_3, (1,1), strides = strides, name=conv_name_base+'1', use_bias=use_bias)(input_tensor)
	shortcut = BatchNorm(axis = bn_axis, name=bn_name_base+'2')(x, training=train_bn)

	x = tf.keras.layers.add([x,shortcut])
	x = tf.keras.layers.Activation('relu',name='res'+str(stage)+block+'_out')(x)

	return x

def ResNet101(include_top=True, weights=None, input_shape=None, pooling=None, classes=19, activation='sigmoid', train_bn=True, stage5=False):

	_image_input_shape(input_shape, default_size = 256, min_size= 139,
						 data_format=tf.keras.backend,image_data_format(), require_flatten=False, weights=weights)

	img_input = tf.keras.layers.Input(shape=input_shape)
'''	if tf.backend.image_data_format() == 'channels_first':
		bn_axis = 1
	else:
		bn_axis = 3'''

	# Stage 1
	x = tf.keras.layers.ZeroPadding(padding=(3,3), name='conv1_pad')(img_input)
	x = tf.keras.layers.Conv2D(64, (7,7), strides = (2,2), padding = 'valid', kernel_initializer='he_normal', name='conv1')(x)
	x = BatchNorm(name='bn_conv1', axis=bn_axis)(x, training=train_bn)
	x = tf.keras.layers.Activation('relu')(x)
	x = tf.keras.layers.MaxPooling2D((3,3), strides=(2,2))(x)

	# Stage 2
	x = conv_block(x, 3, [64,64,256], stage=2, block='a', strides = (1,1), train_bn=train_bn)(x)
	x = identity_block(x, 3, [64,64,256], stage=2, block='b', train_bn=train_bn)
	x = identity_block(x, 3, [64,64,256], stage=2, block='c', train_bn=train_bn)

	# Stage 3
	x = conv_block(x, 3, [128,128,512], stage=3, block='a', strides=(1,1), train_bn=train_bn)
	x = identity_block(x, 3, [128,128,512], stage=3, block='b', train_bn=train_bn)
	x = identity_block(x, 3, [128,128,512], stage=3, block='c', train_bn=train_bn)
	x = identity_block(x, 3, [128,128,512], stage=3, block='d', train_bn=train_bn)

	# Stage 4
	x = conv_block(x, 3, [256,256,1024], stage=4, block='a', train_bn=train_bn)
	for i in range(22):
		x = identity_block(x, 3, [256,256, 1024], stage=4, block=chr(98+i), train_bn=train_bn)

	# Stage 5
	if stage5:
		x = conv_block(x, 3, [512,512,2048], stage=5, block='a', train_bn=train_bn)
		x = identity_block(x, 3, [256,256,2048], stage=5,block='b', train_bn=train_bn)
		x = identity_block(x, 3, [256,256,2048], stage=5, block='c', train_bn=train_bn)

	x = tf.keras.layers.AveragePooling2D((7,7), name='avg_pool')(x)

	if include_top:
		tf.keras.layers.Flatten()(x)
		tf.keras.layers.Dense(classes, activation=activation, name='fc2')(x)

	else:
		if pooling == 'avg':
			x = tf.keras.layers.AveragePooling2D()(x)
		elif pooling == 'max':
			x = tf.keras.layers.MaxPooling2D()(x) 

	model = tf.keras.Model(img_input, x, name='resnet101')

	return model
