from keras.layers import Input, Dense, merge, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from motion_tracker.utils.image_generator import CompositeGenerator
from keras import backend as K
from time import sleep


############################### SETUP ###############################
img_edge_size = 100
backend_id = 'th'

if backend_id == 'tf':
    channel_index = 3
else:
    channel_index = 1

if backend_id == 'th':
    img_shape = (3, img_edge_size, img_edge_size)
    mask_shape = (1, img_edge_size, img_edge_size)
if backend_id == 'tf':
    img_shape = (img_edge_size, img_edge_size, 3)
    mask_shape = (img_edge_size, img_edge_size, 1)


###### DEFINE FEATURIZER APPLIED TO STARTING AND ENDING IMAGES ######
generic_img = Input(shape=img_shape)
layer = Convolution2D(20, 3, 3, activation='relu', border_mode='same',
                                dim_ordering=backend_id)(generic_img)
layer = MaxPooling2D(pool_size=(3, 3), dim_ordering=backend_id)(layer)
layer = Convolution2D(30, 3, 3, activation='relu', border_mode='same',
                                dim_ordering=backend_id)(layer)
layer = Convolution2D(30, 3, 3, activation='relu', border_mode='same',
                                dim_ordering=backend_id)(layer)
layer = Convolution2D(30, 3, 3, activation='relu', border_mode='same',
                                dim_ordering=backend_id)(layer)
reusable_img_featurizer = Model(generic_img, layer)


########### APPLY FEATURIZER TO STARTING AND ENDING IMAGES ###########
start_img = Input(shape=img_shape, name='start_img')
start_img_features = reusable_img_featurizer(start_img)
end_img = Input(shape=img_shape, name='end_img')
end_img_features = reusable_img_featurizer(end_img)


#### ADD MASK OF STARTING BOUNDING BOX TO FEATURIZED STARTING IMG ####

start_box_mask = Input(shape=mask_shape, name='start_box_mask')

# to downscale mask the same way we did the images, we use the same pooling layers
# This is works ONLY if border_mode = same for all conv layers in the image featurizer
pooling_layers = [layer for layer in reusable_img_featurizer.layers if "pooling" in layer.name]
start_box_mask_layer = start_box_mask
for layer in pooling_layers:
    start_box_mask_layer = layer(start_box_mask_layer)


start_img_features = merge([start_img_features, start_box_mask_layer],
                           mode='concat', concat_axis=channel_index)
start_img_features = Convolution2D(40, 3, 3, activation='relu', border_mode='same',
                                   dim_ordering=backend_id)(start_img_features)

################## FLATTEN AND MERGE EVERYTHING TOGETHER ################

start_box = Input(shape=(4,), name='start_box')
start_img_features = Flatten()(start_img_features)
end_img_features = Flatten()(end_img_features)
layer = merge([start_img_features, end_img_features, start_box],
              mode='concat',
              concat_axis=1)

########################## FC LAYERS AFTER MERGE ##########################
dense_layer_widths = [60, 60]
for n_nodes in dense_layer_widths:
    layer = Dense(n_nodes, activation='relu')(layer)

########################## CREATE OUTPUT #################################
x0 = Dense(1, activation='linear', name='x0')(layer)
y0 = Dense(1, activation='linear', name='y0')(layer)
x1 = Dense(1, activation='linear', name='x1')(layer)
y1 = Dense(1, activation='linear', name='y1')(layer)

mymodel = Model(input=[start_img, end_img, start_box, start_box_mask], output=[x0, y0, x1, y1])

mymodel.compile(loss ={'x0': 'mean_absolute_error',
                        'y0': 'mean_absolute_error',
                        'x1': 'mean_absolute_error',
                        'y1': 'mean_absolute_error'},
                        loss_weights={'x0': 0.25, 'y0': 0.25,
                                      'x1': 0.25, 'y1': 0.25},
                optimizer='adam')

print(mymodel.summary())

################################ FIT MODEL ################################
mygen = CompositeGenerator(output_width = img_edge_size,
						  output_height = img_edge_size,
                          crops_per_image=5,
                          batch_size = 10,
                          desired_dim_ordering = backend_id).flow()
print('Fitting')
mymodel.fit_generator(mygen, samples_per_epoch=1000, nb_epoch=6,
					  max_q_size=5, verbose=1)

################## PRINT MEAN AND SD OF BOX LOCATIONS ###################
sleep(1)
X, y = next(mygen)
preds = mymodel.predict(X)
print('Mean Locations For Predicted Boxes:  ', [i.mean() for i in preds])
print('Std dev of locations for predicted boxes:  ', [i.std() for i in preds])
print('Loss on new batch:  ', mymodel.evaluate(X, y))
