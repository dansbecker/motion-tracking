from keras.layers import Input, Dense, merge, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from motion_tracker.utils.image_generator import CompositeGenerator
from keras import backend as K
from time import sleep

img_edge_size = 100
backend_id = 'th'

if backend_id == 'th':
    img_shape = (3, img_edge_size, img_edge_size)
if backend_id == 'tf':
    img_shape = (3, img_edge_size, img_edge_size)

generic_img = Input(shape=img_shape)
layer = Convolution2D(30, 5, 5, activation='relu', border_mode='same',
                                dim_ordering=backend_id)(generic_img)
layer = Convolution2D(25, 4, 4, activation='relu', border_mode='same',
                                dim_ordering=backend_id)(layer)
layer = Convolution2D(16, 3, 3, activation='relu', border_mode='same',
                                dim_ordering=backend_id)(layer)
layer = Convolution2D(16, 3, 3, activation='relu', border_mode='valid',
                                dim_ordering=backend_id)(layer)
layer = Convolution2D(12, 3, 3, activation='relu', border_mode='valid',
                                dim_ordering=backend_id)(layer)
layer = MaxPooling2D(pool_size=(2,2), dim_ordering=backend_id)(layer)
out_from_reusable_model = Flatten()(layer)
reusable_img_featurizer = Model(generic_img, out_from_reusable_model)

start_img = Input(shape=img_shape, name='start_img')
end_img = Input(shape=img_shape, name='end_img')

start_img_features = reusable_img_featurizer(start_img)
end_img_features = reusable_img_featurizer(end_img)
start_box = Input(shape=(4,), name='start_box')
entryto_dense = merge([start_img_features, end_img_features, start_box],
                       mode='concat', concat_axis=1)

layer = Dense(50, activation='relu')(entryto_dense)
remaining_layer_widths = [50, 25]
for n_nodes in remaining_layer_widths:
    layer = Dense(n_nodes, activation='relu')(layer)

x0 = Dense(1, activation='linear', name='x0')(layer)
y0 = Dense(1, activation='linear', name='y0')(layer)
x1 = Dense(1, activation='linear', name='x1')(layer)
y1 = Dense(1, activation='linear', name='y1')(layer)

mymodel = Model(input=[start_img, start_box, end_img], output=[x0, y0, x1, y1])

mymodel.compile(loss ={'x0': 'mean_absolute_error',
                        'y0': 'mean_absolute_error',
                        'x1': 'mean_absolute_error',
                        'y1': 'mean_absolute_error'},
                        loss_weights={'x0': 0.25, 'y0': 0.25,
                                      'x1': 0.25, 'y1': 0.25},
                optimizer='adam')

print(mymodel.summary())
mygen = CompositeGenerator(output_width = img_edge_size,
						  output_height = img_edge_size,
                          crops_per_image=10,
                          batch_size = 20,
                          desired_dim_ordering = backend_id).flow()
print('Fitting')
mymodel.fit_generator(mygen, samples_per_epoch=1000, nb_epoch=10,
					   max_q_size=5, verbose=1)
sleep(1)
a = next(mygen)
preds = mymodel.predict(a[0])
actuals = a[1]
print([i.mean() for i in preds])
print([i.std() for i in preds])
