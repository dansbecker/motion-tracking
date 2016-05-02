from keras.layers import Input, Dense, merge, Flatten
from keras.layers.convolutional import Convolution2D
from keras.models import Model
from motion_tracker.utils.image_generator import imagenet_generator
from keras import backend as K


generic_img = Input(shape=(256, 256, 3))
generic_layer_1 = Convolution2D(2, 3, 3, activation='relu', dim_ordering='tf')(generic_img)
generic_layer_2 = Convolution2D(2, 3, 3, activation='relu', dim_ordering='tf')(generic_layer_1)
out_from_reusable_model = Flatten()(generic_layer_2)
reusable_img_featurizer = Model(generic_img, out_from_reusable_model)


start_img = Input(shape=(256, 256, 3), name='start_img')
end_img = Input(shape=(256, 256, 3), name='end_img')

start_img_features = reusable_img_featurizer(start_img)
end_img_features = reusable_img_featurizer(end_img)
start_box = Input(shape=(4,), name='start_box')
entry_to_dense = merge([start_img_features, end_img_features, start_box],
                       mode='concat', concat_axis=1)

l1 = Dense(5, activation='relu')(entry_to_dense)
l2 = Dense(5, activation='relu')(l1)
end_box = Dense(4, activation='relu', name='end_box')(l2)

my_model = Model(input=[start_img, start_box, end_img], output=end_box)

my_model.compile(loss = 'mean_absolute_error', optimizer='Adam')
my_gen = imagenet_generator()
my_model.fit_generator(my_gen, samples_per_epoch=1000, nb_epoch=10, max_q_size=5)
