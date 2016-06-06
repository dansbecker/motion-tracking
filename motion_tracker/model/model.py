from keras.layers import Input, Dense, merge, Flatten, Convolution2D, MaxPooling2D, Dropout, ZeroPadding2D
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from motion_tracker.utils.image_generator import CompositeGenerator
import os
import numpy as np
from time import sleep
from motion_tracker.utils.crop_vis import show_img

def make_model(img_edge_size, backend_id):
    '''returns untraned version model used for image tracking
        img_edge: number of pixels for images (height and width are same)
        backend_id: either "tf" or "th"
    '''

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
    layer = ZeroPadding2D(padding=(2, 2), dim_ordering=backend_id)(generic_img)
    layer = Convolution2D(24, 11, 11, activation='relu',
                          border_mode='same', dim_ordering=backend_id)(layer)
    layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), 
                         dim_ordering=backend_id)(layer)
    layer = BatchNormalization()(layer)
    layer = ZeroPadding2D(padding=(1, 1), dim_ordering=backend_id)(layer)
    layer = Convolution2D(56, 5, 5, activation='relu', border_mode='same',
                          dim_ordering=backend_id)(layer)
    layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), 
                         dim_ordering=backend_id)(layer)
    layer = BatchNormalization()(layer)
    layer = ZeroPadding2D(padding=(1, 1), dim_ordering=backend_id)(layer)
    layer = Convolution2D(96, 3, 3, activation='relu', border_mode='same',
                          dim_ordering=backend_id)(layer)
    layer = BatchNormalization()(layer)
    layer = ZeroPadding2D(padding=(1, 1), dim_ordering=backend_id)(layer)
    layer = Convolution2D(96, 3, 3, activation='relu', border_mode='same',
                          dim_ordering=backend_id)(layer)
    layer = BatchNormalization()(layer)
    layer = ZeroPadding2D(padding=(1, 1), dim_ordering=backend_id)(layer)
    layer = Convolution2D(64, 3, 3, activation='relu', border_mode='same',
                          dim_ordering=backend_id)(layer)
    layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), 
                         dim_ordering=backend_id)(layer)
    layer = BatchNormalization()(layer)
    layer = Convolution2D(20, 3, 3, activation='relu', border_mode='same',
                          dim_ordering=backend_id)(layer)
    layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), 
                         dim_ordering=backend_id)(layer)
    layer = BatchNormalization()(layer)
    reusable_img_featurizer = Model(generic_img, layer)

    ########### APPLY FEATURIZER TO STARTING AND ENDING IMAGES ###########
    start_img = Input(shape=img_shape, name='start_img')
    start_img_features = reusable_img_featurizer(start_img)
    end_img = Input(shape=img_shape, name='end_img')
    end_img_features = reusable_img_featurizer(end_img)

    ################## FLATTEN AND MERGE EVERYTHING TOGETHER ################

    start_img_features = Flatten()(start_img_features)
    end_img_features = Flatten()(end_img_features)
    layer = merge([start_img_features, end_img_features],
                   mode='concat',
                   concat_axis=1)

    ########################## FC LAYERS AFTER MERGE ##########################
    dense_layer_widths = [1024, 1024, 250]
    for n_nodes in dense_layer_widths:
        layer = Dense(n_nodes, activation='linear')(layer)

    ########################## CREATE OUTPUT #################################
    x0 = Dense(1, activation='linear', name='x0')(layer)
    y0 = Dense(1, activation='linear', name='y0')(layer)
    x1 = Dense(1, activation='linear', name='x1')(layer)
    y1 = Dense(1, activation='linear', name='y1')(layer)

    my_model = Model(input=[start_img, end_img],
                    output=[x0, y0, x1, y1])
    my_model.compile(loss ={'x0': 'mean_absolute_error',
                           'y0': 'mean_absolute_error',
                           'x1': 'mean_absolute_error',
                           'y1': 'mean_absolute_error'},
                           loss_weights={'x0': 0.25, 'y0': 0.25,
                                         'x1': 0.25, 'y1': 0.25},
                            optimizer='adam')
    return my_model


def fit_model(my_model, my_gen, img_edge_size, backend_id):
    '''Fits and returns model using image/box pairs from my_gen

        my_model: the keras model object to be fit
        my_gen: the generator that supplies training data
        img_edge: number of pixels for images (height and width are same)
        backend_id: either "tf" or "th"
    '''
    print('Fitting model')

    my_model.fit_generator(my_gen, samples_per_epoch=2000,
                           nb_epoch=100, max_q_size=5, verbose=1)
    return my_model



if __name__ == "__main__":
    img_edge_size = 224
    backend_id = 'th'
    weights_fname = './work/model_weights.h5'
    model_spec = './work/model_architecture.json'
    my_gen = CompositeGenerator(output_width = img_edge_size,
						        output_height = img_edge_size,
                                crops_per_image=10,
                                batch_size = 50,
                                desired_dim_ordering = backend_id).flow()

    my_model = make_model(img_edge_size, backend_id)
    if os.path.exists(weights_fname):
        print('Loading saved model')
        my_model.load_weights(weights_fname)
    print(my_model.summary())

    my_model = fit_model(my_model, my_gen, img_edge_size, backend_id)
    print('Saving model')
    my_model.save_weights(weights_fname, overwrite=True)


    sleep(1)    # let generator finish existing threads
    print('Model Results')
    X, y = next(my_gen)
    preds = my_model.predict(X)
    print('Mean Locations For Predicted Boxes:  ', [i.mean() for i in preds])
    print('Std dev of locations for predicted boxes:  ', [i.std() for i in preds])
    print('Loss on new batch:  ', my_model.evaluate(X, y))
    for i in range(50): 
        actual_box = np.array([y[coord][i] for coord in ('x0', 'y0', 'x1', 'y1')])
        pred_box = np.array([preds[coord][i] for coord in (0, 1, 2, 3)])
        boxes = {'actual': actual_box, 'predicted': pred_box}
        img = X['end_img'][i].swapaxes(0, 2)
        show_img(img, boxes=boxes, save=True, filepath='work/preds/predicted_{}.jpg'.format(i))

