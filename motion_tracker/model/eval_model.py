import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from motion_tracker.model.model import make_model
from motion_tracker.utils.image_generator import AlovGenerator

if __name__ == "__main__":
    img_edge_size = 224
    backend_id = 'th'
    error_analysis_dir = './work/error_analysis/'
    os.makedirs(error_analysis_dir, exist_ok=True)
    weights_fname = './work/model_weights.h5'

    my_gen = AlovGenerator(output_width = img_edge_size,
						   output_height = img_edge_size,
                           crops_per_image=4,
                           batch_size = 500,
                           desired_dim_ordering = backend_id).flow()

    my_model = make_model(img_edge_size, backend_id)
    my_model.load_weights(weights_fname)
    X, y = next(my_gen)
    preds = my_model.predict(X)
    pred_df = pd.DataFrame({'pred_x0': preds[0].ravel(),
                             'pred_y0': preds[1].ravel(),
                             'pred_x1': preds[2].ravel(),
                             'pred_y1': preds[3].ravel()})
    actual_df = pd.DataFrame(y)
    pred_df = pred_df.join(actual_df)
    for coord in ('x0', 'y0', 'x1', 'y1'):
        this_coord_pred = pred_df['pred_' + coord]
        this_coord_actual = pred_df[coord]
        pred_df[coord+'_error'] = (this_coord_actual - this_coord_pred).abs()
        my_plot = sns.jointplot(this_coord_pred, this_coord_actual, kind="kde")
        my_plot.savefig(error_analysis_dir + coord + '.png')
    pred_df['mean_coordinate_error'] = pred_df.eval('(x0_error + y0_error + x1_error + y1_error) / 4')
    error_plot = sns.distplot(pred_df.mean_coordinate_error)
    plt.savefig(error_analysis_dir + 'mean_coordinate_error_dist.png')
    plt.close()
    pred_df.to_csv(error_analysis_dir + 'prediction_error_info.csv', index=False)
