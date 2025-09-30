from federated_gbdt.models.gbdt.dp_fedxgb import PrivateGBDT
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import dataloader as dl
import warnings
warnings.filterwarnings("ignore")
n = 10



sum = 0
for i in range(1,6):
    X, y = dl.wine()

    y_scale = dl.scale(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_scale,
                    test_size = 0.2, random_state = 2987)

    client_X_train, client_X_test, client_y_train, client_y_test = dl.horizontally_split_data(n, 
                    X_train, X_test, y_train, y_test)
    xgb_model = PrivateGBDT(num_trees=20, n=n, epsilon=1, split_method="hist_based", 
                                dp_method="gaussian_cdp")
    xgb_model = xgb_model.fit(n, client_X_train, client_y_train, X_train)
    y_pred = xgb_model.predict_weight(X_test)

    y_pred, y_test_tmp = dl.rescale(y, y_test, y_pred)


    rmse = mean_squared_error(y_test_tmp, y_pred) ** 0.5
    sum += rmse
    print('The rmse of prediction under budget = ', i, 'is:', rmse)

print("average rmse = ", sum/5)