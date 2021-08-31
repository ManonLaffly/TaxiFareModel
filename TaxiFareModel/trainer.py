# imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from TaxiFareModel.utils import compute_rmse
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from TaxiFareModel.encoders import TimeFeaturesEncoder
from TaxiFareModel.encoders import DistanceTransformer


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        time_pipe = make_pipeline(TimeFeaturesEncoder('pickup_datetime'),
                                  OneHotEncoder(handle_unknown='ignore'))
        dist_pipe = make_pipeline(DistanceTransformer(), StandardScaler())

        # column transformer
        dist_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        time_cols = ['pickup_datetime']

        preproc_pipe = ColumnTransformer([('distance', dist_pipe, dist_cols),
                                  ('time', time_pipe, time_cols)],
                                         remainder="drop") # remainder='passthrough'

        # workflow
        self.pipeline = Pipeline(steps=[('preproc', preproc_pipe),
                         ('linear_model', LinearRegression())])


    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()

        self.pipeline.fit(self.X, self.y)
        return self.pipeline

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse



if __name__ == "__main__":
    # get data
    # clean data
    # set X and y
    # hold out
    # train
    # evaluate
    print('TODO')
