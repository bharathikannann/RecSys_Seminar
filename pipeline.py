from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd

from myfm import MyFMRegressor
from myfm.utils.callbacks import RegressionCallback

class MyRegressionCallback(RegressionCallback):
   def __call__(self, i, fm, hyper, history):
      should_stop, description = super(MyRegressionCallback, self).__call__(i, fm, hyper, history)
      trace_result = self.result_trace[-1]
      if len(self.result_trace) > 8:
         for index in range(8):
            old_trace_result = self.result_trace[-(index + 1)]
            if abs(old_trace_result['rmse'] - trace_result['rmse']) > 0.0001:
               return (should_stop, description)
         return (True, description)
      return (should_stop, description)

dataset = load_breast_cancer()
X = pd.read_csv('x_data', sep=',', encoding='latin-1', nrows=100, header=None)
y = pd.read_csv('y_data', sep=',', encoding='latin-1', nrows=100, header=None)
group_shapes = pd.read_csv('group_shapes', sep=',', encoding='latin-1', header=None)

X_train, X_test, y_train, y_test = train_test_split(
   X, y, random_state=42
)

callback = MyRegressionCallback(5, X_test, y_test.values)

fm = MyFMRegressor(rank=2).fit(X_train, y_train, n_iter=10000, group_shapes=group_shapes.values[0], callback=callback)
error = metrics.mean_squared_error(y_test, fm.predict(X_test), squared=False)
print(error)