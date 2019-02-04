# Machine Learning challenge

The goal of this challenge is to build a Machine Learning model to predict if a given adult's yearly income is above or below $50k.

To succeed, I developed a `solution` Python package that implements a `get_pipeline` function that returns:

- [x] an [sklearn.pipeline.Pipeline](http://scikit-learn.org/stable/modules/pipeline.html)
- [x] that chains a series of [sklearn Transformers](http://scikit-learn.org/stable/data_transforms.html) to preprocess the data,
- [x] and ends with a [custom sklearn Estimator](http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator) that wraps a [TensorFlow model](https://www.tensorflow.org/get_started/custom_estimators),
- [x] and will be fed a pandas DataFrame of the [Adult Data Set](http://mlr.cs.umass.edu/ml/datasets/Adult) to train and evaluate the pipeline.
