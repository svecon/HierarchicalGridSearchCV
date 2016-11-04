# HierarchicalGridSearchCV
A variant of scikit GridSearchCV which saves time using memory tradeoff

When using a Pipeline in scikit GridSearchCV, all models in the Pipeline need to be re-fitted for every single combination of parameters.

This HierarchicalGridSearchCV fits the Pipeline level by level and therefore it saves a lot of time re-fitting the models in the lower levels of the pipeline.

## Why to use this

Imagine we have a pipeline with 3 steps (2 preprocessing steps and 1 classifier).

* 1st step has 2 param with 3 possible values.
* 2nd step has 2 params with 5 possible values each.
* 3rd step has 1 param with 4 possible values.

In scikit GridSearchCV, all models would be fitted for every combination of the parameters: 3*3*5*5*4=900 times.

Whereas in HierarchicalGridSearchCV:
* 1st model would only be fitted 3*3=9 times
* 2nd model would only be fitted 3*3*5*5=225 times
* 3rd model would be fitted 3*3*5*5*4=900 times

In this particular scenario HierarchicalGridSearchCV will spend 100 times less time fitting the 1st model and 4 times less time fitting 2nd model.

## Code example

Code example when using the standard GridSearchCV:

```
from sklearn.model_selection import GridSearchCV
from sklearn import svm, datasets
from sklearn.pipeline import Pipeline
from metric_learn import LFDA

iris = datasets.load_iris()
clf = GridSearchCV(
    Pipeline([('lfda', LFDA()), ('svr', svm.SVC())]),
    {'lfda__k':(1, 3), 'svr__C':(.1, .5)}
)
clf.fit(iris.data, iris.target)
print(clf.best_score_, clf.best_params_)
```

Code example when using the HierarchicalGridSearchCV:

```
from hierarchical_grid_search_cv.HierarchicalGridSearchCV import HierarchicalGridSearchCV
from sklearn import svm, datasets
from sklearn.pipeline import Pipeline
from metric_learn import LFDA

iris = datasets.load_iris()
clf = HierarchicalGridSearchCV(
    [LFDA(), svm.SVC()],
    [{'k':(1, 3)}, {'C':(.1, .5)}]
)
clf.fit(iris.data, iris.target)
print(clf.best_score_, clf.best_params_)
```
