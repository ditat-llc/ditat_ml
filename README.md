# ditat_ml
ditat.io framework for deployment of machine learning models.

For further information visit [ditat.io](https://ditat.io)

## Installation (and updates)
```bash
pip install git+https://github.com/ditat-llc/ditat_ml.git (--upgrade)
```
Using ssh
```bash
pip install git+ssh://git@github.com/ditat-llc/ditat_ml.git (--upgrade)
```
The other option is to clone the repo and install it locally.
```bash
pip install -e ./ditat_ml
```

## Usage

This framework gives you the flexibility to operate with high-level api (Pipeline) and also with its low-level functions, among others.

### High-level usage
**Train and Test**

```python
from ditat_ml import Pipeline

p = Pipeline()
p.load_data(path='dataset.csv')
p.load_X_y(X_columns, y_columns)
p.preprocessing(**mappings ) # Check documentation for more details.
p.scale()

p.model = YOUR_MODEL_CHOICE

p.train()
```

**Deploy**

One you are satisfied with your model's performance, you can deploy it.
```python
p.deploy(name='my_model')
```

**Predict**
```python
from ditat_ml import Pipeline

p = Pipeline()
predictions = p.predict(path='dataset.csv', model_name='my_model')
```

### Low-level usage
You can also use its low-level functions to give you more flexibility.
```python
from ditat_ml import utility_functions
```
