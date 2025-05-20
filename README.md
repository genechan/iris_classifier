## Test app.py

### build model

```
python3 src/train_model3.py
```

### run server

```
python3 src/app.py
```

### run Curl

```
curl -X POST -H "Content-Type: application/json" -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}' http://127.0.0.1:5000/predict
```

### Results of Curl

```
{"species":"setosa"}
```
