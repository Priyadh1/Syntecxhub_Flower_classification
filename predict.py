import argparse
import joblib
import pandas as pd

parser = argparse.ArgumentParser(description='Predict iris species from measurements in cm')
parser.add_argument('--sepal_length', type=float, required=True)
parser.add_argument('--sepal_width', type=float, required=True)
parser.add_argument('--petal_length', type=float, required=True)
parser.add_argument('--petal_width', type=float, required=True)
args = parser.parse_args()

model = joblib.load('logistic_regression_model.joblib')

x = pd.DataFrame(
    [[args.sepal_length, args.sepal_width, args.petal_length, args.petal_width]],
    columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
)

prediction = model.predict(x)[0]
confidence = model.predict_proba(x)[0].max()
print(f'Predicted species: {prediction} (confidence: {confidence:.1%})')
