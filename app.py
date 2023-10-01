import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the pickled SVM model with GridSearchCV
with open('svm_gsr.pkl', 'rb') as model_file:
    svm_gsr = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the HTML form
    features = {
        'ri': float(request.form['ri']),
        'na': float(request.form['na']),
        'al': float(request.form['al']),
        'ca': float(request.form['ca'])
    }

    # Convert the features to a NumPy array
    final_features = np.array(list(features.values())).reshape(1, -1)

    # Make predictions using your loaded SVM model
    prediction = svm_gsr.best_estimator_.predict(final_features)

    # You can format the output as needed
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Predicted glass type: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
