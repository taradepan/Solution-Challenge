from flask import Flask, render_template,request
import pickle
import numpy as np
import pandas
import nbformat
from nbconvert import HTMLExporter
import ml
import os

app = Flask(__name__, template_folder='templates')


model_file_path = './model.pkl'

if os.path.exists(model_file_path):
    model = pickle.load(open(model_file_path, 'rb'))
else:
    ml.train_model()
    # Assuming that train_model() saves the model to model_file_path
    model = pickle.load(open(model_file_path, 'rb'))

# Convert Jupyter Notebook to HTML
# def notebook_to_html(notebook_path):
#     with open(notebook_path, 'r') as nb_file:
#         nb_content = nb_file.read()
#     nb = nbformat.reads(nb_content, as_version=4)
#     html_exporter = HTMLExporter()
#     html_content, _ = html_exporter.from_notebook_node(nb)
#     return html_content

@app.route('/')
# def jupyter_in_flask():
#     notebook_path = 'Bitcoin-Prediction1.ipynb'
#     jupyter_html = notebook_to_html(notebook_path)
#     return render_template('jupiter_template.html', jupyter_html=jupyter_html)

# if __name__ == '__main__':
#     app.run(debug=True)

def index():
    return render_template("./index.html")

@app.route('/predict',methods=['POST'])
def predict():
    input_text = request.form['text']
    input_text_sp = input_text.split(',')
    np_data = np.asarray(input_text_sp, dtype=np.float32)
    prediction = model.predict(np_data.reshape(1,-1))

    if prediction == 1:
        output = "This person  has a Parkinson Disease"
    else:
        output = "This person has no Parkinson Disease"

    return render_template("./index.html",message= output)


if __name__ == "__main__":
    app.run(debug=True)