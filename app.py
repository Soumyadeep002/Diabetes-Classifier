from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = './artifacts/Diabetes_classifier.pickle'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
        if (request.method=='POST'):
            name=request.form.get('name')
            email=request.form.get('email')
            phone=request.form.get('phone')
            gender=request.form.get('gender')
            age = int(request.form['age'])
            height=float(request.form.get('height'))
            weight=float(request.form.get('weight'))
            preg = int(request.form['preg'])
            bp = int(request.form['bp'])
            glucose = int(request.form['glucose'])
            st = int(request.form['skithik'])
            insulin = int(request.form['insulin'])
            bmi = float(weight/(height*height))
            dpf = float(request.form['dpf'])
            
            data = np.array([[preg,  glucose, bp, st, insulin, bmi, dpf, age]])
            my_prediction = classifier.predict(data)
            
            return render_template("result.html", prediction = my_prediction)

@app.route("/about")
def about():
     return render_template("about.html")

@app.route("/contact")
def contat():
     return render_template("contact.html")



if __name__=="__main__":

    
    app.run(debug=True)
