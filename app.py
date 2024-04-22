import pickle
import bz2
from flask import Flask, request, jsonify, render_template, session,redirect
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from app_logger import log
from Model.mongodb import mongodbconnection
from Model.sqldb import mysqlconnector
import warnings
warnings.filterwarnings("ignore")


app = Flask(__name__)
app.secret_key='123-456-789'

# Import Classification and Regression model file
pickle_in = bz2.BZ2File('mdl/classification.pkl', 'rb')
R_pickle_in = bz2.BZ2File('mdl/regression.pkl', 'rb')
model_C = pickle.load(pickle_in)
model_R = pickle.load(R_pickle_in)

# Data retrieved from DB using mongoconnection module
dbcon = mongodbconnection(username={username}, password={password})
list_cursor = dbcon.getdata(dbName='FireDataML', collectionName='ml_task')
log.info('Connected to Mongodb and data retrieved')

# Data From MongoDB is used for Standardization
df = pd.DataFrame(list_cursor)
df.drop('_id', axis=1, inplace=True)
log.info('DataFrame created')
scaler = StandardScaler()
X = df.drop(['FWI', 'Classes'], axis=1)
# Standardize
X_reg_scaled = scaler.fit_transform(X)
log.info('Standardization done')

obj_model=mysqlconnector()
# Route for homepage
@app.route('/')
def home():
    log.info('Home page loaded successfully')
    return render_template('index.html')


@app.route('/signup',methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        user_name = request.form["username"]
        Email     = request.form["email"]
        Password  = request.form["password"]
        check = obj_model.user_exists_signup(Email)
        log.info('Information retrieved from SQL.')
        if len(check)>0:
            return render_template('signup.html',message="User already exists")
        else:
            res = obj_model.user_signup(user_name,Email,Password)
            session["id"] = res[0][1]
            print("session --> ",session)
            return redirect("/model-input")
    else:
        return render_template("signup.html")

@app.route('/login',methods=['GET','POST'])
def login():
    if request.method == 'POST':
        Email = request.form["email"]
        Password = request.form["password"]
        check = obj_model.user_login(Email,Password)
        log.info('Information retrieved from SQL.')
        if len(check)>0:
            session["id"] = check[0][1]
            print("session --> ",session)
            return redirect("/model-input")
        else:
            return render_template('login.html',message="Invalid Email or Password!")
    else:
        if 'id' in session:
            return render_template('nologinreq.html')
        else:
            return render_template("login.html")

@app.route('/logout')
def logout():
        session.pop("id")
        log.info("Log out successful.")
        return redirect('/')

@app.route('/model-input')
def modelinput():
    if 'id' in session:
        return render_template('model-input.html')
    else:
        return render_template('nologin.html')

# Route for API Testing
@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']
        print(data)
        log.info('Input from Api testing', data)
        new_data = [list(data.values())]
        final_data = scaler.transform(new_data)
        output = int(model_C.predict(final_data)[0])
        if output == 1:
            text = 'The Forest in Danger'
        else:
            text = 'Forest is Safe'
        return jsonify(text, output)
    except Exception as e:
        output = 'Check the input again!'
        log.error('error in input from Postman', e)
        return jsonify(output)


# Route for Classification Model
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        final_features = [np.array(data)]
        final_features = scaler.transform(final_features)
        output = model_C.predict(final_features)[0]
        log.info('Prediction done for Classification model')
        if output == 0:
            text = 'Forest is Safe!'
            return render_template('model-input.html', prediction_text1="{} ".format(text))
        text = 'Forest is in Danger!'
        return render_template('model-input.html', prediction_text1="{} --- Fire can occur!".format(text))
    except Exception as e:
        log.error('Input error, check input', e)
        return render_template('model-input.html', prediction_text1="Check the Input again!!!")


# Route for Regression Model
@app.route('/predictR', methods=['POST'])
def predictR():
    try:
        data = [float(x) for x in request.form.values()]
        data = [np.array(data)]
        data = scaler.transform(data)
        output = model_R.predict(data)[0]
        log.info('Prediction done for Regression model')
        if output > 29.0:
            return render_template('model-input.html', prediction_text2="Fire Weather Index is {:.4f} ---- Extremely hazardous. Fire occurrence very likely".format(output))
        elif (output>=16.0 and output<=29.0):
            return render_template('model-input.html', prediction_text2="Fire Weather Index is {:.4f} ---- Fire occurrence more favourable".format(output))
        elif (output>=8.0 and output<16.0):
            return render_template('model-input.html', prediction_text2="Fire Weather Index is {:.4f} ---- Fire occurrence favourable".format(output))
        elif (output>=4.0 and output<8.0):
            return render_template('model-input.html', prediction_text2="Fire Weather Index is {:.4f} ---- Fire occurrence less favourable".format(output))
        elif (output>=1.0 and output<4.0):
            return render_template('model-input.html', prediction_text2="Fire Weather Index is {:.4f} ---- Fire occurrence unfavourable".format(output))
        else:
            return render_template('model-input.html', prediction_text2="Fire Weather Index is {:.4f} ---- Safe.. Fire occurrence very low".format(output))
    except Exception as e:
        log.error('Input error, check input', e)
        return render_template('model-input.html', prediction_text2="Check the Input again!!!")


# Run APP in Debug mode
if __name__ == "__main__":
    app.run(debug=True)
