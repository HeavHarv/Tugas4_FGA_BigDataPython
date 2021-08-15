
# flask_ngrok_example.py
from flask import Flask,render_template,flash, redirect,url_for,session,logging,request
from flask_sqlalchemy import SQLAlchemy
#from flask_ngrok import run_with_ngrok

# !pip install SQLAlchemy
# !pip install Flask-SQLAlchemy

# app = Flask(__name__)
# app = Flask(__name__, static_url_path='/static')
app = Flask(__name__, static_folder='static')
#run_with_ngrok(app)  # Start ngrok when app is run

# app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////fga_big_data.db'
db = SQLAlchemy(app)

class user(db.Model):
  # __tablename__ = 'user'
  id = db.Column(db.Integer, primary_key=True)
  # Mail 	Password 	Name 	Level
  Mail = db.Column(db.Text) # sbg Username
  Password = db.Column(db.Text)
  Name = db.Column(db.Text)
  Level = db.Column(db.Text)

  # username = Mail
  # password = Password

@app.route("/")
def index():
    
    return render_template("index.html")

@app.route("/signout")
def sign_out():
    session.pop('name')
    session.pop('mail')
    return redirect(url_for("index"))

@app.route("/login",methods=["GET", "POST"])
def login():
    msg = ""
    if request.method == "POST":
        mail = request.form["mail"]
        passw = request.form["passw"]
        
        login = user.query.filter_by(Mail=mail, Password=passw).first()
        print(login)
        if login is not None:
            # return redirect(url_for("index"))
            session['name'] = login.Name
            session['mail'] = login.Mail
            # return render_template('bigdataApps.html', login=login)

            return redirect(url_for("bigdataApps"))
        elif login is None:
            msg = "Masukkan Username (Email) dan Password dgn Benar"
            print(msg)

    return render_template("login.html", msg = msg)

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        mail = request.form['mail']
        uname = request.form['uname']
        passw = request.form['passw']

        register = user(Mail = mail, Name = uname, Password = passw)
        # register = user(Mail = mail, Password = passw)
        db.session.add(register)
        db.session.commit()

        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/bigdataApps", methods=["GET", "POST"])
def bigdataApps():
  if request.method == 'POST':
    import pandas as pd
    import numpy as np
    dataset = request.FILES['inputDataset']
    # dataset = request.files['inputDataset']
    #dataset = 'dataset_dump.csv'

    persentase_data_training = 90
    banyak_fitur = int(request.POST['banyakFitur'])
    banyak_hidden_neuron = int(request.POST['banyakHiddenNeuron'])
    dataset = pd.read_csv(dataset, delimiter=';', names = ['Tanggal', 'Harga'], usecols=['Harga'])
    minimum = int(dataset.min()-10000)
    maksimum = int(dataset.max()+10000)
    new_banyak_fitur = banyak_fitur + 1
    hasil_fitur = []
    for i in range((len(dataset)-new_banyak_fitur)+1):
      kolom = []
      j = i
      while j < (i+new_banyak_fitur):
        kolom.append(dataset.values[j][0])
        j += 1
      hasil_fitur.append(kolom)
    hasil_fitur = np.array(hasil_fitur)
    data_normalisasi = (hasil_fitur - minimum)/(maksimum - minimum)
    data_training = data_normalisasi[:int(persentase_data_training*len(data_normalisasi)/100)]
    data_testing = data_normalisasi[int(persentase_data_training*len(data_normalisasi)/100):]
    
    #Training
    bobot = np.random.rand(banyak_hidden_neuron, banyak_fitur)
    bias = np.random.rand(banyak_hidden_neuron)
    h = 1/(1 + np.exp(-(np.dot(data_training[:, :banyak_fitur], np.transpose(bobot)) + bias)))
    h_plus = np.dot(np.linalg.inv(np.dot(np.transpose(h),h)),np.transpose(h))
    output_weight = np.dot(h_plus, data_training[:, banyak_fitur])
    
    #Testing
    h = 1/(1 + np.exp(-(np.dot(data_testing[:, :banyak_fitur], np.transpose(bobot)) + bias)))
    predict = np.dot(h, output_weight)
    predict = predict * (maksimum - minimum) + minimum
    
    #MAPE
    aktual = np.array(hasil_fitur[int(persentase_data_training*len(data_normalisasi)/100):, banyak_fitur])
    mape = np.sum(np.abs(((aktual - predict)/aktual)*100))/len(predict)
    return render_template('bigdataApps.html', {
        'y_aktual' : list(aktual),
        'y_prediksi' : list(predict),
        'mape' : mape
        })
  else:
    return render_template('bigdataApps.html')

if __name__ == '__main__':
  db.create_all()
  app.secret_key = 'webapp'
  app.run()