from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, LoginManager, login_user, login_required, logout_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
from flask_migrate import Migrate
import openai
from flask import Flask, render_template, request, jsonify, redirect, url_for
from nltk.tokenize import word_tokenize  # Importez word_tokenize
from nltk.corpus import stopwords  # Ajoutez l'importation pour les stop words
from openai import OpenAIError
#chatbot
from openai import OpenAIError
from flask_cors import CORS
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
llm = OpenAI()

#survey
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import base64
from io import BytesIO






app = Flask(__name__, template_folder='template')

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_BINDS'] = {
    'second_db': 'sqlite:///site.db'
}
app.config['SECRET_KEY'] = 'thisisasecretkey'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)





bcrypt = Bcrypt(app)
migrate = Migrate(app, db)


login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'



# Load the KNN model
model_path_knn = "model/knn_model.joblib"
knn = joblib.load(model_path_knn)

# Load the Linear Regression model
lr_model = joblib.load("model/linear_regression_model.joblib")

# Load the LSTM model
lstm_model = load_model('model/lstm_model.h5')
lstm_model.compile(run_eagerly=True)
scaler = MinMaxScaler()

def predict_knn(features):
    input_data_knn = pd.DataFrame([features], columns=['High', 'Low', 'Open_Price'])
    prediction_knn = knn.predict(input_data_knn)
    return prediction_knn[0]

def predict_lr(stock_price):
    input_df_lr = pd.DataFrame({'Stock Price': [stock_price]})
    imputer = SimpleImputer(strategy='mean')
    input_df_lr_no_nan = imputer.fit_transform(input_df_lr)
    prediction_lr = lr_model.predict(input_df_lr_no_nan)
    return prediction_lr[0]

def predict_lstm(start_date, end_date):
    tf.config.run_functions_eagerly(False)

    num_days = (end_date - start_date).days + 1
    input_data = np.arange(num_days).reshape(-1, 1)
    normalized_input_data = scaler.fit_transform(input_data)
    predicted_values = lstm_model.predict(normalized_input_data)

    tf.config.run_functions_eagerly(True)

    denormalized_predictions = scaler.inverse_transform(predicted_values)

    return denormalized_predictions.flatten().tolist()

def generate_date_range(start_date, end_date):
    date_range = [str(start_date + timedelta(days=i)) for i in range((end_date-start_date).days + 1)]
    return date_range

@app.route('/')
def home():
     return render_template('visitor.html')
@app.route('/client')
def client_home():
     return render_template('ClientInterface.html')
@app.route('/Close')
def close_home():
    return render_template('predictclose.html')
@app.route('/Dividend')
def Dividend_home():
    return render_template('predictDividend.html')
@app.route('/sentiment')
def sentiment():
    return render_template('sentiment.html')
# KNN Routes
@app.route('/knn')
def knn_home():
    return render_template('predictclose.html')
@app.route('/index')
def index_home():
     return render_template('index.html')
@app.route('/predict_knn', methods=['POST'])
def predict_knn_route():
    try:
        features_knn = [
            float(request.form['High']),
            float(request.form['Low']),
            float(request.form['Open_Price']),
        ]

        prediction_knn = predict_knn(features_knn)

        return render_template('predictclose.html', prediction=prediction_knn)

    except Exception as e:
        error_message = f"Error: {str(e)}"
        return render_template('predictclose.html', error=error_message)

# Linear Regression Routes
@app.route('/linear_regression')
def linear_regression_home():
    return render_template('predictDividend.html')

@app.route('/predict_lr', methods=['POST'])
def predict_lr_route():
    try:
        stock_price = float(request.form['Stock_Price'].replace(',', ''))
        prediction_lr = predict_lr(stock_price)

        return render_template('predictDividend.html', prediction_lr=prediction_lr)

    except Exception as e:
        app.logger.error(f"Exception: {str(e)}")
        return render_template('predictDividend.html', error_lr="Error occurred. Please check your input.")

# LSTM Routes
@app.route('/lstm')
def lstm_home():
    return render_template('predictclose1.html')

@app.route('/predict_lstm', methods=['POST'])
def predict_lstm_route():
    try:
        start_date_str = request.form['start_date']
        end_date_str = request.form['end_date']

        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()

        predicted_values = predict_lstm(start_date, end_date)
        date_range = generate_date_range(start_date, end_date)

        return jsonify({'dates': date_range, 'values': predicted_values})

    except Exception as e:
        app.logger.error(f"Exception: {str(e)}")
        return jsonify({'error': 'Error occurred. Please check your input.'})





@app.route('/powerBi')
def power_home():
    return render_template('dashboard.html')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)


class RegisterForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField('Register')

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                'That username already exists. Please choose a different one.')


class LoginForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField('Login')



@app.route('/check_username/<username>', methods=['GET'])
def check_username(username):
    existing_user_username = User.query.filter_by(username=username).first()
    return jsonify({'exists': existing_user_username is not None})

from flask_login import login_user  # Assuming you're using Flask-Login

from flask import flash

from flask import render_template, flash

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    login_successful = None  # Variable to indicate login success

    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()

        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            login_successful = True  # Set the variable to True for successful login
            flash('Login successful', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
            login_successful = False  # Set the variable to False for unsuccessful login

    return render_template('login.html', form=form, login_successful=login_successful)

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('ClientInterface.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('login.html', form=form)
# ____________________________________________________________________


# Mettez ici votre nouvelle clé secrète OpenAI



def get_gpt3_response(prompt):
    try:
        # Vérifiez la longueur des tokens dans la requête
        token_count = len(prompt.split())
        print(f"Nombre de tokens dans la requête : {token_count}")

        # Vérifiez si le nombre de tokens dépasse la limite
        if token_count > 4096:
            return "La requête dépasse la limite de tokens. Veuillez raccourcir votre texte."

        # Appel à l'API OpenAI
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150
        )

        if response.choices and response.choices[0].text:
            return response.choices[0].text.strip()
        else:
            return "Une réponse valide n'a pas été générée."

    except OpenAIError as e:
        # Gestion des erreurs OpenAI
        print(f"Erreur OpenAI : {e}")
        return "Une erreur s'est produite lors de la génération de la réponse."
    except Exception as e:
        # Gestion des autres erreurs
        print(f"Erreur inattendue : {e}")
        return "Une erreur inattendue s'est produite."

@app.route('/ask', methods=['POST'])
def ask_question():
    user_input = request.form['user_input']
    bot_response = get_gpt3_response(user_input)
    return render_template('ClientInterface.html', user_input=user_input, bot_response=bot_response)

# ____________________________________________________________________
# ____________________________________________________________________

# Chargement du modèle SVM
svm_model = joblib.load("model/svm_model.joblib")
tfidf_vectorizer = joblib.load("model/tfidf_vectorizer.joblib")
stop_words = set(stopwords.words('english'))

# ...

# Route pour la prédiction du sentiment
def preprocess_text(text):
    words = word_tokenize(text)
    filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
    return ' '.join(filtered_words)
@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    try:
      
        # Récupérer le texte du formulaire
        text_input = request.form['text_input']

        # Prétraitement du texte
        processed_text = preprocess_text(text_input)

        # Vectorisation avec le vecteur TF-IDF
        text_tfidf = tfidf_vectorizer.transform([processed_text])

        # Prédire avec le modèle SVM
        prediction = svm_model.predict(text_tfidf)

        # Rendre le résultat à la page HTML
        return render_template('sentiment.html', prediction=prediction[0])

    except Exception as e:
        # Gérer les erreurs et les afficher dans la page HTML
        error_message = f"Error: {str(e)}"
        return render_template('sentiment.html', error=error_message)

# Fonction de prétraitement des textes
def preprocess_text(text):
    words = word_tokenize(text)
    filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
    return ' '.join(filtered_words)

# ____________________________________________________________________
#chatbot

@app.route('/generate_response', methods=['POST'])
def generate_response():
    prompt = request.form['prompt']
    
    # Use OpenAI GPT-3.5 to generate a response
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=150  # You can adjust the max_tokens parameter based on your needs
    ).choices[0].text

    return render_template('ClientInterface.html', prompt=prompt, response=response)

#survey


class Investor(db.Model):
  
    id = db.Column(db.String(50), primary_key=True)
    survey1_responses = db.relationship('Survey1Response', backref='investor', lazy=True)
    survey2_responses = db.relationship('Survey2Response', backref='investor', lazy=True)

class Survey1Response(db.Model):
    
    id = db.Column(db.Integer, primary_key=True)
    investor_id = db.Column(db.String(50), db.ForeignKey('investor.id'), nullable=False)
    response = db.Column(db.String(10))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class Survey2Response(db.Model):
   
    id = db.Column(db.Integer, primary_key=True)
    investor_id = db.Column(db.String(50), db.ForeignKey('investor.id'), nullable=False)
    response = db.Column(db.String(10))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


# Function to generate and save pie chart
def generate_pie_chart(responses, title):
    df = pd.DataFrame([(response.investor_id, response.response) for response in responses],
                      columns=['Investor ID', 'Response'])
    response_counts = df['Response'].value_counts()

    # Define custom colors
    colors = ['#808080', '#101010', '#FF0000']

    plt.figure(figsize=(6, 6))
    plt.pie(response_counts, labels=response_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title(title)

    img_buf = BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    img_buf.seek(0)
    img_b64 = base64.b64encode(img_buf.read()).decode('utf-8')

    return img_b64

# Route to display Survey 1
@app.route('/survey1', methods=['GET', 'POST'])
def survey1():
    if request.method == 'POST':
        investor_id = request.form['investor_id']
        response = request.form['response']

        investor = Investor.query.get(investor_id)
        if investor is None:
            investor = Investor(id=investor_id)
            db.session.add(investor)

        survey_response = Survey1Response(investor=investor, response=response)
        db.session.add(survey_response)
        db.session.commit()

    # Fetch data for both surveys
    chart1 = generate_pie_chart(Survey1Response.query.all(), 'Survey 1 Responses')
    chart2 = generate_pie_chart(Survey2Response.query.all(), 'Survey 2 Responses')

    return render_template('dashboard.html', chart1=chart1, chart2=chart2)

# Route to display Survey 2
@app.route('/survey2', methods=['GET', 'POST'])
def survey2():
    if request.method == 'POST':
        investor_id = request.form['investor_id']
        response = request.form['response']

        investor = Investor.query.get(investor_id)
        if investor is None:
            investor = Investor(id=investor_id)
            db.session.add(investor)

        survey_response = Survey2Response(investor=investor, response=response)
        db.session.add(survey_response)
        db.session.commit()

    # Fetch data for both surveys
    chart1 = generate_pie_chart(Survey1Response.query.all(), 'Survey 1 Responses')
    chart2 = generate_pie_chart(Survey2Response.query.all(), 'Survey 2 Responses')

    return render_template('dashboard.html', chart1=chart1, chart2=chart2)

if __name__ == '__main__':
    app.run(debug=True)

