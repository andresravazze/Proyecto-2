import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
import pickle
import numpy as np

# Cargar modelo
import tensorflow as tf
new_model = tf.keras.models.load_model('model5.keras')
scaler = pickle.load(open("scaler.pkl", 'rb'))
new_model.summary()
print(new_model.summary())

# Iniciar la app de Dash
app = dash.Dash(__name__)

# Cargar los datos iniciales
data = pd.read_csv('new_data.csv')

# Layout del tablero
app.layout = html.Div([
    html.H1("Evaluación de Suscripción de Depósito"),

    # Entradas de usuario
    html.Div([
        html.Label("Que edad tienes"),
        dcc.Input(id='age', type='number', value=1),

        html.Label("tiene un credito en default"),
        dcc.Dropdown(
            id='default',
            options=[
                {'label': 'Sí', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=0
        ),
        
        html.Label("tiene un prestamo en mora"),
        dcc.Dropdown(
            id='housing',
            options=[
                {'label': 'Sí', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=0
        ),

        html.Label("Tiene algun prestamo"),
        dcc.Dropdown(
            id='loan',
            options=[
                {'label': 'Sí', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=0
        ),

        html.Label("ultimo dia de contacto del mes"),
        dcc.Input(id='day', type='number', min=1, max=31, value=5),

        html.Label("ultima duracion del contacto"),
        dcc.Input(id='duration', type='number', value=100),

        html.Label("numero de contactos en la ultima campaña"),
        dcc.Input(id='campaign', type='number', value=1),

        html.Label("numero de dias que han pasado desde el ultimo contacto"),
        dcc.Input(id='pdays', type='number', value=-1),

        html.Label("saldo"),
        dcc.Input(id='balance', type='number', value=1000),

        html.Label("Cual es su trabajo"),
        dcc.Dropdown(
            id='job',
            options=[
                {'label': 'admin.', 'value': 'admin.'},
                {'label': 'unknown', 'value': 'unknown'},
                {'label': 'unemployed', 'value': 'unemployed'},
                {'label': 'management', 'value': 'management'},
                {'label': 'housemaid', 'value': 'housemaid'},
                {'label': 'entrepreneur', 'value': 'entrepreneur'},
                {'label': 'student', 'value': 'student'},
                {'label': 'blue-collar', 'value': 'blue-collar'},
                {'label': 'self-employed', 'value': 'self-employed'},
                {'label': 'retired', 'value': 'retired'},
                {'label': 'technician', 'value': 'technician'},
                {'label': 'services', 'value': 'services'}
            ],
            value='unknown'
        ),

        html.Label("Esta divorciado?"),
        dcc.Dropdown(
            id='marital_divorced',
            options=[
                {'label': 'Sí', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=0
        ), 
        
        html.Label("Esta Casado"),
        dcc.Dropdown(
            id='marital_married',
            options=[
                {'label': 'Sí', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=0
        ),
        html.Label("tiene educacion primaria"),
        dcc.Dropdown(
            id='education_primary',
            options=[
                {'label': 'Sí', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=0
        ),

        html.Label("teiene educacion secundaria"),
        dcc.Dropdown(
            id='education_tertiary',
            options=[
                {'label': 'Sí', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=0
        ),

        html.Label("Medio de comunicacion"),
        dcc.Dropdown(
            id='contact',
            options=[
                {'label': 'Unknown', 'value': 'unknown'},
                {'label': 'Telephone', 'value': 'telephone'},
                {'label': 'Cellular', 'value': 'cellular'}
            ],
            value='unknown'
        ),

        html.Label("Resultado de la campaña de contacto"),
        dcc.Dropdown(
            id='poutcome',
            options=[
                {'label': 'Unknown', 'value': 'unknown'},
                {'label': 'Other', 'value': 'other'},
                {'label': 'Failure', 'value': 'failure'},
                {'label': 'Success', 'value': 'success'}
            ],
            value='unknown'
        ),

        html.Label("Fue contactado "),
        dcc.Dropdown(
            id='contacted',
            options=[
                {'label': 'Sí', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=0
        ),

        html.Button('Predecir Suscripción', id='predict-button', n_clicks=0)
    ], style={'display': 'flex', 'flex-direction': 'column', 'width': '30%'}),

    # Resultado de predicción
    html.Div(id='prediction-output', style={'margin-top': '20px'}),
    html.Div(id='income-output', style={'margin-top': '20px'}),



])

# Callback para predecir suscripción y graficar ingresos
@app.callback(
    Output('prediction-output', 'children'),
    Output('income-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('age', 'value'),
    State('default', 'value'),
    State('housing', 'value'),
    State('loan', 'value'),
    State('day', 'value'),
    State('duration', 'value'),
    State('campaign', 'value'),
    State('pdays', 'value'),
    State('balance', 'value'),
    State('job', 'value'),
    State('marital_divorced', 'value'),
    State('marital_married', 'value'),
    State('education_primary', 'value'),
    State('education_tertiary', 'value'),
    State('contact', 'value'),
    State('poutcome', 'value'),
    State('contacted', 'value')
)
def predict_subscription(n_clicks, age, default, housing, loan, day, duration, campaign, pdays, balance, 
                         job, marital_divorced, marital_married, education_primary, education_tertiary, contact, poutcome, contacted):
    # Crear el DataFrame con los valores ingresados, incluyendo las columnas de trabajo, contacto y poutcome
    job_options = ["admin.", "unknown", "unemployed", "management", "housemaid", 
                   "entrepreneur", "student", "blue-collar", "self-employed", 
                   "retired", "technician", "services"]
    contact_options = ["unknown", "telephone", "cellular"]
    poutcome_options = ["unknown", "other", "failure", "success"]

    # Inicializamos un diccionario con todas las categorías de trabajo, contacto y poutcome en 0
    user_data = {f'job_{j}': 0 for j in job_options}
    user_data.update({f'contact_{c}': 0 for c in contact_options})
    user_data.update({f'poutcome_{p}': 0 for p in poutcome_options})
    
    # Establecemos a 1 solo la categoría seleccionada por el usuario en job, contact y poutcome
    user_data[f'job_{job}'] = 1
    user_data[f'contact_{contact}'] = 1
    user_data[f'poutcome_{poutcome}'] = 1
    
    # Añadir las demás características del usuario
    user_data.update({
        'age': age,
        'default': default,
        'housing': housing,
        'loan': loan,
        'day': day,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'balance': balance,
        'marital_divorced': marital_divorced,
        'marital_married': marital_married,
        'education_primary': education_primary,
        'education_tertiary': education_tertiary,
        
        'contacted': contacted
    })
    
    # Convertimos el diccionario a un DataFrame de una fila
    user_data = pd.DataFrame([user_data])
    
# Verificar tipos de datos
    print("Tipos de datos en user_data:")
    print(user_data.dtypes)
    
    # Convertir tipos de datos a float32
    user_data = user_data.astype(np.float32)


    # Verificar si hay valores NaN
    if user_data.isnull().values.any():
        print("Hay valores NaN en user_data. Por favor, revisa los datos.")
        return "Error: hay valores NaN en los datos ingresados."

    # Imprimir los datos que se van a usar para la predicción
    print("Datos de entrada para la predicción:")
    print(user_data)

    # Realizar la predicción
    user_data = user_data[['age','default','balance','housing','loan','day','duration','campaign','pdays','job_admin.','job_blue-collar','job_entrepreneur','job_housemaid','job_management','job_retired','job_self-employed','job_services','job_student','job_technician','job_unemployed','job_unknown','marital_divorced','marital_married','education_primary','education_tertiary','contact_cellular','contact_telephone','contact_unknown','poutcome_failure','poutcome_other','poutcome_success','poutcome_unknown','contacted']]
    user_data=scaler.transform(user_data)
    prob = new_model(user_data)[:, 0].numpy()  # Asumimos que devuelve la probabilidad de suscripción
    print(prob)
    prediction = "Apto para suscripción" if prob >= 0.2 else "No apto para suscripción"
    prediction_text = f"Resultado de predicción: {prediction} (Probabilidad: {prob[0]:.5f})"
    
    # Gráfica de ingresos (average_balance)
    price = 0.2 * balance
    cost_contact =3
    tp =843
    fp = 1090
    fn = 248
    income = ((price - cost_contact) * (tp / 100)) + (-cost_contact * (fp / 100)) + (-price * (fn / 100))

    return prediction_text, f"El ingreso especifico de este usuario es {income}"

if __name__ == '__main__':
    app.run_server(debug=True)

print(user_data.dtypes)
