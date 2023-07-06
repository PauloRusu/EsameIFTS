import hashlib
import PySimpleGUI as sg
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tkinter import filedialog
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import sqlite3 

# Configurazione del database SQLite
conn = sqlite3.connect('users.db')
conn.execute('CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)')
conn.close()

# Variabili globali per la codifica delle caratteristiche categoriche
label_encoders = []
onehot_encoders = []
model = None

# Funzione per l'hashing della password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Funzione per la registrazione di un nuovo utente
def register_user(username, password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username=?', (username,))
    existing_user = cursor.fetchone()
    if existing_user:
        conn.close()
        return False
    
    cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hash_password(password)))
    conn.commit()
    conn.close()
    return True

# Funzione per il login dell'utente
def login_user(username, password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username=? AND password=?', (username, hash_password(password)))
    user = cursor.fetchone()
    conn.close()
    return user

# Funzione per caricare il DataFrame completo dal database
def load_df_from_db():
    conn = sqlite3.connect('users.db')
    df = pd.read_sql_query('SELECT * FROM data', conn)
    conn.close()
    return df

# Funzione per caricare il file CSV
def load_csv(file_path):
    df = pd.read_csv(file_path, delimiter=';')
    return df

# Funzione per addestrare il modello
def train_model(data):
    X = data.drop('QualityTest', axis=1)
    y = data['QualityTest']

    # Codifica delle caratteristiche categoriche
    global label_encoders, onehot_encoders
    label_encoders = []
    onehot_encoders = []
    for col in X.columns:
        if X[col].dtype == 'object':
            label_encoder = LabelEncoder()
            X[col] = label_encoder.fit_transform(X[col])
            label_encoders.append((col, label_encoder))

            onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            X_encoded = onehot_encoder.fit_transform(X[col].values.reshape(-1, 1))
            onehot_encoders.append((col, onehot_encoder))

            # Aggiunta delle nuove colonne codificate al DataFrame
            for i in range(X_encoded.shape[1]):
                X[col + '_' + str(i)] = X_encoded[:, i]

    # Rimozione delle colonne originali codificate
    X = X.drop(columns=[col for col, _ in label_encoders])

 # Aggiunta di "global model" per riferirsi alla variabile globale
    global model 
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

# Funzione per testare il modello
def test_model(data):
    X = data.copy()  # Crea una copia del DataFrame di test
    X = data.drop('QualityTest', axis=1)
    

    # Codifica delle caratteristiche categoriche
    for col, label_encoder in label_encoders:
        if col in X.columns:
            X[col] = label_encoder.transform(X[col])

    for col, onehot_encoder in onehot_encoders:
        if col in X.columns:
            X_encoded = onehot_encoder.transform(X[col].values.reshape(-1, 1))
            for i in range(X_encoded.shape[1]):
                X[col + '_' + str(i)] = X_encoded[:, i]

    # Rimozione delle colonne originali codificate
    X = X.drop(columns=[col for col, _ in label_encoders])

    y_pred = model.predict(X)
    accuracy = accuracy_score(data['QualityTest'], y_pred)
    return accuracy, y_pred

def plot_quality_test(df, y_pred):
    if y_pred is not None:
        df['QualityTest_Prediction'] = y_pred 
    plt.figure(figsize=(8, 6))
    material_values = df['Materiale'].unique()
    uv_filter_values = df['ProtezioneUV'].unique()
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    markers = ['o', 's', 'v', 'P', 'X', '*', '+']
    for material, color, marker in zip(material_values, colors, markers):
        for uv_filter in uv_filter_values:
            mask = (df['Materiale'] == material) & (df['ProtezioneUV'] == uv_filter)
            plt.scatter(df.loc[mask, 'Materiale'], df.loc[mask, 'ProtezioneUV'], color=color, marker=marker, label=f"{material} - {uv_filter}")

    plt.xlabel('PropertyA')
    plt.ylabel('PropertyB')
    plt.title('Quality Test Results')
    plt.legend()
    plt.show()


def open_file_dialog(window):
    root = sg.tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])
    if file_path:
        df = load_csv(file_path)
        conn = sqlite3.connect('users.db')
        df.to_sql('data', conn, if_exists='replace', index=False)
        conn.close()
        sg.popup('File uploaded and saved to the database successfully!')
    else:
        sg.popup('No file selected')

def open_file_dialog_prediction(window):
    root = sg.tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])
    if file_path:
        df = load_csv(file_path)
        conn = sqlite3.connect('users.db')
        df_test = pd.read_sql_query('SELECT * FROM data', conn)
        conn.close()
        accuracy, y_pred = test_model(df_test)
        sg.popup(f"Accuracy: {accuracy}")
        plot_quality_test(df_test, y_pred)
    else:
        sg.popup('No file selected')

def login_layout():
    sg.theme('DefaultNoMoreNagging')
    layout = [
        [sg.Text('Username')],
        [sg.Input(key='username')],
        [sg.Text('Password')],
        [sg.Input(key='password', password_char='*')],
        [sg.Button('Login'), sg.Button('Register')]
    ]
    return sg.Window('Login', layout, finalize=True)

def dashboard_layout(username):
    sg.theme('DefaultNoMoreNagging')
    layout = [
        [sg.Text(f'Logged in as: {username}')],
        [sg.Button('Upload File')],
        [sg.Button('Start Analysis')],
        [sg.Button('View Results')],
        [sg.Button('Visualize All')],
        [sg.Button('Logout')]
    ]
    return sg.Window('Dashboard', layout, finalize=True)

def main():
    login_window = login_layout()
    dashboard_window = None
    df_train = None

    while True:
        window, event, values = sg.read_all_windows()

        if window == login_window and event == sg.WINDOW_CLOSED:
            break

        if window == login_window and event == 'Login':
            username = values['username']
            password = values['password']
            user = login_user(username, password)
            if user:
                login_window.hide()
                dashboard_window = dashboard_layout(username)
            else:
                sg.popup('Invalid username or password')

        if window == login_window and event == 'Register':
            username = values['username']
            password = values['password']
            registered = register_user(username, password)
            if registered:
                sg.popup('Registration successful. Please log in.')
            else:
                sg.popup('Username already exists. Please choose a different username.')

        if window == dashboard_window and event == sg.WINDOW_CLOSED:
            break

        if window == dashboard_window and event == 'Upload File':
            open_file_dialog(window)

        if window == dashboard_window and event == 'Start Analysis':
            conn = sqlite3.connect('users.db')
            df_train = pd.read_sql_query('SELECT * FROM data', conn)
            conn.close()
            train_model(df_train)
            sg.popup('Model trained successfully!')

        if window == dashboard_window and event == 'View Results':
            df_test = load_df_from_db()
            if df_test is not None and not df_test.empty and model is not None:
                accuracy, y_pred = test_model(df_test)
                sg.popup(f"Accuracy: {accuracy}")
                plot_quality_test(df_test, y_pred)
            else:
                sg.popup('Please upload a file and train the model first')

        if window == dashboard_window and event == 'Visualize All':
            conn = sqlite3.connect('users.db')
            df = pd.read_sql_query('SELECT * FROM data', conn)
            conn.close()
            plot_quality_test(df, None)

        if window == dashboard_window and event == 'Logout':
            dashboard_window.hide()
            login_window.un_hide()

    login_window.close()
    if dashboard_window:
        dashboard_window.close()

if __name__ == '__main__':
    main()