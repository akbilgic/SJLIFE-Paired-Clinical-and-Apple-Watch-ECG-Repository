import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def load_data():
    apple_ecg = np.load('./500hz_10sec_apple/500hz_10sec_apple_ecg_132.npy')
    X_values = apple_ecg.flatten()

    clinical_ecg = np.load('first_lead_clinical/first_lead_clinical_ecg_132.npy')
    y_values = clinical_ecg.flatten()

    return X_values, y_values

def prepare_data(X, y, test_size=0.2, random_state=42):
    # sklearn modelleri girişi 2D array olarak ister:
    X = X.reshape(-1, 1)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_linear_regression(X_train, y_train):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr

def train_mlp(X_train, y_train, hidden_layers=(50,50), max_iter=500):
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        max_iter=max_iter,
        random_state=42
    )
    mlp.fit(X_train, y_train)
    return mlp

def evaluate_model(model, X_test, y_test, name="Model"):
    y_pred = model.predict(X_test)
    mse  = mean_squared_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    print(f"[{name}] MSE: {mse:.4f} | R²: {r2:.4f}")
    return mse, r2

def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model '{filename}' olarak kaydedildi.")

def load_model(filename):
    return joblib.load(filename)


import matplotlib.pyplot as plt
def plot_results(y_true, y_pred, title="Model Predictions"):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Gerçek ECG', linewidth=1)
    plt.plot(y_pred, label='Tahmin Edilen ECG', alpha=0.7, linewidth=1)
    plt.title(title)
    plt.xlabel('Zaman')
    plt.ylabel('ECG Değeri')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # 1) Veriyi yükle
    X, y = load_data()

    # 2) Eğitim / test olarak böl
    X_train, X_test, y_train, y_test = prepare_data(X, y)

    # 3) Doğrusal Regresyon eğit
    lr_model = train_linear_regression(X_train, y_train)
    evaluate_model(lr_model, X_test, y_test, name="LinearRegression")

    # 4) MLPRegressor eğit
    mlp_model = train_mlp(X_train, y_train)
    evaluate_model(mlp_model, X_test, y_test, name="MLPRegressor")

    # 5) En iyi modeli kaydetmek isterseniz:
    # (örneğin MLP daha yüksek R² verdi ise onu kaydedin)
    save_model(mlp_model, "best_model.pkl")

    # 6) Modeli yükle:
    loaded_model = load_model("best_model.pkl")

    # 7) Modeli kullanarak tahmin yap
    new_input_data = np.load('./500hz_10sec_apple/500hz_10sec_apple_ecg_133.npy')
    new_input_data = new_input_data.flatten().reshape(-1, 1)

    new_real_data = np.load('first_lead_clinical/first_lead_clinical_ecg_133.npy')
    new_real_data = new_real_data.flatten()

    y_pred = loaded_model.predict(new_input_data)
    evaluate_model(loaded_model, new_input_data, new_real_data, name="LoadedModel")
    #print(f"Yüklenen model ile tahmin: {y_pred}")

    # 8) Grafik oluşturma
    plot_results(new_real_data, y_pred, title="Yüklenen Model Tahminleri")