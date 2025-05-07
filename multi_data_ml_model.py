import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

def load_all_data(apple_dir, clinical_dir):
    """
    Load and flatten all .npy files from two directories,
    then concatenate into X and y arrays.
    Assumes matching filenames (e.g. ..._132.npy in both).
    """
    # find and sort all .npy paths
    apple_paths    = sorted(glob.glob(os.path.join(apple_dir, '*.npy')))
    clinical_paths = sorted(glob.glob(os.path.join(clinical_dir, '*.npy')))

    if len(apple_paths) != len(clinical_paths):
        print(f"Warning: {len(apple_paths)} apple files vs {len(clinical_paths)} clinical files")

    X_list, y_list = [], []
    for a_path, c_path in zip(apple_paths, clinical_paths):
        # load, flatten, and collect
        X_list.append(np.load(a_path).flatten())
        y_list.append(np.load(c_path).flatten())

    # concatenate into one big 1D array each
    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    return X, y

def prepare_data(X, y, test_size=0.2, random_state=42):
    # reshape X to 2D as required by scikit-learn
    X = X.reshape(-1, 1)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_linear_regression(X_train, y_train):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr

def train_mlp(X_train, y_train, hidden_layers=(50, 50), max_iter=500):
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
    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print(f"[{name}] MSE: {mse:.6f} | R²: {r2:.6f}")
    return mse, r2, y_pred

def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved as '{filename}'")

def load_model(filename):
    return joblib.load(filename)

def plot_results(y_true, y_pred, title="Model Predictions"):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true,  label='True ECG',      linewidth=1)
    plt.plot(y_pred,  label='Predicted ECG',  alpha=0.7, linewidth=1)
    plt.title(title)
    plt.xlabel('Time (samples)')
    plt.ylabel('ECG value')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Directories containing your .npy files
    APPLE_DIR    = './500hz_10sec_apple'
    CLINICAL_DIR = './first_lead_clinical'

    # 1) Load all data
    X, y = load_all_data(APPLE_DIR, CLINICAL_DIR)

    # 2) Split into train/test
    X_train, X_test, y_train, y_test = prepare_data(X, y)

    # 3) Train & evaluate Linear Regression
    lr_model = train_linear_regression(X_train, y_train)
    _, _, lr_pred = evaluate_model(lr_model, X_test, y_test, name="LinearRegression")

    # 4) Train & evaluate MLPRegressor
    mlp_model = train_mlp(X_train, y_train)
    _, _, mlp_pred = evaluate_model(mlp_model, X_test, y_test, name="MLPRegressor")

    # 5) Decide best model (here by R²) and save
    _, lr_r2 = mean_squared_error(y_test, lr_pred), r2_score(y_test, lr_pred)
    _, mlp_r2 = mean_squared_error(y_test, mlp_pred), r2_score(y_test, mlp_pred)
    best_model = mlp_model if mlp_r2 > lr_r2 else lr_model
    best_name  = "MLPRegressor" if mlp_r2 > lr_r2 else "LinearRegression"
    save_model(best_model, f"./best_model/{best_name}_best_model.pkl")

    # 6) Load best model
    loaded_model = load_model(f"./best_model/{best_name}_best_model.pkl")

    # 7) Optional: predict on a new file pair
    new_X = np.load(os.path.join(APPLE_DIR,    '500hz_10sec_apple_ecg_133.npy')).flatten().reshape(-1,1)
    new_y = np.load(os.path.join(CLINICAL_DIR, 'first_lead_clinical_ecg_133.npy')).flatten()
    _, _, new_pred = evaluate_model(loaded_model, new_X, new_y, name="LoadedModel")

    # 8) Plot predictions vs truth for the new data
    plot_results(new_y, new_pred, title=f"Predictions by {best_name} on New ECG Sample")
