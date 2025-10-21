# Import library yang diperlukan
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# 1. Set URI untuk menyimpan eksperimen MLflow secara lokal
# Ini akan membuat folder 'mlruns' di direktori yang sama
mlflow.set_tracking_uri("file:./mlruns")

# 2. Muat dataset
df = pd.read_csv('heart_preprocessed.csv')

# 3. Pisahkan fitur (X) dan target (y)
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# 4. Bagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Mulai eksperimen MLflow
# autolog() akan secara otomatis mencatat parameter, metrik, dan model
mlflow.autolog()

with mlflow.start_run():
    
    # Inisialisasi dan latih model
    model = LogisticRegression(max_iter=1000) # max_iter ditambah agar model konvergen
    model.fit(X_train, y_train)
    
    # Lakukan prediksi pada data uji
    y_pred = model.predict(X_test)
    
    # Hitung akurasi 1
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model: Logistic Regression")
    print(f"Accuracy: {accuracy}")
    
print("\nEksperimen selesai. Cek folder 'mlruns' dan jalankan 'mlflow ui' di terminal untuk melihat hasilnya.")

