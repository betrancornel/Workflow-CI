# Import library yang diperlukan
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# Baris mlflow.set_tracking_uri DIHAPUS karena tidak diperlukan saat dijalankan oleh GitHub Actions.

# 1. Muat dataset
df = pd.read_csv('heart_preprocessed.csv')

# 2. Pisahkan fitur (X) dan target (y)
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# 3. Bagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Aktifkan autologging MLflow
# autolog() akan secara otomatis mencatat semua parameter dan metrik ke dalam run
# yang sudah dimulai oleh perintah 'mlflow run'.
mlflow.autolog()

# Blok 'with mlflow.start_run()' DIHAPUS.
# Kode di bawah ini sekarang dijalankan langsung tanpa blok 'with'.

# Inisialisasi dan latih model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Lakukan prediksi pada data uji
y_pred = model.predict(X_test)

# Hitung akurasi
accuracy = accuracy_score(y_test, y_pred)

print(f"Model: Logistic Regression")
print(f"Accuracy: {accuracy}")

print("\nPelatihan model selesai.")