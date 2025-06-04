import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Veriyi oku
df = pd.read_csv(r'C:\Users\User\verimaden\SeoulBikeData.csv', encoding='ISO-8859-1')
df.head()
# Veri çerçevesinin yapısı
df.info()

# Sayısal sütunların istatistikleri
df.describe()

# Sütun isimlerini göster
df.columns

# Eksik değerleri kontrol et
df.isnull().sum()
# Tarih işleme
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Weekday'] = df['Date'].dt.weekday
df['Weekend'] = df['Weekday'].apply(lambda x: 1 if x >= 5 else 0)

# Kategorik sütunları işle
df['Holiday'] = df['Holiday'].map({'Holiday': 1, 'No Holiday': 0})
df['Functioning Day'] = df['Functioning Day'].map({'Yes': 1, 'No': 0})
df.head()
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(df['Rented Bike Count'], bins=50, kde=True)
plt.title("Histogram of Rented Bike Count")

plt.subplot(1, 2, 2)
sns.boxplot(x=df['Rented Bike Count'])
plt.title("Boxplot of Rented Bike Count")

plt.tight_layout()
plt.show()

plt.figure(figsize=(11, 4))

grouped = df.groupby(['Weekday', 'Hour'])['Rented Bike Count'].mean().reset_index()

sns.lineplot(data=grouped, x='Hour', y='Rented Bike Count', hue='Weekday', palette='tab10')
plt.title("Haftanın Günlerine Göre Saatlik Ortalama Kiralama Sayısı")
plt.xlabel("Saat")
plt.ylabel("Ortalama Kiralama")
plt.legend(title='Gün (0=Pazartesi)')
plt.grid(True)
plt.show()

# Aylık saatlik ortalama
plt.figure(figsize=(11, 4))
grouped = df.groupby(['Month', 'Hour'])['Rented Bike Count'].mean().reset_index()

sns.lineplot(data=grouped, x='Hour', y='Rented Bike Count', hue='Month', palette='tab10')
plt.title("Aylara Göre Saatlik Ortalama Kiralama Sayısı")
plt.xlabel("Saat")
plt.ylabel("Ortalama Kiralama")
plt.legend(title='Gün (0=Pazartesi)')
plt.grid(True)
plt.show()


# Aylık ortalama kiralama (1–12)
monthly_avg = df.groupby('Month')['Rented Bike Count'].mean().reindex(range(1,13))

plt.figure(figsize=(8, 4))
sns.barplot(
    x=monthly_avg.index,
    y=monthly_avg.values,
    hue=monthly_avg.index,
    palette='viridis',
    legend=False
)
plt.title("Aylara Göre Ortalama Kiralanan Bisiklet Sayısı")
plt.xlabel("Ay")
plt.ylabel("Ortalama Kiralama")
plt.grid(True)
plt.show()
# Hedef ve sayısal sütunlar
target = 'Rented Bike Count'
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Kategorik ikili değişkenleri çıkar
binary_categorical = ['Holiday', 'Functioning Day', 'Weekend']
features = [col for col in numeric_cols if col != target and col not in binary_categorical]

# Her sütun için scatter plot çiz
for col in features:
    plt.figure(figsize=(7, 4))
    sns.scatterplot(data=df, x=col, y=target, alpha=0.5)
    plt.title(f"{col} vs {target}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(10, 8))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.savefig("korelasyon_matrisi.png", dpi=300)  # Görseli yüksek çözünürlükle kaydeder
plt.show()
# Kategorik sütunları belirle
categorical_cols = df.select_dtypes(include='object').columns.tolist()
categorical_cols += ['Holiday', 'Functioning Day', 'Weekend', 'Weekday', 'Month']  # numeric ama kategorik gibi davranır

for col in categorical_cols:
    if col in df.columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df, x=col, y='Rented Bike Count')
        plt.title(f"{col} - Rented Bike Count Dağılımı")
        plt.xticks(rotation=30)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Hedef ve özellikler
X = df.drop(columns=['Rented Bike Count', 'Date'])
y = df['Rented Bike Count']

# Kategorik sütunları object yap
categorical_cols = ['Seasons', 'Holiday', 'Functioning Day', 'Month', 'Weekday', 'Weekend']
for col in categorical_cols:
    X[col] = X[col].astype('object')

# Ön işlemci
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)
# Eğitim/test ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# --- RANDOM FOREST PIPELINE ---
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

rf_params = {
    'regressor__n_estimators': [100],
    'regressor__max_depth': [10, 18],
    'regressor__min_samples_split': [2, 5],
}
# --- XGBOOST PIPELINE ---
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(random_state=42, verbosity=0))
])

xgb_params = {
    'regressor__n_estimators': [100],
    'regressor__max_depth': [3, 6],
    'regressor__learning_rate': [0.1, 0.3],
}

# GridSearch
rf_search = GridSearchCV(rf_pipeline, rf_params, cv=3, scoring='r2', n_jobs=-1)
xgb_search = GridSearchCV(xgb_pipeline, xgb_params, cv=3, scoring='r2', n_jobs=-1)
# Eğit
rf_search.fit(X_train, y_train)
xgb_search.fit(X_train, y_train)
# En iyi modeller
rf_best = rf_search.best_estimator_
xgb_best = xgb_search.best_estimator_

print(rf_best)
print(xgb_best)
# Tahmin
rf_pred = rf_best.predict(X_test)
xgb_pred = xgb_best.predict(X_test)

def evaluate(name, y_test, y_pred):
    print(f"\nModel: {name}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")

evaluate("Random Forest", y_test, rf_pred)
evaluate("XGBoost", y_test, xgb_pred)
# Metrikleri hesaplayan fonksiyon
def get_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae

# Random Forest grafiği
rf_r2, rf_rmse, rf_mae = get_metrics(y_test, rf_pred)

plt.figure(figsize=(14, 5))
plt.plot(y_test.values[:100], label='Gerçek Değerler', color='black', linewidth=2)
plt.plot(rf_pred[:100], label=f'Random Forest Tahmini\nR²={rf_r2:.2f}, RMSE={rf_rmse:.2f}, MAE={rf_mae:.2f}', linestyle='--', color='red')
plt.title("Random Forest: Gerçek vs Tahmin Edilen Değerler (İlk 100 Gözlem)")
plt.xlabel("Gözlem İndeksi")
plt.ylabel("Rented Bike Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# XGBoost grafiği
xgb_r2, xgb_rmse, xgb_mae = get_metrics(y_test, xgb_pred)

plt.figure(figsize=(14, 5))
plt.plot(y_test.values[:100], label='Gerçek Değerler', color='black', linewidth=2)
plt.plot(xgb_pred[:100], label=f'XGBoost Tahmini\nR²={xgb_r2:.2f}, RMSE={xgb_rmse:.2f}, MAE={xgb_mae:.2f}', linestyle='--', color='red')
plt.title("XGBoost: Gerçek vs Tahmin Edilen Değerler (İlk 100 Gözlem)")
plt.xlabel("Gözlem İndeksi")
plt.ylabel("Rented Bike Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# Özellik adlarını al
ohe_features = rf_best.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out(categorical_cols).tolist()
numeric_features = [col for col in X.columns if col not in categorical_cols]
all_features = ohe_features + numeric_features
# Özellik önemleri - XGBoost
importances = xgb_best.named_steps['regressor'].feature_importances_
importance_df = pd.DataFrame({'Feature': all_features, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
# Görselleştir
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df.head(15), x='Importance', y='Feature', hue='Feature', palette='viridis')
plt.title('XGBoost Feature Importances (Top 15)')
plt.tight_layout()
plt.show()

# 10-fold KFold tanımı
kf = KFold(n_splits=10, shuffle=True, random_state=42)

def cross_val_metrics(model, X, y):
    r2_scores = []
    rmse_scores = []
    mae_scores = []

    for train_index, test_index in kf.split(X):
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_test_fold)

        r2_scores.append(r2_score(y_test_fold, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_test_fold, y_pred)))
        mae_scores.append(mean_absolute_error(y_test_fold, y_pred))

    print(f"R2 (mean ± std): {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    print(f"RMSE (mean ± std): {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
    print(f"MAE (mean ± std): {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
print("Random Forest CV Performance:")
cross_val_metrics(rf_best, X_train, y_train)

print("\nXGBoost CV Performance:")
cross_val_metrics(xgb_best, X_train, y_train)