import streamlit as st
import pandas as pd
import statsmodels.api as sm
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =========================
# Load Models & Encoders
# =========================
def load_glm_model_info(path):
    model = sm.load(path)
    try:
        df = model.model.data.frame
        day_categories = df['day'].cat.categories.tolist()
        product_categories = df['product_category'].cat.categories.tolist()
        harga_median = df['unit_price'].median()
    except Exception:
        day_categories = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        product_categories = ['kopi', 'non-kopi']
        harga_median = 10000
    return model, day_categories, product_categories, harga_median

def load_xgb_model(path):
    bundle = joblib.load(path)
    return bundle['model'], bundle['le_day'], bundle['le_product']

# =========================
# Preprocessing
# =========================
def preprocess_data(file, day_categories, product_categories):
    if file.name.endswith('.csv'):
        data = pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        data = pd.read_excel(file)
    else:
        raise ValueError("Hanya mendukung file .csv dan .xlsx")

    data['transaction_date'] = pd.to_datetime(data['transaction_date'])
    data['day'] = pd.Categorical(data['transaction_date'].dt.day_name(), categories=day_categories)
    data['product_category'] = pd.Categorical(data['product_category'], categories=product_categories)
    return data

# =========================
# Prediction Functions
# =========================
def predict_glm(model, data):
    data['predicted_qty_glm'] = model.predict(data)
    return data

def predict_xgb(model, data, le_day, le_product):
    data = data[data['day'].isin(le_day.classes_) & data['product_category'].isin(le_product.classes_)]
    data['day_encoded'] = le_day.transform(data['day'])
    data['product_category_encoded'] = le_product.transform(data['product_category'])
    features = data[['day_encoded', 'product_category_encoded', 'unit_price']]
    data['predicted_qty_xgb'] = model.predict(features)
    return data

# =========================
# Diskon & Output
# =========================
def hitung_diskon(qty, threshold=1.5, max_diskon=0.2):
    if qty < threshold:
        diskon = (1 - qty / threshold) * max_diskon
        return round(min(diskon, max_diskon), 2)
    return 0.0

def apply_diskon(data, pred_col='predicted_qty_glm'):
    data['diskon'] = data[pred_col].apply(hitung_diskon)
    data['harga_setelah_diskon'] = data['unit_price'] * (1 - data['diskon'])
    return data

# =========================
# Evaluation & Visualization
# =========================
def eval_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return mse, mse**0.5, mean_absolute_error(y_true, y_pred), r2_score(y_true, y_pred)

def plot_predictions(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['transaction_qty'], label='Aktual', linewidth=2)
    ax.plot(data['predicted_qty_glm'], label='GLM', linewidth=1.5)
    ax.plot(data['predicted_qty_xgb'], label='XGBoost', linewidth=1.5)
    ax.set(title='Prediksi vs Aktual', xlabel='Index', ylabel='Jumlah Transaksi')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)

# =========================
# Helper: Hari & Tampilan
# =========================
def tampilkan_extreme_prediksi(data, hari, model_col='predicted_qty_glm'):
    data_hari = data[data['day'] == hari]
    if data_hari.empty:
        return None, None
    return (
        data_hari.sort_values(by=model_col).iloc[0],
        data_hari.sort_values(by=model_col, ascending=False).iloc[0]
    )

day_mapping_id = {
    'Monday': 'Senin', 'Tuesday': 'Selasa', 'Wednesday': 'Rabu', 'Thursday': 'Kamis',
    'Friday': 'Jumat', 'Saturday': 'Sabtu', 'Sunday': 'Minggu'
}
day_mapping_en = {v: k for k, v in day_mapping_id.items()}

# =========================
# Streamlit App
# =========================
st.title("Prediksi Transaksi & Evaluasi Model (GLM vs XGBoost)")

uploaded_file = st.file_uploader("Unggah data transaksi (.csv atau .xlsx)", type=["csv", "xlsx"])

if uploaded_file:
    glm_model, day_cats, prod_cats, harga_median = load_glm_model_info("model_glm_hale.sm")
    xgb_model, le_day, le_product = load_xgb_model("model_xgb_hale.pkl")

    data = preprocess_data(uploaded_file, day_cats, prod_cats)
    data = predict_glm(glm_model, data)
    data = predict_xgb(xgb_model, data
