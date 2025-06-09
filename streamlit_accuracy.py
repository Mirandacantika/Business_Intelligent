import streamlit as st
import pandas as pd
import statsmodels.api as sm
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Load model GLM & info ---
def load_glm_model_info(path):
    model = sm.load(path)
    try:
        day_categories = model.model.data.frame['day'].cat.categories.tolist()
        product_categories = model.model.data.frame['product_category'].cat.categories.tolist()
        harga_median = model.model.data.frame['unit_price'].median()
    except Exception:
        day_categories = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        product_categories = ['kopi', 'non-kopi']
        harga_median = 10000  # fallback default
    return model, day_categories, product_categories, harga_median

# --- Preprocessing data untuk GLM ---
def preprocess_data_glm(file, day_categories, product_categories):
    filename = file.name.lower()
    if filename.endswith('.csv'):
        data = pd.read_csv(file)
    elif filename.endswith('.xlsx'):
        data = pd.read_excel(file)
    else:
        raise ValueError("Format file tidak dikenali. Harus .csv atau .xlsx")

    data['transaction_date'] = pd.to_datetime(data['transaction_date'])
    data['day'] = data['transaction_date'].dt.day_name()
    data['day'] = pd.Categorical(data['day'], categories=day_categories)
    data['product_category'] = pd.Categorical(data['product_category'], categories=product_categories)
    return data

# --- Load model XGB + encoder ---
def load_xgb_model(path):
    xgb_bundle = joblib.load(path)
    return xgb_bundle['model'], xgb_bundle['le_day'], xgb_bundle['le_product']

# --- Prediksi GLM ---
def predict_glm_qty(model, data):
    data['predicted_qty_glm'] = model.predict(data)
    return data

# --- Prediksi XGB ---
def predict_xgb_qty(model, data, le_day, le_product):
    known_days = set(le_day.classes_)
    known_products = set(le_product.classes_)
    data = data[data['day'].isin(known_days) & data['product_category'].isin(known_products)]
    data['day_encoded'] = le_day.transform(data['day'])
    data['product_category_encoded'] = le_product.transform(data['product_category'])
    X = data[['day_encoded', 'product_category_encoded', 'unit_price']].values
    data['predicted_qty_xgb'] = model.predict(X)
    return data

# --- Fungsi hitung diskon ---
def hitung_diskon(pred_qty):
    threshold = 1.5
    diskon_maks = 0.2  # maksimal 20%
    if pred_qty < threshold:
        diskon = (1 - pred_qty / threshold) * diskon_maks
        return round(min(diskon, diskon_maks), 2)
    return 0.0

# --- Tampilkan prediksi terendah dan tertinggi ---
def tampilkan_prediksi_untuk_hari(data, hari, model_col):
    data_hari = data[data['day'] == hari]
    if data_hari.empty:
        return None, None
    return (
        data_hari.sort_values(by=model_col).head(1).iloc[0],
        data_hari.sort_values(by=model_col, ascending=False).head(1).iloc[0]
    )

# --- Apply diskon ---
def apply_diskon(data, col_predicted='predicted_qty_glm'):
    data['diskon'] = data[col_predicted].apply(hitung_diskon)
    data['harga_setelah_diskon'] = data['unit_price'] * (1 - data['diskon'])
    return data

# --- Evaluasi metrik ---
def eval_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

# --- Plot perbandingan prediksi ---
def plot_predictions(df_compare):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_compare['transaction_qty'].values, label='Aktual', linewidth=2)
    ax.plot(df_compare['predicted_qty_glm'].values, label='Prediksi GLM', linewidth=1.5)
    ax.plot(df_compare['predicted_qty_xgb'].values, label='Prediksi XGBoost', linewidth=1.5)
    ax.set_title('Perbandingan Prediksi vs Aktual', fontsize=16)
    ax.set_xlabel('Index', fontsize=12)
    ax.set_ylabel('Jumlah Transaksi', fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)

# --- Mapping hari ---
day_mapping_id = {
    'Monday': 'Senin', 'Tuesday': 'Selasa', 'Wednesday': 'Rabu', 'Thursday': 'Kamis',
    'Friday': 'Jumat'
}
day_mapping_en = {v: k for k, v in day_mapping_id.items()}

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("\U0001F4C8 Prediksi Transaksi & Evaluasi Model")

uploaded_file = st.file_uploader("\U0001F4C2 Unggah file data (.csv/.xlsx)", type=["csv", "xlsx"])

if uploaded_file:
    glm_model, day_cats, prod_cats, harga_median = load_glm_model_info("model_glm_hale.sm")
    xgb_model, le_day, le_product = load_xgb_model("model_xgb_hale.pkl")

    data = preprocess_data_glm(uploaded_file, day_cats, prod_cats)
    data = predict_glm_qty(glm_model, data)
    data = predict_xgb_qty(xgb_model, data, le_day, le_product)
    data = apply_diskon(data, col_predicted='predicted_qty_glm')

    st.subheader("\U0001F4CA Tabel Hasil Prediksi")
    st.dataframe(
        data[['transaction_date', 'day', 'product_category', 'unit_price',
              'predicted_qty_glm', 'predicted_qty_xgb', 'diskon', 'harga_setelah_diskon']]
        .sort_values(by='predicted_qty_glm')
        .head(10),
        use_container_width=True
    )

    with st.expander("\U0001F50D Lihat Produk dengan Penjualan Terendah/Tertinggi per Hari"):
        hari_id = st.selectbox("Pilih Hari", list(day_mapping_id.values()))
        hari_en = day_mapping_en[hari_id]

        for model_col, model_label, color_low, color_high in [
            ('predicted_qty_glm', 'GLM', '#fff3cd', '#d1ecf1'),
            ('predicted_qty_xgb', 'XGBoost', '#ffeeba', '#bee5eb')
        ]:
            rendah, tinggi = tampilkan_prediksi_untuk_hari(data, hari_en, model_col)
            if rendah is not None:
                st.markdown(f"""
                <div style='background-color:{color_low};padding:15px;margin-bottom:10px;
                            border-left:6px solid #ffc107;border-radius:5px;color:#000;'>
                    <h4>{model_label} - Penjualan Terendah</h4>
                    Produk: <strong>{rendah['product_category']}</strong><br>
                    Rekomendasi Diskon: {rendah['diskon']*100:.0f}%
                </div>
                <div style='background-color:{color_high};padding:15px;margin-bottom:30px;
                            border-left:6px solid #17a2b8;border-radius:5px;color:#000;'>
                    <h4>{model_label} - Penjualan Tertinggi</h4>
                    Produk: <strong>{tinggi['product_category']}</strong><br>
                    Rekomendasi Diskon: {tinggi['diskon']*100:.0f}%
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning(f"Tidak ada data {model_label} untuk hari {hari_id}.")
                
    mse_glm, rmse_glm, mae_glm, r2_glm = eval_metrics(data['transaction_qty'], data['predicted_qty_glm'])
    mse_xgb, rmse_xgb, mae_xgb, r2_xgb = eval_metrics(data['transaction_qty'], data['predicted_qty_xgb'])

    eval_df = pd.DataFrame({
        'Model': ['GLM Poisson', 'XGBoost'],
        'MSE': [mse_glm, mse_xgb],
        'RMSE': [rmse_glm, rmse_xgb],
        'MAE': [mae_glm, mae_xgb],
        'R2': [r2_glm, r2_xgb]
    })

    st.subheader("\U0001F4C8 Evaluasi Model")
    st.dataframe(eval_df, use_container_width=True)
    st.download_button("Download Evaluasi sebagai CSV", data=eval_df.to_csv(index=False), file_name='evaluasi_model.csv', mime='text/csv')

    st.subheader("\U0001F4C9 Grafik Perbandingan Prediksi")
    plot_predictions(data)
