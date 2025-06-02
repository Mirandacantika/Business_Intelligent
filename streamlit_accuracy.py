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
    X = data[['day_encoded', 'product_category_encoded', 'unit_price']]
    data['predicted_qty_xgb'] = model.predict(X)
    return data

# --- Fungsi hitung diskon ---
def hitung_diskon(pred_qty):
    threshold=2
    diskon_maks=1
    if pred_qty < threshold:
        diskon = (1 - pred_qty / threshold) * diskon_maks
        return round(min(diskon, diskon_maks), 2)
    return 0.0

def tampilkan_prediksi_terendah_untuk_hari(data, hari, jumlah_item=1):
    data_hari = data[data['day'] == hari]
    if data_hari.empty:
        return None
    return data_hari.sort_values(by='predicted_qty_glm').head(jumlah_item)

def tampilkan_prediksi_terendah_untuk_hari_xgb(data, hari, jumlah_item=1):
    data_hari = data[data['day'] == hari]
    if data_hari.empty:
        return None
    return data_hari.sort_values(by='predicted_qty_xgb').head(jumlah_item)

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
    ax.plot(df_compare['transaction_qty'].values, label='Aktual', color='dodgerblue', linewidth=2)
    ax.plot(df_compare['predicted_qty_glm'].values, label='Prediksi GLM', color='orange', linewidth=1.5)
    ax.plot(df_compare['predicted_qty_xgb'].values, label='Prediksi XGBoost', color='limegreen', linewidth=1.5)
    ax.set_title('Perbandingan Prediksi vs Aktual', fontsize=16)
    ax.set_xlabel('Index', fontsize=12)
    ax.set_ylabel('Jumlah Transaksi', fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)

# --- Mapping hari ---
day_mapping_id = {
    'Monday': 'Senin', 'Tuesday': 'Selasa', 'Wednesday': 'Rabu', 'Thursday': 'Kamis',
    'Friday': 'Jumat', 'Saturday': 'Sabtu', 'Sunday': 'Minggu'
}
day_mapping_en = {v: k for k, v in day_mapping_id.items()}

# --- Streamlit app ---
st.title("Prediksi Transaksi & Evaluasi Perbandingan Model (GLM vs XGBoost)")

uploaded_file = st.file_uploader("Unggah file data (.xlsx) untuk prediksi", type=["csv", "xlsx"])

if uploaded_file is not None:
    glm_model, day_cats, prod_cats, harga_median = load_glm_model_info("model_glm_hale.sm")
    xgb_model, le_day, le_product = load_xgb_model("model_xgb_hale.pkl")

    data = preprocess_data_glm(uploaded_file, day_cats, prod_cats)
    data = predict_glm_qty(glm_model, data)
    data = predict_xgb_qty(xgb_model, data, le_day, le_product)
    data = apply_diskon(data, col_predicted='predicted_qty_glm')

    st.subheader("\U0001F4CA Data Hasil Prediksi (10 Data Pertama)")
    st.dataframe(
        data[['transaction_date', 'day', 'product_category', 'unit_price',
              'predicted_qty_glm', 'predicted_qty_xgb', 'diskon', 'harga_setelah_diskon']]
        .sort_values(by='predicted_qty_glm', ascending=True)
        .head(10),
        use_container_width=True
    )

    if st.checkbox("\U0001F50D Tampilkan produk dengan prediksi penjualan berdasarkan hari"):
        hari_id_dipilih = st.selectbox("Pilih Hari", list(day_mapping_id.values()))
        hari_en = day_mapping_en[hari_id_dipilih]
        hasil = tampilkan_prediksi_terendah_untuk_hari(data, hari_en)
        hasil_xgb = tampilkan_prediksi_terendah_untuk_hari_xgb(data, hari_en)
        hasil_tertinggi = data[data['day'] == hari_en].sort_values(by='predicted_qty_glm', ascending=False)
        hasil_tertinggi_xgb = data[data['day'] == hari_en].sort_values(by='predicted_qty_xgb', ascending=False)

        if hasil is not None and not hasil.empty:
            hasil = hasil.iloc[0]
            hasil_tertinggi = hasil_tertinggi.iloc[0]
            st.markdown(f"**GLM: {hasil['product_category']} (Diskon: {hasil['diskon']*100:.0f}%) - Rp {int(hasil['harga_setelah_diskon']):,}**")
            st.markdown(f"**GLM (Tertinggi): {hasil_tertinggi['product_category']} - Rp {int(hasil_tertinggi['harga_setelah_diskon']):,}**")
        else:
            st.warning(f"Tidak ada data GLM untuk hari {hari_id_dipilih}.")

        if hasil_xgb is not None and not hasil_xgb.empty:
            hasil_xgb = hasil_xgb.iloc[0]
            hasil_tertinggi_xgb = hasil_tertinggi_xgb.iloc[0]
            st.markdown(f"**XGB: {hasil_xgb['product_category']} (Diskon: {hasil_xgb['diskon']*100:.0f}%) - Rp {int(hasil_xgb['harga_setelah_diskon']):,}**")
            st.markdown(f"**XGB (Tertinggi): {hasil_tertinggi_xgb['product_category']} - Rp {int(hasil_tertinggi_xgb['harga_setelah_diskon']):,}**")
        else:
            st.warning(f"Tidak ada data XGBoost untuk hari {hari_id_dipilih}.")

    mse_glm, rmse_glm, mae_glm, r2_glm = eval_metrics(data['transaction_qty'], data['predicted_qty_glm'])
    mse_xgb, rmse_xgb, mae_xgb, r2_xgb = eval_metrics(data['transaction_qty'], data['predicted_qty_xgb'])

    eval_df = pd.DataFrame({
        'Model': ['GLM Poisson', 'XGBoost'],
        'MSE': [mse_glm, mse_xgb],
        'RMSE': [rmse_glm, rmse_xgb],
        'MAE': [mae_glm, mae_xgb],
        'R2': [r2_glm, r2_xgb]
    })

    st.subheader("\U0001F4C8 Evaluasi Perbandingan Model")
    st.table(eval_df)

    csv_eval = eval_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Evaluasi CSV", data=csv_eval, file_name='evaluasi_perbandingan_model.csv', mime='text/csv')

    st.subheader("\U0001F4C9 Grafik Perbandingan Prediksi vs Actual")
    plot_predictions(data)
