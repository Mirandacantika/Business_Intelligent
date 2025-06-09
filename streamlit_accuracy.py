import streamlit as st
import pandas as pd
import statsmodels.api as sm
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Constants
DEFAULT_DAY_CATEGORIES = [
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
]
DEFAULT_PRODUCT_CATEGORIES = ['kopi', 'non-kopi']
DEFAULT_MEDIAN_PRICE = 10_000
DISCOUNT_THRESHOLD = 1.5  # Predicted quantity threshold
dISCOUNT_MAX = 0.2        # Maximum discount (20%)

# Mappings
DAY_ID_TO_EN = {
    'Senin': 'Monday', 'Selasa': 'Tuesday', 'Rabu': 'Wednesday',
    'Kamis': 'Thursday', 'Jumat': 'Friday', 'Sabtu': 'Saturday', 'Minggu': 'Sunday'
}
DAY_EN_TO_ID = {v: k for k, v in DAY_ID_TO_EN.items()}


def load_glm_model_info(path: str):
    """
    Load a GLM model and extract category & median price info from its training data.
    Falls back to default categories and median price if metadata unavailable.
    """
    model = sm.load(path)
    try:
        df = model.model.data.frame
        day_cats = df['day'].cat.categories.tolist()
        prod_cats = df['product_category'].cat.categories.tolist()
        median_price = df['unit_price'].median()
    except Exception:
        day_cats, prod_cats, median_price = (
            DEFAULT_DAY_CATEGORIES,
            DEFAULT_PRODUCT_CATEGORIES,
            DEFAULT_MEDIAN_PRICE
        )
    return model, day_cats, prod_cats, median_price


def load_xgb_model(path: str):
    """
    Load an XGBoost model bundled with its label encoders.
    """
    bundle = joblib.load(path)
    return bundle['model'], bundle['le_day'], bundle['le_product']


def preprocess_data(file, day_cats: list, prod_cats: list) -> pd.DataFrame:
    """
    Read uploaded CSV/XLSX, parse dates, and set categorical columns.
    """
    if file.name.lower().endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df['day'] = pd.Categorical(
        df['transaction_date'].dt.day_name(),
        categories=day_cats,
        ordered=True
    )
    df['product_category'] = pd.Categorical(
        df['product_category'],
        categories=prod_cats,
        ordered=True
    )
    return df


def predict_glm(model, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['predicted_qty_glm'] = model.predict(df)
    return df


def predict_xgb(model, df: pd.DataFrame, le_day, le_prod) -> pd.DataFrame:
    df = df.copy()
    # Filter unknown categories
    df = df[df['day'].isin(le_day.classes_) & df['product_category'].isin(le_prod.classes_)]
    df['day_encoded'] = le_day.transform(df['day'])
    df['product_encoded'] = le_prod.transform(df['product_category'])
    features = df[['day_encoded', 'product_encoded', 'unit_price']]
    df['predicted_qty_xgb'] = model.predict(features)
    return df


def calculate_discount(pred_qty: float) -> float:
    """Return discount rate based on predicted quantity."""
    if pred_qty < DISCOUNT_THRESHOLD:
        discount = (1 - pred_qty / DISCOUNT_THRESHOLD) * dISCOUNT_MAX
        return round(min(discount, dISCOUNT_MAX), 2)
    return 0.0


def apply_discounts(df: pd.DataFrame, pred_col: str = 'predicted_qty_glm') -> pd.DataFrame:
    df = df.copy()
    df['discount_rate'] = df[pred_col].apply(calculate_discount)
    df['price_after_discount'] = df['unit_price'] * (1 - df['discount_rate'])
    return df


def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2


def plot_comparison(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['transaction_qty'], label='Aktual', linewidth=2)
    ax.plot(df['predicted_qty_glm'], label='GLM', linewidth=1.5)
    ax.plot(df['predicted_qty_xgb'], label='XGBoost', linewidth=1.5)
    ax.set(
        title='Prediksi vs Aktual',
        xlabel='Index',
        ylabel='Jumlah Transaksi'
    )
    ax.legend()
    ax.grid(linestyle='--', alpha=0.5)
    st.pyplot(fig)


def display_summary(df: pd.DataFrame, day: str, model_key: str):
    """Render a summary card for lowest/highest predicted items on a given day."""
    subset = df[df['day'] == day]
    if subset.empty:
        st.warning(f"Tidak ada data untuk {model_key} di hari {DAY_EN_TO_ID[day]}")
        return

    lowest = subset.nsmallest(1, f'predicted_qty_{model_key}')
    highest = subset.nlargest(1, f'predicted_qty_{model_key}')

    for label, row in [('Terendah', lowest.iloc[0]), ('Tertinggi', highest.iloc[0])]:
        color = '#fff3cd' if label == 'Terendah' else '#d1ecf1'
        border = '#ffc107' if label == 'Terendah' else '#17a2b8'
        with st.container():
            st.markdown(f"""
            <div style='background-color:{color};padding:15px;border-left:6px solid {border};border-radius:5px;'>
                <h4>{model_key.upper()} - Penjualan {label}</h4>
                Produk: <strong>{row['product_category']}</strong><br>
                Diskon: {row['discount_rate']*100:.0f}%<br>
                Harga Setelah Diskon: Rp {int(row['price_after_discount']):,}
            </div>
            """, unsafe_allow_html=True)


def main():
    st.title("Prediksi Transaksi & Evaluasi Model")
    uploaded = st.file_uploader("Unggah .csv atau .xlsx", type=['csv', 'xlsx'])

    if not uploaded:
        st.info("Silakan unggah file data untuk memulai.")
        return

    # Load models
    glm_model, day_cats, prod_cats, _ = load_glm_model_info("model_glm_hale.sm")
    xgb_model, le_day, le_prod = load_xgb_model("model_xgb_hale.pkl")

    # Prepare data
    df = preprocess_data(uploaded, day_cats, prod_cats)
    df = predict_glm(glm_model, df)
    df = predict_xgb(xgb_model, df, le_day, le_prod)
    df = apply_discounts(df)

    # Preview
    st.subheader("Data Prediksi (Top 10 Terendah GLM)")
    st.dataframe(
        df.sort_values('predicted_qty_glm').head(10),
        use_container_width=True
    )

    # Day-specific view
    if st.checkbox("Tampilkan per hari"):  # 🔍
        day_id = st.selectbox("Pilih Hari", list(DAY_ID_TO_EN.keys()))
        day_en = DAY_ID_TO_EN[day_id]
        st.write(f"### Hari: {day_id}")
        display_summary(df, day_en, 'glm')
        display_summary(df, day_en, 'xgb')

    # Metrics table
    mse_g, rmse_g, mae_g, r2_g = compute_metrics(df['transaction_qty'], df['predicted_qty_glm'])
    mse_x, rmse_x, mae_x, r2_x = compute_metrics(df['transaction_qty'], df['predicted_qty_xgb'])

    metrics_df = pd.DataFrame(
        [
            ('GLM Poisson', mse_g, rmse_g, mae_g, r2_g),
            ('XGBoost', mse_x, rmse_x, mae_x, r2_x)
        ],
        columns=['Model', 'MSE', 'RMSE', 'MAE', 'R2']
    )

    st.subheader("Evaluasi Model")
    st.table(metrics_df)
    st.download_button(
        label="Download Evaluasi CSV",
        data=metrics_df.to_csv(index=False).encode('utf-8'),
        file_name='evaluasi_model.csv',
        mime='text/csv'
    )

    # Plot
    st.subheader("Grafik Prediksi vs Aktual")
    plot_comparison(df)


if __name__ == '__main__':
    main()
