import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import datetime

# --- CẤU HÌNH ---
st.set_page_config(page_title="Smart Weather AI", page_icon="🧠", layout="wide")
st.title("🧠 Dự báo Thời Tiết AI (Chế độ Smart Context)")
st.markdown("---")

# --- 1. LOAD MODEL & DATA ---
@st.cache_resource
def load_resources():
    # Load Models
    model_rain = load_model('models/lstm_weather_model.h5')
    scaler_rain = joblib.load('models/scaler.pkl')
    model_reg = load_model('models/weather_regression_model.keras')
    scaler_reg_in = joblib.load('models/scaler_reg_input.pkl')
    scaler_reg_out = joblib.load('models/scaler_reg_target.pkl')
    
    # Load CSV (Để tìm kiếm lịch sử tương đồng)
    try:
        df = pd.read_csv('weatherAUS.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        # Fill NA trước để tìm kiếm không bị lỗi
        features = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 
                    'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 
                    'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm']
        df[features] = df[features].ffill().bfill().fillna(0)
        return model_rain, scaler_rain, model_reg, scaler_reg_in, scaler_reg_out, df
    except Exception as e:
        return None, None, None, None, None, None

try:
    m_rain, s_rain, m_reg, s_reg_in, s_reg_out, df_db = load_resources()
    if df_db is None:
        st.error("Thiếu file 'weatherAUS.csv' hoặc Model. Vui lòng kiểm tra lại.")
        st.stop()
    else:
        st.success("✅ Hệ thống đã sẵn sàng kết nối dữ liệu!")
except Exception as e:
    st.error(f"Lỗi khởi động: {e}")
    st.stop()

# --- 2. GIAO DIỆN NHẬP TAY (SIDEBAR) ---
st.sidebar.header("📝 Nhập thông số hôm nay")

def user_input_ui():
    # Nhập ngày để tính Month Sin/Cos
    date_pick = st.sidebar.date_input("Ngày giả lập", datetime.date.today())
    
    st.sidebar.subheader("Nhiệt độ")
    min_t = st.sidebar.number_input("Min Temp (°C)", value=20.0, step=0.5)
    max_t = st.sidebar.number_input("Max Temp (°C)", value=30.0, step=0.5)
    t_9am = st.sidebar.number_input("Temp 9am (°C)", value=22.0, step=0.5)
    t_3pm = st.sidebar.number_input("Temp 3pm (°C)", value=28.0, step=0.5)
    
    st.sidebar.subheader("Mưa & Ẩm")
    rain = st.sidebar.number_input("Lượng mưa (mm)", value=0.0, step=1.0)
    hum_9 = st.sidebar.slider("Độ ẩm 9am (%)", 0, 100, 60)
    hum_3 = st.sidebar.slider("Độ ẩm 3pm (%)", 0, 100, 40)
    
    st.sidebar.subheader("Gió & Áp suất")
    gust = st.sidebar.slider("Gió giật (km/h)", 0, 100, 35)
    w_9 = st.sidebar.slider("Gió 9am (km/h)", 0, 100, 15)
    w_3 = st.sidebar.slider("Gió 3pm (km/h)", 0, 100, 20)
    p_9 = st.sidebar.number_input("Áp suất 9am (hPa)", value=1015.0, step=1.0)
    p_3 = st.sidebar.number_input("Áp suất 3pm (hPa)", value=1012.0, step=1.0)
    
    # Tạo dictionary dữ liệu thô
    data = {
        'MinTemp': min_t, 'MaxTemp': max_t, 'Rainfall': rain, 
        'WindGustSpeed': gust, 'WindSpeed9am': w_9, 'WindSpeed3pm': w_3,
        'Humidity9am': hum_9, 'Humidity3pm': hum_3,
        'Pressure9am': p_9, 'Pressure3pm': p_3,
        'Temp9am': t_9am, 'Temp3pm': t_3pm,
        'Date': date_pick
    }
    return data

user_data = user_input_ui()

# --- 3. THUẬT TOÁN TÌM KIẾM NGỮ CẢNH (SMART CONTEXT) ---
def find_best_history_match(df, user_input_dict):
    # Các cột dùng để so sánh (Features quan trọng nhất)
    # Ta tìm ngày nào trong quá khứ có Nhiệt độ, Mưa, Gió gần giống nhất với cái user nhập
    compare_cols = ['MaxTemp', 'Rainfall', 'WindGustSpeed', 'Humidity3pm', 'Pressure3pm']
    
    # Tạo vector input của user
    user_vector = np.array([user_input_dict[c] for c in compare_cols])
    
    # Lấy dữ liệu từ DB
    db_matrix = df[compare_cols].values
    
    # Tính khoảng cách Euclidean giữa input user và toàn bộ lịch sử
    # (Công thức: căn bậc 2 của tổng bình phương sai số)
    distances = np.linalg.norm(db_matrix - user_vector, axis=1)
    
    # Tìm index của ngày giống nhất (khoảng cách nhỏ nhất)
    # Lưu ý: Phải chọn ngày có index > 30 để lấy được lịch sử
    sorted_indices = np.argsort(distances)
    
    best_idx = -1
    for idx in sorted_indices:
        if idx > 30: # Đảm bảo có đủ lịch sử
            best_idx = idx
            break
            
    # Lấy 29 ngày trước ngày đó
    # Logic: [Day_Match-29] ... [Day_Match-1] + [User_Input]
    history_29_days = df.iloc[best_idx-29 : best_idx].copy()
    
    return history_29_days, distances[best_idx]

# --- 4. XỬ LÝ VÀ DỰ BÁO ---
if st.button("🚀 Tạo ngữ cảnh & Dự báo"):
    
    # 4.1 Tìm 29 ngày quá khứ phù hợp nhất
    with st.spinner("Đang quét dữ liệu lịch sử để tìm mẫu thời tiết tương đồng..."):
        history_df, diff_score = find_best_history_match(df_db, user_data)
    
    st.info(f"💡 AI đã tìm thấy một chuỗi thời tiết trong quá khứ khớp với input của bạn (Sai số: {diff_score:.2f}). Đang ghép nối...")

    # 4.2 Tạo DataFrame 1 dòng từ User Input
    user_row = pd.DataFrame([user_data]) # Chứa các cột features + Date
    
    # 4.3 Ghép 29 ngày lịch sử + 1 ngày User
    # Cần đảm bảo cột giống nhau
    cols_to_use = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 
                   'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 
                   'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm']
    
    full_30_days = pd.concat([history_df[cols_to_use], user_row[cols_to_use]], ignore_index=True)
    
    # Hiển thị biểu đồ chứng minh "Không phải Flat Line"
    st.subheader("📈 Biểu đồ dữ liệu đầu vào (Đã tái tạo ngữ cảnh)")
    chart_data = full_30_days[['MaxTemp', 'Humidity3pm']].copy()
    chart_data['Source'] = ['History']*29 + ['User Input'] # Đánh dấu điểm cuối
    st.line_chart(chart_data)
    
    # --- 5. PREDICT (GIỐNG CODE CŨ) ---
    
    # Feature Engineering (Month Sin/Cos)
    # Lấy tháng từ ngày user chọn
    m = user_data['Date'].month
    m_sin = np.sin(2 * np.pi * m / 12)
    m_cos = np.cos(2 * np.pi * m / 12)
    
    # Gán Month cho cả 30 ngày (Giả định cùng tháng)
    full_30_days['Month'] = m
    full_30_days['Month_sin'] = m_sin
    full_30_days['Month_cos'] = m_cos
    
    # Chuẩn bị Input Array
    # Model Reg (14 features)
    cols_reg = cols_to_use + ['Month_sin', 'Month_cos']
    X_reg = s_reg_in.transform(full_30_days[cols_reg].values)[np.newaxis, :, :]
    
    # Model Rain (13 features)
    cols_rain = cols_to_use + ['Month']
    X_rain = s_rain.transform(full_30_days[cols_rain].values)[np.newaxis, :, :]
    
    # Chạy Model
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌡️ Dự báo Chỉ số (Ngày mai)")
        pred_vals = s_reg_out.inverse_transform(m_reg.predict(X_reg))
        st.metric("Max Temp", f"{pred_vals[0][0]:.1f} °C")
        st.metric("Humidity", f"{pred_vals[0][1]:.1f} %")
        st.metric("Wind Gust", f"{pred_vals[0][2]:.1f} km/h")
        
    with col2:
        st.subheader("🌧️ Dự báo Mưa (Ngày mai)")
        prob = m_rain.predict(X_rain)[0][0]
        st.metric("Xác suất mưa", f"{prob*100:.1f}%")
        if prob > 0.5:
            st.error("DỰ BÁO: CÓ MƯA")
        else:
            st.success("DỰ BÁO: KHÔNG MƯA")