import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import datetime
# 👇 THƯ VIỆN MỚI CỦA GOOGLE 👇
from google import genai 

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Smart Weather AI", page_icon="🧠", layout="wide")
st.title("🧠 Dự báo Thời Tiết AI (Chế độ Smart Context)")

# 👇 DÁN MÃ KEY CỦA BẠN VÀO GIỮA DẤU NGOẶC KÉP NÀY 👇
MY_API_KEY = "AIzaSyAf-iXfYqSSIVFXaxRDHS4YEPAf4O676vk" 

# Khởi tạo Client kiểu mới
try:
    client = genai.Client(api_key=MY_API_KEY)
except Exception as e:
    st.error(f"Lỗi Key: {e}")



# ==============================================================================
# BƯỚC 1: HÀM LOGIC TƯ VẤN (DÙNG THƯ VIỆN GOOGLE-GENAI MỚI)
# ==============================================================================
@st.cache_data(show_spinner=False)
def get_smart_advice(rain_prob, temp, humidity, wind_speed):
    prompt = f"""
    Bạn là trợ lý thời tiết. Dự báo ngày mai: 
    - Mưa: {rain_prob*100:.0f}% 
    - Nhiệt độ: {temp:.1f}°C 
    - Độ ẩm: {humidity:.0f}% 
    - Gió: {wind_speed:.0f} km/h.

    Hãy đưa ra 4 lời khuyên ngắn (có emoji) về: Trang phục, Sức khỏe, Di chuyển, Hoạt động.
    """
    
    try:
        # 👇 CÚ PHÁP MỚI CỦA GOOGLE GENAI 👇
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt
        )
        
        status = "Cảnh báo" if rain_prob > 0.5 else "Bình thường"
        
        # Xử lý text trả về
        if response.text:
            advice_list = response.text.split('\n')
            advice_list = [line for line in advice_list if line.strip() != ""]
            return advice_list, status
        else:
            return ["⚠️ AI không trả lời."], "Bình thường"
            
    except Exception as e:
        # In lỗi chi tiết ra để debug nếu sai Key
        return [f"⚠️ Lỗi kết nối AI: {str(e)}"], "Bình thường"

# --- 1. LOAD MODEL & DATA ---
@st.cache_resource
def load_resources():
    # Load Models
    try:
        model_rain = load_model('models/lstm_weather_model.h5')
        scaler_rain = joblib.load('models/scaler.pkl')
        model_reg = load_model('models/weather_regression_model.keras') # Hoặc .h5 tùy file bạn lưu
        scaler_reg_in = joblib.load('models/scaler_reg_input.pkl')
        scaler_reg_out = joblib.load('models/scaler_reg_target.pkl')
        
        # Load CSV (Để tìm kiếm lịch sử tương đồng)
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
        st.error("Lỗi: Không tìm thấy Model hoặc file csv. Vui lòng kiểm tra thư mục 'models/'.")
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
    compare_cols = ['MaxTemp', 'Rainfall', 'WindGustSpeed', 'Humidity3pm', 'Pressure3pm']
    user_vector = np.array([user_input_dict[c] for c in compare_cols])
    db_matrix = df[compare_cols].values
    distances = np.linalg.norm(db_matrix - user_vector, axis=1)
    
    sorted_indices = np.argsort(distances)
    best_idx = -1
    for idx in sorted_indices:
        if idx > 30: 
            best_idx = idx
            break
            
    history_29_days = df.iloc[best_idx-29 : best_idx].copy()
    return history_29_days, distances[best_idx]

# --- 4. XỬ LÝ VÀ DỰ BÁO ---
if st.button("🚀 Tạo ngữ cảnh & Dự báo"):
    
    # 4.1 Tìm 29 ngày quá khứ
    with st.spinner("Đang quét dữ liệu lịch sử để tìm mẫu thời tiết tương đồng..."):
        history_df, diff_score = find_best_history_match(df_db, user_data)
    
    st.info(f"💡 AI đã tìm thấy một chuỗi thời tiết trong quá khứ khớp với input của bạn (Sai số: {diff_score:.2f}). Đang ghép nối...")

    # 4.2 Tạo DataFrame 1 dòng từ User Input
    user_row = pd.DataFrame([user_data])
    
    # 4.3 Ghép 29 ngày lịch sử + 1 ngày User
    cols_to_use = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 
                   'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 
                   'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm']
    
    full_30_days = pd.concat([history_df[cols_to_use], user_row[cols_to_use]], ignore_index=True)
    
    # Hiển thị biểu đồ
    st.subheader("📈 Biểu đồ dữ liệu đầu vào (Đã tái tạo ngữ cảnh)")
    chart_data = full_30_days[['MaxTemp', 'Humidity3pm']].copy()
    chart_data['Source'] = ['History']*29 + ['User Input']
    st.line_chart(chart_data)
    
    # --- 5. PREDICT ---
    
    # A. Feature Engineering cho Regression
    m = user_data['Date'].month
    m_sin = np.sin(2 * np.pi * m / 12)
    m_cos = np.cos(2 * np.pi * m / 12)
    full_30_days['Month_sin'] = m_sin
    full_30_days['Month_cos'] = m_cos
    
    cols_reg = cols_to_use + ['Month_sin', 'Month_cos']
    X_reg = s_reg_in.transform(full_30_days[cols_reg].values)[np.newaxis, :, :]
    
    # B. Feature Engineering cho Model Mưa
    full_30_days['RainToday'] = full_30_days['Rainfall'].apply(lambda x: 1.0 if x >= 1.0 else 0.0)
    cols_rain_fixed = cols_to_use + ['RainToday']
    X_rain = s_rain.transform(full_30_days[cols_rain_fixed].values)[np.newaxis, :, :]
    
    # --- CHẠY MODEL ---
    col1, col2 = st.columns(2)
    
    # Lưu các giá trị dự báo để dùng cho phần AI tư vấn bên dưới
    pred_temp_val = 0
    pred_humid_val = 0
    pred_wind_val = 0
    pred_rain_prob = 0

    with col1:
        st.subheader("🌡️ Dự báo Chỉ số (Ngày mai)")
        pred_vals = s_reg_out.inverse_transform(m_reg.predict(X_reg))
        
        pred_temp_val = pred_vals[0][0]
        pred_humid_val = pred_vals[0][1]
        pred_wind_val = pred_vals[0][2]

        st.metric("Max Temp", f"{pred_temp_val:.1f} °C")
        st.metric("Humidity", f"{pred_humid_val:.1f} %")
        st.metric("Wind Gust", f"{pred_wind_val:.1f} km/h")
        
    with col2:
        st.subheader("🌧️ Dự báo Mưa (Ngày mai)")
        prob = m_rain.predict(X_rain)[0][0]
        pred_rain_prob = prob

        st.metric("Xác suất mưa", f"{prob*100:.1f}%")
        
        if prob > 0.5:
            st.error("☔ DỰ BÁO: CÓ MƯA (Yes)")
        else:
            st.success("☀️ DỰ BÁO: KHÔNG MƯA (No)")

    # ==============================================================================
    # BƯỚC 2: TÍCH HỢP GIAO DIỆN TƯ VẤN (AI ADVICE UI)
    # ==============================================================================
    st.markdown("---")
    st.subheader("🤖 Trợ lý AI gợi ý trang phục & Hành động")
    
    # Gọi hàm logic tư vấn đã viết ở trên
    suggestions, status = get_smart_advice(pred_rain_prob, pred_temp_val, pred_humid_val, pred_wind_val)
    
   # 1. Chuẩn bị nội dung Text trước
    advice_content = f"### 📝 Tổng hợp lời khuyên cho bạn ({status}):\n"
    for item in suggestions:
        advice_content += f"- {item}\n" # Cộng dồn các dòng lời khuyên lại
            
    # 2. Hiển thị ra màn hình (Không dùng "with")
    if status == "Nguy hiểm":
        st.error(advice_content, icon="🚨")
    elif status == "Cảnh báo":
        st.warning(advice_content, icon="⚠️")
    else:
        st.info(advice_content, icon="ℹ️")
            
    # Hiển thị gợi ý hình ảnh (Visual Suggestion)
    st.write("")
    st.write("**Gợi ý set đồ phù hợp:**")
    c1, c2, c3, c4 = st.columns(4)

    # Cột 1: Dù hay Kính
    with c1:
        if pred_rain_prob > 0.5:
            st.image("https://cdn-icons-png.flaticon.com/512/3343/3343640.png", caption="Mang Ô/Áo mưa", width=80)
        else:
            st.image("https://cdn-icons-png.flaticon.com/512/869/869869.png", caption="Kính râm", width=80)

    # Cột 2: Áo khoác
    with c2:
        if pred_temp_val < 20:
            st.image("https://cdn-icons-png.flaticon.com/512/2806/2806240.png", caption="Áo khoác ấm", width=80)
        elif pred_temp_val < 25:
            st.image("https://cdn-icons-png.flaticon.com/512/2589/2589255.png", caption="Áo khoác nhẹ", width=80)
        else:
            st.image("https://cdn-icons-png.flaticon.com/512/2589/2589255.png", caption="Áo thun thoáng mát", width=80)

    # Cột 3: Phụ kiện khác
    with c3:
        if pred_temp_val > 30:
            st.image("https://cdn-icons-png.flaticon.com/512/2857/2857448.png", caption="Kem chống nắng", width=80)
        elif pred_humid_val < 40:
            st.image("https://cdn-icons-png.flaticon.com/512/3233/3233543.png", caption="Kem dưỡng ẩm", width=80)
        else:
            st.image("https://cdn-icons-png.flaticon.com/512/2917/2917995.png", caption="Bình nước cá nhân", width=80)

    # Cột 4: Phương tiện
    with c4:
        if pred_wind_val > 40 or pred_rain_prob > 0.6:
             st.image("https://cdn-icons-png.flaticon.com/512/3062/3062837.png", caption="Hạn chế xe máy", width=80)
        else:
             st.image("https://cdn-icons-png.flaticon.com/512/3194/3194668.png", caption="Đi lại tự do", width=80)