import streamlit as st
import joblib
import pandas as pd
import base64
import matplotlib.pyplot as plt



# ===== Apply custom CSS from external file =====
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ===== THEME SWITCHER =====
theme = st.sidebar.radio("🎨 Choose Theme:", ["🌞 Light", "🌙 Dark"])
if theme == "🌙 Dark":
    st.markdown("""
        <style>
        .stApp {
            background-color: #1e1e1e;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)
    st.sidebar.success("🌌 Dark mode activated!")
else:
    st.sidebar.info("☀️ Light mode activated!")

# ===== SIDEBAR HELP =====
st.sidebar.title("🧾 Instructions")
st.sidebar.markdown("""
- Use the text box to check a single news article.
- Or upload a CSV file with a text column.
- Get results + charts + download option.
""")
st.sidebar.markdown("📄 Example CSV format:")
st.sidebar.code("text\nFake news content 1\nReal news content 2")

# ===== PAGE TITLE =====
st.set_page_config(page_title="Fake News Detection", layout="centered")
st.title("📰 Fake News Detection System")
st.markdown("<h3>Check if a news article is <em>Real or Fake</em> using Machine Learning.</h3>", unsafe_allow_html=True)

# ===== LOAD MODEL =====
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ========== 1️⃣ SINGLE TEXT CHECK ========== #
st.subheader("✍️ Enter a News Article")

user_input = st.text_area("Enter news text below", height=200)

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]

        result_message = f"The news you checked is predicted as: {'Real ✅' if prediction == 1 else 'Fake ❌'}."

        if prediction == 1:
            st.success("✅ This news is Real.")
        else:
            st.error("❌ This news is Fake.")

# ========== 2️⃣ BULK FILE UPLOAD ========== #
st.markdown("---")
st.subheader("📂 Bulk Fake News Check (Upload CSV)")

uploaded_file = st.file_uploader("📁 Upload a CSV file with a *text* column", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if 'text' not in df.columns:
            st.error("❌ CSV must have a column named 'text'.")
        else:
            # Predict
            X = vectorizer.transform(df['text'])
            predictions = model.predict(X)
            df['Prediction'] = ['Real' if p == 1 else 'Fake' for p in predictions]

            st.success("✅ Predictions completed!")
            st.write(df)

            # ===== COUNTS =====
            real_count = sum(df['Prediction'] == 'Real')
            fake_count = sum(df['Prediction'] == 'Fake')
            st.info(f"🟢 Real News: {real_count} | 🔴 Fake News: {fake_count}")

            # ===== PIE CHART =====
            st.subheader("📊 Prediction Summary (Pie Chart)")
            count_data = df['Prediction'].value_counts()
            fig, ax = plt.subplots()
            ax.pie(count_data, labels=count_data.index, autopct='%1.1f%%', colors=["green", "red"])
            ax.axis('equal')
            st.pyplot(fig)

            # ===== DOWNLOAD BUTTON =====
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">📥 Download Results as CSV</a>'
            st.markdown(href, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Error reading file: {e}")
