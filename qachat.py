import os
from transformers import pipeline
from text_classification import get_sentiment_label
from utills import format_sentiment, save_to_csv
import streamlit as st
import google.generativeai as genai

# Cấu hình API key cho Gemini
try:
    genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))
except:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Tải mô hình Gemini Pro và nhận phản hồi
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

def get_gemini_response(question):
    """Lấy phản hồi từ mô hình Gemini"""
    response = chat.send_message(question)
    return response

# Khởi tạo ứng dụng Streamlit
st.set_page_config(page_title="Q&A Chatbot")
st.header("Gemini-Pro Sentiment Analysis Chatbot")

# Khởi tạo session state cho lịch sử chat nếu chưa tồn tại
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Nhập liệu từ người dùng và nút submit trong Streamlit
input = st.text_input("Input: ", key="input")
submit = st.button("Ask the question")

if submit and input:
    # Phân tích cảm xúc của câu hỏi người dùng
    user_sentiment = get_sentiment_label(input)
    formatted_user_sentiment = format_sentiment(user_sentiment)

    # Lấy phản hồi từ Gemini
    response = get_gemini_response(f'{input} give me a response within one line')

    # Phân tích cảm xúc của phản hồi chatbot
    bot_sentiment = get_sentiment_label(response.text)
    formatted_bot_sentiment = format_sentiment(bot_sentiment)

    # Thêm câu hỏi của người dùng và cảm xúc vào session state
    st.session_state['chat_history'].append(("You", input))
    st.session_state['chat_history'].append(("Sentiment (You)", formatted_user_sentiment))

    # Hiển thị phản hồi của chatbot và cảm xúc
    st.subheader("The Response is")
    st.write(f"{response.text} {formatted_bot_sentiment}", unsafe_allow_html=True)

    # Thêm phản hồi của chatbot và cảm xúc vào session state
    st.session_state['chat_history'].append(("Bot", f'{response.text} {formatted_bot_sentiment}'))
    
    # Lưu câu hỏi, phản hồi và cảm xúc vào CSV
    save_to_csv(input, response.text, user_sentiment)
    save_to_csv(input, response.text, bot_sentiment)

# Hiển thị lịch sử chat
st.subheader("The Chat History is")
for role, text in st.session_state['chat_history']:
    st.write(f"{role}: {text}", unsafe_allow_html=True)
