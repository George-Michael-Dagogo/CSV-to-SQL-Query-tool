import streamlit as st
from time import sleep
import random

# Page config
st.set_page_config(
    page_title="For Onyinyechi ❤️",
    page_icon="❤️",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(45deg, #ffebee, #fce4ec);
    }
    .stMarkdown {
        text-align: center;
    }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
        color: #d32f2f;
    }
    .message-font {
        font-size: 18px !important;
        line-height: 1.6;
        color: #333;
    }
    .footer-font {
        font-style: italic;
        color: #880e4f;
    }
    </style>
    """, unsafe_allow_html=True)

# Container for the main content
with st.container():
    # Animate hearts at the top
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("❤️", help=None)
    with col2:
        st.markdown("💝", help=None)
    with col3:
        st.markdown("❤️", help=None)

    # Title with typing effect
    st.markdown("<h1 class='big-font'>Dearest Onyinyechi Ojinere Maryann</h1>", unsafe_allow_html=True)
    
    # Add some space
    st.write("")
    
    # Message paragraphs with fade-in effect
    messages = [
        "On this special Valentine's Day, my heart beats with joy knowing I have the chance to express my feelings for you. Your name, Onyinyechi, carries the beauty of God's gift, and truly, you are a precious gift to this world.",
        
        "Your smile brightens even the darkest days, and your presence fills every moment with warmth and happiness. The way you carry yourself with such grace and kindness inspires everyone around you. Your strength, intelligence, and beautiful spirit make you truly exceptional.",
        
        "Like the stars that light up the night sky, you bring sparkle and magic to my world. Your laughter is my favorite melody, and your happiness means everything to me. Every moment spent with you is a treasure I hold dear to my heart.",
        
        "Maryann, you are the embodiment of love and grace. Your heart is pure gold, and your soul radiates with a beauty that words cannot fully capture. You make this world a better place just by being in it."
    ]

    # Display each message with a slight delay
    for msg in messages:
        st.markdown(f"<p class='message-font'>{msg}</p>", unsafe_allow_html=True)
        sleep(0.5)  # Small delay between paragraphs

    # Footer
    st.write("")
    st.markdown("<p class='footer-font'>Happy Valentine's Day, my beautiful Onyinyechi ❤️</p>", unsafe_allow_html=True)
    st.markdown("<p class='footer-font'>With all my love and devotion</p>", unsafe_allow_html=True)

    # Animated hearts at the bottom
    cols = st.columns(5)
    for col in cols:
        with col:
            st.markdown("❤️", help=None)

# Add some interactivity
if st.button("💌 Send Love"):
    st.balloons()
    st.snow()
    st.success("Happy Valentine's Day! 💕")

# Background music option
st.sidebar.markdown("### Valentine's Options")
if st.sidebar.button("🎵 Play Romance"):
    st.toast("💝 Spreading love and joy!", icon="💖")