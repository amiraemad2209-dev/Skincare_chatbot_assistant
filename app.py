import os
import time
import random
import threading
from collections import deque

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from resolver.shortcuts import SHORTCUT_REGISTRY
from resolver.resolver import resolve_shortcut

from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
import base64

def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

img = img_to_base64("images/simple.png")

# =========================
# ENV
# =========================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# =========================
# RATE LIMITER
# =========================
class RPMRateLimiter:
    def __init__(self, rpm=30, window_s=60):
        self.rpm = rpm
        self.window_s = window_s
        self.requests = deque()
        self.lock = threading.Lock()

    def cleanup(self):
        cutoff = time.time() - self.window_s
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()

    def acquire(self):
        while True:
            with self.lock:
                self.cleanup()
                if len(self.requests) < self.rpm:
                    self.requests.append(time.time())
                    return
            time.sleep(0.5)


rate_limiter = RPMRateLimiter()


# =========================
# HELPERS
# =========================
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text


def evaluate_response(response):
    return {
        "characters": len(response),
        "words": len(response.split()),
        "is_empty": len(response.strip()) == 0
    }


def invoke_with_retry(conversation, user_input, max_attempts=5):
    for attempt in range(max_attempts):
        try:
            return conversation.predict(input=user_input)

        except Exception as e:
            status_code = getattr(e, "status_code", None)
            if status_code not in [429, 500, 502, 503, 504]:
                raise e

            time.sleep(min(2 ** attempt, 60) + random.uniform(0, 0.5))

    raise Exception("Max retries exceeded")


# =========================
# STREAMLIT APP
# =========================
def main():

    st.set_page_config(
        page_title="SkinCare Assistant",
        layout="wide"
    )
    st.markdown("""
<style>

/* =========================
   Background
========================= */
.stApp {
    background:
    linear-gradient(
        rgba(255,240,245,0.3),
        rgba(255,228,236,0.4)
    ),
     url("data:image/png;base64,{img}");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* =========================
   Main Container
========================= */
.block-container {
    width: 100%;
    max-width: 100%;

    margin: 0;
    padding: 2rem;
    background: rgba(255,255,255,0.3);
    backdrop-filter: blur(12px);

    border-radius: 25px;

    box-shadow: 0 8px 32px rgba(194,24,91,0.20);
}

/* =========================
   Titles
========================= */
h1 {
    color: #c2185b !important;
    text-align: center;
    font-weight: 700;
}

h3, h4 {
    color: #d81b60 !important;
}

h2 {
    color: #ffffff !important;
    font-weight: 70;
}

/* =========================
   Text
========================= */
p, label {
    color: #5f4b53 !important;
}

/* =========================
   Sidebar
========================= */

section[data-testid="stSidebar"] {
    background: rgba(0,0,0, 0.85); 
    backdrop-filter: blur(12px);
}

section[data-testid="stSidebar"] * {
    color: #c2185b !important;
}

/* =========================
   Buttons
========================= */
.stButton > button {
    background: linear-gradient(
        135deg,
        #e91e63,
        #c2185b
    );

    color: #ffffff !important;  

    border: none;
    border-radius: 12px;

    padding: 0.6rem 1.2rem;
    font-weight: bold;
}

.stButton > button:hover {
    transform: scale(1.03);
    transition: 0.3s ease;
}
.stButton button * {
    color: white !important;
}

/* =========================
   Text Inputs
========================= */
.stTextInput input,
.stTextArea textarea {
    border-radius: 12px !important;
    border: 2px solid #f8bbd0 !important;

    background-color: rgba(255,255,255,0.85) !important;
    color: #5f4b53 !important;
}

/* =========================
   Select Box
========================= */
.stSelectbox label {
    color: #c2185b !important;
    font-weight: 600;
}

.stSelectbox div[data-baseweb="select"] {
    border-radius: 12px;
}

/* =========================
   Chat Messages
========================= */
.stChatMessage {
    background: rgba(255,255,255,0.65);
    border-radius: 18px;
    padding: 12px;
}

/* =========================
   Metrics
========================= */
[data-testid="stMetricValue"] {
    color: #c2185b !important;
}

[data-testid="stMetricLabel"] {
    color: #5f4b53 !important;
}

</style>

"""

, unsafe_allow_html=True)

    st.title("SkinCare Assistant")
    # =========================
    # SESSION STATE INIT
    # =========================
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "input_text" not in st.session_state:
        st.session_state.input_text = ""

    if "memory" not in st.session_state:
        st.session_state.memory = None

    if "conversation" not in st.session_state:
        st.session_state.conversation = None


    # =========================
    # SIDEBAR
    # =========================
    st.sidebar.title("⚙ Settings")
    
    MODEL_NAME = "llama-3.3-70b-versatile"
    MEMORY_LEN = 5
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])


    # =========================
    # MEMORY
    # =========================
    
    if st.session_state.memory is None:
        st.session_state.memory = ConversationBufferMemory(
        k=MEMORY_LEN,
        return_messages=True
       )

    # =========================
    # LLM
    # =========================
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
       model_name=MODEL_NAME,
        temperature=0.3,
        max_tokens=1024
    )


    # =========================
    # PROMPT
    # =========================
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template="""
You are a skincare assistant.

Only answer skincare questions.

If unrelated:
Arabic: أنا أساعد فقط في العناية بالبشرة.
English: I can only help with skincare-related questions.

History:
{history}

Human: {input}
AI:
"""
    )


    # =========================
    # CONVERSATION
    # =========================
    if st.session_state.conversation is None:
        st.session_state.conversation = ConversationChain(
            llm=llm,
            memory=st.session_state.memory,
            prompt=prompt
        )


    # =========================
    # PDF TEXT
    # =========================
    extracted_text = ""
    if uploaded_file:
        extracted_text = extract_text_from_pdf(uploaded_file)
        st.sidebar.success("PDF loaded")


    # =========================
    # SHORTCUTS
    # =========================
    
    st.subheader("Quick Shortcuts")
    st.markdown(" Make it Easy & Faster !")

    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)

    if c1.button("Routine"):
        st.session_state.input_text = "search for skincare routine for "

    if c2.button("Acne"):
        st.session_state.input_text = "what is the suitable acne treatment for "

    if c3.button(" Keep Safe"):
        st.session_state.input_text = "what are the safe skincare ingredients whice I can use ? "


    user_question = st.text_area(
        "Ask your question",
        value=st.session_state.input_text
    )


    # =========================
    # SEND
    # =========================
    if st.button("Send"):

        if not user_question.strip():
            st.warning("Enter a question")
            return

        rate_limiter.acquire()

        resolved = resolve_shortcut(user_question, SHORTCUT_REGISTRY)
        final_input = resolved.prompt

        if extracted_text:
            final_input += "\n\nPDF Context:\n" + extracted_text[:4000]

        with st.spinner("Thinking..."):
            response = invoke_with_retry(
                st.session_state.conversation,
                final_input
            )

        st.session_state.chat_history.append({
            "human": user_question,
            "AI": response
        })

        st.session_state.input_text = ""

        st.subheader(f"- {user_question} : ")


        #---------------------------------------------------

        st.session_state.last_response = response
        st.session_state.last_question = user_question

    # =========================
    # DISPLAY LAST ANSWER
    # =========================
    if "last_response" in st.session_state:
       
       st.write(st.session_state.last_response)

       
        #st.subheader("Evaluation")
        #st.json(evaluate_response(response))

    wait_for_all_tracers()


    # =========================
    # INGREDIENT CHECKER
    # =========================
    st.subheader("🧪 Ingredient Checker")

    ingredient_name = st.text_input("Enter Ingredient Name")

    if st.button("Check Ingredient"):

        if not ingredient_name.strip():
            st.warning("Please enter an ingredient")

        else:

            ingredient_prompt = f"""
You are a skincare ingredient expert.

Analyze this ingredient:
{ingredient_name}

Return:
1. What is it?
2. Benefits
3. Side effects
4. Skin types
5. What not to combine it with
6. Usage tips

Respond in same language.
"""

            rate_limiter.acquire()

            with st.spinner("Analyzing..."):
                ingredient_response = invoke_with_retry(
                    st.session_state.conversation,
                    ingredient_prompt
                )

            st.session_state.last_ingredient = ingredient_response
            st.session_state.last_ing_name = ingredient_name

    if "last_ingredient" in st.session_state:
         
            st.write(st.session_state.last_ingredient)

    # =========================
    # CHAT HISTORY
    # =========================
    if st.session_state.chat_history:

        #st.subheader("🕒 Chat History")

        for chat in reversed(st.session_state.chat_history):
            #st.markdown(f"**You:** {chat['human']}")
            #st.markdown(f"**Bot:** {chat['AI']}")
            st.write("---")


if __name__ == "__main__":
    main()