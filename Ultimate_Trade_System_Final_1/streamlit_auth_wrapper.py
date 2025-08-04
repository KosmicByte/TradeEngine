
import streamlit as st

def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["authenticated"] = True
            del st.session_state["password"]
        else:
            st.session_state["authenticated"] = False

    if "authenticated" not in st.session_state:
        st.text_input("Enter password:", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["authenticated"]:
        st.error("❌ Incorrect password.")
        return False
    return True

# Example usage
if check_password():
    st.title("🔐 Secure Dashboard")
    st.success("Welcome! You're authenticated.")
    # import and call your real dashboard code here
