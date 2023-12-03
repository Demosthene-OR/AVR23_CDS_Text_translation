import streamlit as st
from translate import Translator

@st.cache_data(ttl="1d")
def trad(message,l):
    try:
        translator = Translator(to_lang=l , from_lang="fr")
        translation = translator.translate(message)
        return translation
    except:
        return "Problème de traduction.."
    
def tr(message):
    if 'Language' not in st.session_state: l = 'fr'
    else: l= st.session_state['Language']
    if l == 'fr': return message
    else: message = message.replace(":red[**","").replace("**]","")
    return trad(message,l)
