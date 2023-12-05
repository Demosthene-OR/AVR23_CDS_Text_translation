import streamlit as st
# from translate import Translator
from deep_translator import GoogleTranslator

@st.cache_data(ttl="2d", show_spinner=False)
def trad(message,l):
    try:
        # Utilisation du module translate
        # translator = Translator(to_lang=l , from_lang="fr")
        # translation = translator.translate(message)

        # Utilisation du module deep_translator 
        translation = GoogleTranslator(source='fr', target=l).translate(message.replace("  \n","§§§"))
        translation = translation.replace("§§§","  \n") # .replace("  ","<br>")

        return translation
    except:
        return "Problème de traduction.."
    
def tr(message):
    if 'Language' not in st.session_state: l = 'fr'
    else: l= st.session_state['Language']
    if l == 'fr': return message
    else: message = message.replace(":red[**","").replace("**]","")
    return trad(message,l)


