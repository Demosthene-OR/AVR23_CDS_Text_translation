import streamlit as st
import os.path
from collections import OrderedDict
from streamlit_option_menu import option_menu
# Define TITLE, TEAM_MEMBERS and PROMOTION values, in config.py.
import config
from tabs.custom_vectorizer import custom_tokenizer, custom_preprocessor
import os
from translate_app import tr

# Initialize a session state variable that tracks the sidebar state (either 'expanded' or 'collapsed').
if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = 'expanded'
else:
    st.session_state.sidebar_state = 'auto'

st.set_page_config (
    page_title=config.TITLE,
    page_icon= "assets/faviconV2.png",
    initial_sidebar_state=st.session_state.sidebar_state
)

# Si l'application tourne localement, session_state.Cloud == 0 
# Si elle tourne sur le Cloud de Hugging Face, ==1
st.session_state.Cloud = 1
# En fonction de la valeur de varible précédente, le data path est différent
if st.session_state.Cloud == 0: 
    st.session_state.DataPath = "../data"
    st.session_state.ImagePath = "../images"
    st.session_state.reCalcule = False
else: 
    st.session_state.DataPath = "data"
    st.session_state.ImagePath = "images"
    st.session_state.reCalcule = False

# Define the root folders depending on local/cloud run
# thisfile = os.path.abspath(__file__)
# if ('/' in thisfile): 
#     os.chdir(os.path.dirname(thisfile))

# Nécessaire pour la version windows 11
if st.session_state.Cloud == 0: 
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Tabs in the ./tabs folder, imported here.
from tabs import intro, exploration_tab, data_viz_tab, id_lang_tab, modelisation_dict_tab, modelisation_seq2seq_tab, game_tab


with open("style.css", "r") as f:
    style = f.read()

st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)


# Add tab in this ordered dict by
# passing the name in the sidebar as key and the imported tab
# as value as follow :
TABS = OrderedDict(
    [
        (tr(intro.sidebar_name), intro),
        (tr(exploration_tab.sidebar_name), exploration_tab),
        (tr(data_viz_tab.sidebar_name), data_viz_tab),
        (tr(id_lang_tab.sidebar_name), id_lang_tab),
        (tr(modelisation_dict_tab.sidebar_name), modelisation_dict_tab),
        (tr(modelisation_seq2seq_tab.sidebar_name), modelisation_seq2seq_tab),
        (tr(game_tab.sidebar_name), game_tab ),
    ]
)


lang_tgt   = ['fr','en','af','ak','sq','de','am','en','ar','hy','as','az','ba','bm','eu','bn','be','my','bs','bg','ks','ca','ny','zh','si','ko','co','ht','hr','da','dz','gd','es','eo','et','ee','fo','fj','fi','fr','fy','gl','cy','lg','ka','el','gn','gu','ha','he','hi','hu','ig','id','iu','ga','is','it','ja','kn','kk','km','ki','rw','ky','rn','ku','lo','la','lv','li','ln','lt','lb','mk','ms','ml','dv','mg','mt','mi','mr','mn','nl','ne','no','nb','nn','oc','or','ug','ur','uz','ps','pa','fa','pl','pt','ro','ru','sm','sg','sa','sc','sr','sn','sd','sk','sl','so','st','su','sv','sw','ss','tg','tl','ty','ta','tt','cs','te','th','bo','ti','to','ts','tn','tr','tk','tw','uk','vi','wo','xh','yi']
label_lang = ['Français', 'Anglais / English','Afrikaans','Akan','Albanais','Allemand / Deutsch','Amharique','Anglais','Arabe','Arménien','Assamais','Azéri','Bachkir','Bambara','Basque','Bengali','Biélorusse','Birman','Bosnien','Bulgare','Cachemiri','Catalan','Chichewa','Chinois','Cingalais','Coréen','Corse','Créolehaïtien','Croate','Danois','Dzongkha','Écossais','Espagnol / Español','Espéranto','Estonien','Ewe','Féroïen','Fidjien','Finnois','Français','Frisonoccidental','Galicien','Gallois','Ganda','Géorgien','Grecmoderne','Guarani','Gujarati','Haoussa','Hébreu','Hindi','Hongrois','Igbo','Indonésien','Inuktitut','Irlandais','Islandais','Italien / Italiano','Japonais','Kannada','Kazakh','Khmer','Kikuyu','Kinyarwanda','Kirghiz','Kirundi','Kurde','Lao','Latin','Letton','Limbourgeois','Lingala','Lituanien','Luxembourgeois','Macédonien','Malais','Malayalam','Maldivien','Malgache','Maltais','MaorideNouvelle-Zélande','Marathi','Mongol','Néerlandais / Nederlands','Népalais','Norvégien','Norvégienbokmål','Norvégiennynorsk','Occitan','Oriya','Ouïghour','Ourdou','Ouzbek','Pachto','Pendjabi','Persan','Polonais','Portugais','Roumain','Russe','Samoan','Sango','Sanskrit','Sarde','Serbe','Shona','Sindhi','Slovaque','Slovène','Somali','SothoduSud','Soundanais','Suédois','Swahili','Swati','Tadjik','Tagalog','Tahitien','Tamoul','Tatar','Tchèque','Télougou','Thaï','Tibétain','Tigrigna','Tongien','Tsonga','Tswana','Turc','Turkmène','Twi','Ukrainien','Vietnamien','Wolof','Xhosa','Yiddish']

@st.cache_data        
def find_lang_label(lang_sel):
    global lang_tgt, label_lang
    return label_lang[lang_tgt.index(lang_sel)]

def run():

    st.sidebar.image(
        "assets/demosthene_logo.png",
        width=270,
    )
    with st.sidebar:
        tab_name = option_menu(None, list(TABS.keys()),
                               # icons=['house', 'bi-binoculars', 'bi bi-graph-up', 'bi-chat-right-text','bi-book', 'bi-body-text'], menu_icon="cast", default_index=0,
                               icons=['house', 'binoculars', 'graph-up', 'search','book', 'chat-right-text','controller'], menu_icon="cast", default_index=0,
                               styles={"container": {"padding": "0!important","background-color": "#10b8dd", "border-radius": "0!important"},
                                       "nav-link": {"font-size": "1rem", "text-align": "left", "margin":"0em", "padding": "0em",
                                                    "padding-left": "0.2em", "--hover-color": "#eee", "font-weight": "400",
                                                    "font-family": "Source Sans Pro, sans-serif"}
                                        })
    # tab_name = st.sidebar.radio("", list(TABS.keys()), 0)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"## {config.PROMOTION}")

    st.sidebar.markdown("### Team members:")
    for member in config.TEAM_MEMBERS:
        st.sidebar.markdown(member.sidebar_markdown(), unsafe_allow_html=True)

    with st.sidebar:
        st.selectbox("langue:",lang_tgt, format_func = find_lang_label, key="Language", label_visibility="hidden")

    tab = TABS[tab_name]
    tab.run()

if __name__ == "__main__":
    run()
