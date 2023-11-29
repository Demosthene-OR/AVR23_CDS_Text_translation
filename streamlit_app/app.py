import streamlit as st
import os.path
from collections import OrderedDict
from streamlit_option_menu import option_menu
# Define TITLE, TEAM_MEMBERS and PROMOTION values, in config.py.
import config
from tabs.custom_vectorizer import custom_tokenizer, custom_preprocessor
import os

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

# Define the root folders depending on local/cloud run
thisfile = os.path.abspath(__file__)
if ('/' in thisfile): 
    os.chdir(os.path.dirname(thisfile))

# Nécessaire pour la version windows 11
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
        (intro.sidebar_name, intro),
        (exploration_tab.sidebar_name, exploration_tab),
        (data_viz_tab.sidebar_name, data_viz_tab),
        (id_lang_tab.sidebar_name, id_lang_tab),
        (modelisation_dict_tab.sidebar_name, modelisation_dict_tab),
        (modelisation_seq2seq_tab.sidebar_name, modelisation_seq2seq_tab),
        (game_tab.sidebar_name, game_tab ),
    ]
)


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

    tab = TABS[tab_name]
    tab.run()


if __name__ == "__main__":
    run()
