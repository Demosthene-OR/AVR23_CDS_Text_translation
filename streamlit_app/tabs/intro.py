import streamlit as st
from translate_app import tr

title = "Démosthène"
sidebar_name = "Introduction"


def run():

    # TODO: choose between one of these GIFs
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/1.gif")
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/2.gif")
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/3.gif")
    # st.image("assets/tough-communication.gif",use_column_width=True)
    
    st.write("")
    if st.session_state.Cloud == 0: 
        st.image("assets/miss-honey-glasses-off.gif",use_column_width=True)
    else:
        st.image("https://media.tenor.com/pfOeAfytY98AAAAC/miss-honey-glasses-off.gif",use_column_width=True)
        
    st.title(tr(title))
    st.markdown('''
                ## **'''+tr("Système de traduction adapté aux lunettes connectées")+'''**  
                ---
                ''')
    st.header("**"+tr("A propos")+"**")
    st.markdown(tr(
        """
        Ce projet a été réalisé dans le cadre d’une formation de Data Scientist, entre juin et novembre 2023.  
        <br>
        :red[**Démosthène**] est l'un des plus grands orateurs de l'Antiquité. Il savait s’exprimer,  et se faire comprendre. 
        Se faire comprendre est l’un des principaux objectifs de la traduction.  
        """)
    , unsafe_allow_html=True)
    st.markdown(tr(
        """
        Démosthène avait de gros problèmes d’élocution. 
        Il les a surmontés en s’entraînant à parler avec des cailloux dans la bouche, 
        à l’image de l’Intelligence Artificielle,  où des entraînements sont nécessaires pour obtenir de bons résultats. 
        Il nous a semblé pertinent de donner le nom de cet homme à un projet qu’il a fort bien illustré, il y a 2300 ans.
        """)
    , unsafe_allow_html=True)

    st.header("**"+tr("Contexte")+"**")
    st.markdown(tr(
        """
        Les personnes malentendantes communiquent difficilement avec autrui. Par ailleurs, toute personne se trouvant dans un pays étranger 
        dont il ne connaît pas la langue se retrouve dans la situation d’une personne malentendante.  
        """)
    , unsafe_allow_html=True)
    st.markdown(tr(
        """
        L’usage de lunettes connectées, dotées de la technologie de reconnaissance vocale et d’algorithmes IA de deep learning, permettrait 
        de détecter la voix d’un interlocuteur, puis d’afficher la transcription textuelle, sur les verres en temps réel.  
        À partir de cette transcription, il est possible d’:red[**afficher la traduction dans la langue du porteur de ces lunettes**].  
        """)
    , unsafe_allow_html=True)

    st.header("**"+tr("Objectifs")+"**")
    st.markdown(tr(
        """
        L’objectif de ce projet est de développer une brique technologique de traitement, de transcription et de traduction, 
        qui par la suite serait implémentable dans des lunettes connectées. Nous avons concentré nos efforts sur la construction 
        d’un :red[**système de traduction**] plutôt que sur la reconnaissance vocale, 
        et ce, pour tout type de public, afin de faciliter le dialogue entre deux individus ne pratiquant pas la même langue.  
        """)
    , unsafe_allow_html=True)
    st.markdown(tr(
        """
        Il est bien sûr souhaitable que le système puisse rapidement :red[**identifier la langue**] des phrases fournies. 
        Lors de la traduction, nous ne prendrons pas en compte le contexte des phrases précédentes ou celles préalablement traduites.  
        """)
    , unsafe_allow_html=True)
    st.markdown(tr(
        """

        Nous évaluerons la qualité de nos résultats en les comparant avec des systèmes performants tels que “[Google translate](https://translate.google.fr/)”
        """)
    , unsafe_allow_html=True)
    st.markdown(tr(
        """
        Le projet est enregistré sur "[Github](https://github.com/Demosthene-OR/AVR23_CDS_Text_translation)"  
        """)
    , unsafe_allow_html=True)

    '''
    sent = \
        """

        """
    st.markdown(tr(sent), unsafe_allow_html=True)
    '''