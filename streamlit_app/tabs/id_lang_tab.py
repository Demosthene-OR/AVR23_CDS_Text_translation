import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import tiktoken
import random
import joblib
import json
import csv
from transformers import pipeline
import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.utils import plot_model
from filesplit.merge import Merge
from extra_streamlit_components import tab_bar, TabBarItemData
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import naive_bayes
from translate_app import tr

title = "Identification de langue"
sidebar_name = "Identification de langue"
dataPath = st.session_state.DataPath

# CountVectorizer a une liste de phrase en entrée.
# Cette fonction met les données d'entrée dans le bon format
def format_to_vectorize(data):
    X_tok = []
    if "DataFrame" in str(type(data)):sentences = data.tolist()
    elif "str" in str(type(data)):
        sentences =[data]
    else: sentences = data
                          
    for sentence in sentences:
        X_tok.append(sentence) 
    return X_tok

def create_BOW(data):
    global vectorizer
    
    X_tok = format_to_vectorize(data)
    X = vectorizer.transform(X_tok)
    return X

def load_vectorizer(tokenizer):
    global dict_token, dict_ids, nb_token
    
    path = dataPath+'/vectorizer_tiktoken_big.pkl'
    vectorizer = joblib.load(path)
    dict_token = {tokenizer.decode([cle]): cle for cle, valeur in vectorizer.vocabulary_.items()}
    dict_ids = {cle: tokenizer.decode([cle]) for cle, valeur in vectorizer.vocabulary_.items()} #dict_ids.items()}
    nb_token = len(vectorizer.vocabulary_)
    return vectorizer

def lang_id_nb(sentences):
    global lan_to_language

    if "str" in str(type(sentences)):
        return lan_to_language[clf_nb.predict(create_BOW(sentences))[0]]
    else: return [lan_to_language[l] for l in clf_nb.predict(create_BOW(sentences))]

@st.cache_resource
def init_nb_identifier():
    
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    # Chargement du classificateur sauvegardé
    clf_nb = joblib.load(dataPath+"/id_lang_tiktoken_nb_sparse_big.pkl")
    vectorizer = load_vectorizer(tokenizer)

    # Lisez le contenu du fichier JSON
    with open(dataPath+'/multilingue/lan_to_language.json', 'r') as fichier:
        lan_to_language = json.load(fichier)
    return tokenizer, dict_token, dict_ids, nb_token, lan_to_language, clf_nb, vectorizer

def encode_text(textes):
    global tokenizer

    max_length=250
    sequences = tokenizer.encode_batch(textes)
    return pad_sequences(sequences, maxlen=max_length, padding='post')

def read_list_lan():

    with open(dataPath+'/multilingue/lan_code.csv', 'r') as fichier_csv:
        reader = csv.reader(fichier_csv)
        lan_code = next(reader)
        return lan_code

@st.cache_resource
def init_dl_identifier():

    label_encoder = LabelEncoder()
    list_lan = read_list_lan()
    lan_identified = [lan_to_language[l] for l in list_lan]
    label_encoder.fit(list_lan)
    merge = Merge(dataPath+"/dl_id_lang_split",  dataPath, "dl_tiktoken_id_language_model.h5").merge(cleanup=False)
    dl_model = keras.models.load_model(dataPath+"/dl_tiktoken_id_language_model.h5")
    return dl_model, label_encoder, list_lan, lan_identified

def lang_id_dl(sentences):
    global dl_model, label_encoder
    
    if "str" in str(type(sentences)): predictions = dl_model.predict(encode_text([sentences]))
    else:  predictions = dl_model.predict(encode_text(sentences))
    # Décodage des prédictions en langues
    predicted_labels_encoded = np.argmax(predictions, axis=1)
    predicted_languages = label_encoder.classes_[predicted_labels_encoded]
    if "str" in str(type(sentences)): return lan_to_language[predicted_languages[0]]
    else: return [l for l in predicted_languages]

@st.cache_resource
def init_lang_id_external():

    lang_id_model_ext = pipeline('text-classification',model="papluca/xlm-roberta-base-language-detection")
    dict_xlmr  = {"ar":"ara", "bg":"bul", "de":"deu", "el": "ell", "en":"eng", "es":"spa", "fr":"fra", "hi": "hin","it":"ita","ja":"jpn", \
                  "nl":"nld", "pl":"pol", "pt":"por", "ru":"rus", "sw":"swh", "th":"tha", "tr":"tur", "ur": "urd", "vi":"vie", "zh":"cmn"}
    sentence_test = pd.read_csv(dataPath+'//multilingue/sentence_test_extract.csv')
    sentence_test = sentence_test[:4750]
    # Instanciation d'un exemple
    exemples = ["Er weiß überhaupt nichts über dieses Buch",                               # Phrase 0
                "Umbrellas sell well",                                                     # Phrase 1
                "elle adore les voitures très luxueuses, et toi ?",                        # Phrase 2
                "she loves very luxurious cars, don't you?",                               # Phrase 3
                "Vogliamo visitare il Colosseo e nuotare nel Tevere",                      # Phrase 4
                "vamos a la playa",                                                        # Phrase 5
                "Te propongo un trato",                                                    # Phrase 6
                "she loves you much, mais elle te hait aussi and das ist traurig",         # Phrase 7  # Attention à cette phrase trilingue
                "Elle a de belles loches"                                                  # Phrase 8
                ]   

    lang_exemples = ['deu','eng','fra','eng','ita','spa','spa','fra','fra']
    return lang_id_model_ext, dict_xlmr, sentence_test, lang_exemples, exemples

@st.cache_data
def display_acp(title, comment):
    data = np.load(dataPath+'/data_lang_id_acp.npz')
    X_train_scaled = data['X_train_scaled']
    y_train_pred = data['y_train_pred']
    label_arrow = ['.', ',', '?', ' a', ' de', ' la', ' que', 'Tom', ' un', ' the', ' in', \
                    ' to', 'I', "'", 'i', ' le', ' en', ' es', 'é', ' l', '!', 'o', ' ist', \
                    ' pas', ' Tom', ' me', ' di', 'Ich', ' is', 'Je', ' nicht', ' you', \
                    ' die', ' à', ' el', ' est', 'a', 'en', ' d', ' è', ' ne', ' se', ' no', \
                    ' una', ' zu', 'Il', '¿', ' of', ' du', "'t", 'ato', ' der', ' il', \
                    ' n', 'El', ' non', ' che', 'are', ' con', 'ó', ' was', 'La', 'No', \
                    ' ?', 'es', 'le', 'L', ' and', ' des', ' s', ' ich', 'as', 'S', ' per', \
                    ' das', ' und', ' ein', 'e', "'s", 'u', ' y', 'He', 'z', 'er', ' m', \
                    'st', ' les', 'Le', ' I', 'ar', 'te', 'Non', 'The', ' er', 'ie', ' v', \
                    ' c', "'est", ' ha', ' den']

    pca = PCA(n_components=2)

    X_new = pca.fit_transform(X_train_scaled)
    coeff = pca.components_.transpose()
    xs = X_new[:, 0]
    ys = X_new[:, 1]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    principalDf = pd.DataFrame({'PC1': xs*scalex, 'PC2': ys * scaley})
    finalDF = pd.concat([principalDf, pd.Series(y_train_pred, name='Langue')], axis=1)

    sns.set_context("poster")  #  Valeur possible:"notebook", "talk", "poster", ou "paper"
    plt.rc("axes", titlesize=32,titleweight='bold')  # Taille du titre de l'axe
    plt.rc("axes", labelsize=18,labelweight='bold')  # Taille des étiquettes de l'axe
    plt.rc("xtick", labelsize=14)  # Taille des étiquettes de l'axe des x
    plt.rc("ytick", labelsize=14)  # Taille des étiquettes de l'axe des y

    st.write(comment)
    st.write("")
    fig = plt.figure(figsize=(20, 15))
    sns.scatterplot(x='PC1', y='PC2', hue='Langue', data=finalDF, alpha=0.5)
    for i in range(50):
        plt.arrow(0, 0, coeff[i, 0]*1.5, coeff[i, 1]*0.8,color='k', alpha=0.08, head_width=0.01, )
        plt.text(coeff[i, 0]*1.5, coeff[i, 1] * 0.8, label_arrow[i], color='k', weight='bold')

    plt.title(title) 
    plt.xlim(-0.4, 0.45)
    plt.ylim(-0.15, 0.28);
    st.pyplot(fig)
    return

@st.cache_data
def read_BOW_examples():
    return pd.read_csv(dataPath+'/lang_id_small_BOW.csv')

def analyse_nb(sel_phrase):
    global lang_exemples,exemples

    def create_small_BOW(s):
        encodage = tokenizer.encode(s)
        sb = [0] * (df_BOW.shape[1]-1)
        nb_unique_token = 0
        for i in range(df_BOW.shape[1]-1):
            for t in encodage:
                if df_BOW.columns[i]==str(t):
                    sb[i] += 1
            if sb[i] > 0: nb_unique_token +=1
        return sb, nb_unique_token

    st.write("#### **"+tr("Probabilité d'appartenance de la phrase à une langue")+" :**")
    st.image("./assets/formule_proba_naive_bayes.png")
    st.write(tr("où **C** est la classe (lan_code), **Fi** est la caractéristique i du BOW, **Z** est l'\"evidence\" servant à regulariser la probabilité"))
    st.write("")
    nb_lang = 5
    lan_code = ['deu','eng','fra','spa','ita']
    lan_color = {'deu':'violet','eng':'green','fra':'red','spa':'blue','ita':'orange'}
    df_BOW = read_BOW_examples()

    clf_nb2 = naive_bayes.MultinomialNB()
    clf_nb2.fit(df_BOW.drop(columns='lan_code').values.tolist(), df_BOW['lan_code'].values.tolist()) 

    nb_phrases_lang =[]
    for l in lan_code:
        nb_phrases_lang.append(sum(df_BOW['lan_code']==l))
    st.write(tr("Phrase à analyser")+" :",'**:'+lan_color[lang_exemples[sel_phrase]]+'['+lang_exemples[sel_phrase],']** - **"'+exemples[sel_phrase]+'"**')

    # Tokenisation et encodage de la phrase
    encodage = tokenizer.encode(exemples[sel_phrase])

    # Création du vecteur BOW de la phrase
    bow_exemple,  nb_unique_token = create_small_BOW(exemples[sel_phrase])
    st.write(tr("Nombre de tokens retenus dans le BOW")+": "+ str(nb_unique_token))
    masque_tokens_retenus = [(1 if token in list(dict_ids.keys()) else 0) for token in encodage]
    str_token = " "
    for i in range(len(encodage)):
        if masque_tokens_retenus[i]==1:
            if (i%2) ==0:
                str_token += "**:red["+tokenizer.decode([encodage[i]])+"]** "
            else:
                str_token += "**:violet["+tokenizer.decode([encodage[i]])+"]** "
        else: str_token += ":green["+tokenizer.decode([encodage[i]])+"] "

    st.write(tr("Tokens se trouvant dans le modèle (en")+" :red["+tr("rouge")+"] "+tr("ou")+" :violet["+tr("violet")+"]) :"+str_token+" ")

    st.write("")
    # Afin de continuer l'analyse on ne garde que les token de la phrase disponibles dans le BOW
    token_used = [str(encodage[i]) for i in range(len(encodage)) if (masque_tokens_retenus[i]==1)]


    # Calcul du nombre d'apparition de ces tokens dans le BOW pour chaque langue, et stockage dans un DataFrame df_count
    def compter_non_zero(colonne):
        return (colonne != 0).sum()

    votes = []
    for i in range(nb_lang):
        #votes.append(list(df_BOW[token_used].loc[df_BOW['lan_code']==lan_code[i]].sum(axis=0)))
        votes.append(list(df_BOW[token_used].loc[df_BOW['lan_code']==lan_code[i]].apply(compter_non_zero)))
    
    col_name = [str(i+1)+'-'+tokenizer.decode([int(token_used[i])]) for i in range(len(token_used))]
    df_count = pd.DataFrame(data=votes,columns=token_used, index=lan_code)
    df_count.columns = col_name
    st.write("\n**"+tr("Nombre d'apparitions des tokens, dans chaque langue")+"**")

    # Lissage de Laplace n°1 (Laplace smoothing )
    # df_count = df_count+1

    st.dataframe(df_count)

    #########
    ######### 3. Calcul de la probabilité d'apparition de chaque token dans chaque langue
    df_proba = df_count.div(nb_phrases_lang, axis = 0)

    # Lissage de Laplace n°2 (Laplace smoothing )
    df_proba = df_proba.replace(0.0,0.0010) 

    # Initialisation de df_proba: Calcul de la probabilité conditionnelle d'appartenance de la phrase à une langue
    df_proba['Proba'] = 1
    # Itérer sur les colonnes et effectuez la multiplication pour chaque ligne
    for col in df_count.columns:
        df_proba['Proba'] *= df_proba[col]

    #########    
    ######### 4.  Calcul (par multiplication) de la probabilité d'appartenance de la phrase à une langue

    # Multiplication par la probabilité de la classe
    p_classe = [(nb_phrases_lang[i]/df_BOW.shape[0]) for i in range(len(nb_phrases_lang))]
    df_proba['Proba'] *= p_classe

    # Diviser par l'evidence
    evidence = df_proba['Proba'].sum(axis=0)
    df_proba['Proba'] *= 1/evidence
    df_proba['Proba'] = df_proba['Proba'].round(3)

    # Affichage de la matrice des probabilités
    st.write("**"+tr("Probabilités conditionnelles d'apparition des tokens retenus, dans chaque langue")+":**")
    st.dataframe(df_proba)
    str_token = "Lang proba max: "#&nbsp;"*20
    for i,token in enumerate(df_proba.columns[:-1]):
        str_token += '*'+token+'*:**:'+lan_color[df_proba[token].idxmax()]+'['+df_proba[token].idxmax()+']**'+"&nbsp;"*2 #8
    st.write(str_token)
    st.write("")

    st.write(tr("Langue réelle de la phrase")+"&nbsp;"*35+": **:"+lan_color[lang_exemples[sel_phrase]]+'['+lang_exemples[sel_phrase]+']**')
    st.write(tr("Langue dont la probabilité est la plus forte ")+": **:"+lan_color[df_proba['Proba'].idxmax()]+'['+df_proba['Proba'].idxmax(),"]** (proba={:.2f}".format(max(df_proba['Proba']))+")")
    prediction = clf_nb2.predict([bow_exemple]) 
    st.write(tr("Langue prédite par Naiva Bayes")+"&nbsp;"*23+": **:"+lan_color[prediction[0]]+'['+prediction[0]+"]** (proba={:.2f}".format(max(clf_nb2.predict_proba([bow_exemple])[0]))+")")
    st.write("")

    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    df_proba_sorted =df_proba.sort_index(ascending=True)
    axs[0].set_title(tr("Probabilités calculée manuellement"), fontsize=12)
    axs[0].barh(df_proba_sorted.index, df_proba_sorted['Proba'])
    axs[1].set_title(tr("Probabilités du classifieur Naive Bayes"), fontsize=12)
    axs[1].barh(df_proba_sorted.index, clf_nb2.predict_proba([bow_exemple])[0]);
    st.pyplot(fig)
    return

#@st.cache_data        
def find_exemple(lang_sel):
    global exemples
    return exemples[lang_sel]

def display_shapley(lang_sel):
    st.write("**"+tr("Analyse de l'importance de chaque token dans l'identification de la langue")+"**")
    st.image('assets/fig_schapley'+str(lang_sel)+'.png')
    st.write("**"+tr("Recapitulatif de l'influence des tokens sur la selection de la langue")+"**")
    st.image('assets/fig_schapley_recap'+str(lang_sel)+'.png')
    return

def run():
    global tokenizer, vectorizer, dict_token, dict_ids, nb_token, lan_to_language, clf_nb
    global dl_model, label_encoder, toggle_val, custom_sentence, list_lan, lan_identified
    global lang_exemples, exemples
   

    tokenizer, dict_token, dict_ids, nb_token, lan_to_language, clf_nb, vectorizer = init_nb_identifier()
    dl_model, label_encoder, list_lan, lan_identified = init_dl_identifier()
    lang_id_model_ext, dict_xlmr, sentence_test, lang_exemples, exemples= init_lang_id_external()

    st.write("")
    st.title(tr(title))
    st.write("## **"+tr("Explications")+" :**\n")
    st.markdown(tr(
        """
        Afin de mettre en oeuvre cette fonctionnalité nous avons utilisé un jeu d'entrainement multilinge de <b> 9.757.778 phrases dans 95 langues</b>.   
        Les 95 langues identifiées sont:
        """)
    , unsafe_allow_html=True)
    st.selectbox(label="Lang",options=sorted(lan_identified),label_visibility="hidden")
    st.markdown(tr(
        """
        Nous avons utilisé 2 méthodes pour identifier la langue d'un texte:  
        1. un classificateur **Naïve Bayes**  
        2. un modèle de **Deep Learning**  
        """)
    , unsafe_allow_html=True)
    st.markdown(tr(
        """
        Les 2 modèles ont un accuracy similaire sur le jeu de test: **:red[96% pour NB et 97,5% pour DL]**  
        <br>
        """)
        , unsafe_allow_html=True)
    
    chosen_id = tab_bar(data=[
        TabBarItemData(id="tab1", title=tr("Id. Naïve Bayes"), description=tr("avec le Bag Of Words")),
        TabBarItemData(id="tab2", title=tr("Id. Deep Learning"), description=tr(" avec Keras")),
        TabBarItemData(id="tab3", title=tr("Interpretabilité"), description=tr("du modèle Naïve Bayes "))],
        default="tab1")
    
    if (chosen_id == "tab1") or (chosen_id == "tab2"):
        st.write("## **"+tr("Paramètres")+" :**\n")

        toggle_val = st.toggle(tr('Phrase à saisir/Phrase test'), value=True, help=tr("Off = phrase à saisir, On = selection d'une phrase test parmi 9500 phrases"))
        if toggle_val:
            custom_sentence= st.selectbox(tr("Selectionnez une phrases test à identifier")+":", sentence_test['sentence'] )
        else:
            custom_sentence = st.text_area(label=tr("Saisir le texte dont vous souhaitez identifier la langue:"))
            st.button(label=tr("Validez"), type="primary")

        if custom_sentence!='':
            st.write("## **"+tr("Résultats")+" :**\n")
            md = """
                |"""+tr("Identifieur")+"""                          |"""+tr("Langue identifiée")+"""|
                |-------------------------------------|---------------|""" 
            md1 = ""
            if toggle_val:
                lan_reelle = sentence_test['lan_code'].loc[sentence_test['sentence']==custom_sentence].tolist()[0]
                md1 = """
                |"""+tr("Langue réelle")+"""                        |**:blue["""+lan_to_language[lan_reelle]+"""]**|"""
            md2 = """
                |"""+tr("Classificateur Naïve Bayes")+"""           |**:red["""+lang_id_nb(custom_sentence)+"""]**|
                |"""+tr("Modèle de Deep Learning")+"""           |**:red["""+lang_id_dl(custom_sentence)+"""]**|"""
            md3 = """
                |XLM-RoBERTa (Hugging Face)           |**:red["""+lan_to_language[dict_xlmr[lang_id_model_ext(custom_sentence)[0]['label']]]+"""]**|"""
            if toggle_val:
                if not (lan_reelle in list(dict_xlmr.values())):
                    md3=""

            st.markdown(md+md1+md2+md3, unsafe_allow_html=True)

        st.write("## **"+tr("Details sur la méthode")+" :**\n")
        if (chosen_id == "tab1"):
            st.markdown(tr(
                """
                Afin d'utiliser le classificateur Naïve Bayes, il nous a fallu:""")+"\n"+
                "* "+tr("Créer un Bag of Words de token..")+"\n"+
                "* "+tr("..Tokeniser le texte d'entrainement avec CountVectorizer et un tokenizer 'custom', **Tiktoken** d'OpenAI.  ")+"\n"+
                "* "+tr("Utiliser des matrices creuses (Sparse Matrix), car notre BOW contenait 10 Millions de lignes x 59122 tokens.  ")+"\n"+
                "* "+tr("Sauvegarder le vectorizer (non serialisable) et le classificateur entrainé.  ")
            , unsafe_allow_html=True)
            st.markdown(tr(
                """
                L'execution de toutes ces étapes est assez rapide: une dizaine de minutes  
                <br>
                Le résultat est très bon: L'Accuracy sur le jeu de test est = 
                **:red[96%]** sur les 95 langues, et **:red[99,1%]** sur les 5 langues d'Europe de l'Ouest (en,fr,de,it,sp)  
                <br>
                """)
            , unsafe_allow_html=True)
            st.markdown(tr(
                """
                **Note 1:** Les 2 modèles ont un accuracy similaire sur le jeu de test: **:red[96% pour NB et 97,5% pour DL]**  
                **Note 2:** Le modèle *XLM-RoBERTa* de Hugging Face (qui identifie 20 langues seulement) a une accuracy, sur notre jeu de test = **97,8%**, 
                versus **99,3% pour NB** et **99,2% pour DL** sur ces 20 langues.
                """)
            , unsafe_allow_html=True)
        else:
            st.markdown(tr(
                """
                Nous avons mis en oeuvre un modèle Keras avec une couche d'embedding et 4 couches denses (*Voir architecture ci-dessous*).  
                Nous avons utilisé le tokeniser <b>Tiktoken</b> d'OpenAI.  
                La couche d'embedding accepte 250 tokens, ce qui signifie que la détection de langue s'effectue sur approximativement les 200 premiers mots.  
                <br>
                """)
            , unsafe_allow_html=True)
            st.markdown(tr(
                """
                L'entrainement a duré plus de 10 heures..
                Finalement, le résultat est très bon: L'Accuracy sur le jeu de test est = 
                **:red[97,5%]** sur les 95 langues, et **:red[99,1%]** sur les 5 langues d'Europe de l'Ouest (en,fr,de,it,sp).  
                Néanmoins, la durée pour une prédiction est relativement longue: approximativement 5/100 de seconde  
                <br>
                """)
                , unsafe_allow_html=True)
            st.markdown(tr(
                """
                **Note 1:** Les 2 modèles ont un accuracy similaire sur le jeu de test: **:red[96% pour NB et 97,5% pour DL]**""")+"<br>"+
                tr("""
                **Note 2:** Le modèle *XLM-RoBERTa* de Hugging Face (qui identifie 20 langues seulement) a une accuracy, sur notre jeu de test = <b>97,8%</b>, 
                versus **99,3% pour NB** et **99,2% pour DL** sur ces 20 langues.  
                <br>
                """)
                , unsafe_allow_html=True)
            st.write("<center><h5>"+tr("Architecture du modèle utilisé")+":</h5></center>", unsafe_allow_html=True)
            plot_model(dl_model, show_shapes=True, show_layer_names=True, show_layer_activations=True,rankdir='TB',to_file='./assets/model_plot.png')
            col1, col2, col3 = st.columns([0.15,0.7,0.15])
            with col2:
                 st.image('./assets/model_plot.png',use_column_width="auto")
    elif (chosen_id == "tab3"):
        st.write("### **"+tr("Interpretabilité du classifieur Naïve Bayes sur 5 langues")+"**")
        st.write("##### "+tr("..et un Training set réduit (15000 phrases et 94 tokens)"))
        st.write("")

        chosen_id2 = tab_bar(data=[
            TabBarItemData(id="tab1", title=tr("Analyse en Compos. Princ."), description=""),
            TabBarItemData(id="tab2", title=tr("Simul. calcul NB"), description=""),
            TabBarItemData(id="tab3", title=tr("Shapley"), description="")],
            default="tab1")
        if (chosen_id2 == "tab1"):
            display_acp(tr("Importance des principaux tokens dans \n l'identification de langue par l'algorithme Naive Bayes"),tr("Affichage de 10 000 phrases (points) et des 50 tokens les + utilisés (flèches)")) 
        if (chosen_id2 == "tab2") or (chosen_id2 == "tab3"):
            sel_phrase = st.selectbox(tr('Selectionnez une phrase à "interpréter"')+':', range(9), format_func=find_exemple)
            if (chosen_id2 == "tab2"):
                analyse_nb(sel_phrase)
            if (chosen_id2 == "tab3"):
                display_shapley(sel_phrase)


        





    
