import streamlit as st
from PIL import Image
import os
import ast
import contextlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from gensim import corpora
import networkx as nx
from sklearn.manifold import TSNE
from gensim.models import KeyedVectors


title = "Data Vizualization"
sidebar_name = "Data Vizualization"

with contextlib.redirect_stdout(open(os.devnull, "w")):
    nltk.download('stopwords')

# Première ligne à charger
first_line = 0
# Nombre maximum de lignes à charger
max_lines = 140000
if ((first_line+max_lines)>137860):
    max_lines = max(137860-first_line ,0)
# Nombre maximum de ligne à afficher pour les DataFrame
max_lines_to_display = 50

@st.cache_data
def load_data(path):
    
    input_file = os.path.join(path)
    with open(input_file, "r",  encoding="utf-8") as f:
        data = f.read()
        
    # On convertit les majuscules en minulcule
    data = data.lower()
    
    data = data.split('\n')
    return data[first_line:min(len(data),first_line+max_lines)]

@st.cache_data
def load_preprocessed_data(path,data_type):
    
    input_file = os.path.join(path)
    if data_type == 1:
        return pd.read_csv(input_file, encoding="utf-8", index_col=0)
    else:
        with open(input_file, "r",  encoding="utf-8") as f:
            data = f.read()
            data = data.split('\n')
        if data_type==0:
            data=data[:-1]
        elif data_type == 2:
            data=[eval(i) for i in data[:-1]]
        elif data_type ==3:
            data2 = []
            for d in data[:-1]:
                data2.append(ast.literal_eval(d))
            data=data2
        return data
    
@st.cache_data
def load_all_preprocessed_data(lang):
    txt           =load_preprocessed_data('../data/preprocess_txt_'+lang,0)
    corpus        =load_preprocessed_data('../data/preprocess_corpus_'+lang,0)
    txt_split     = load_preprocessed_data('../data/preprocess_txt_split_'+lang,3)
    df_count_word = pd.concat([load_preprocessed_data('../data/preprocess_df_count_word1_'+lang,1), load_preprocessed_data('../data/preprocess_df_count_word2_'+lang,1)]) 
    sent_len      =load_preprocessed_data('../data/preprocess_sent_len_'+lang,2)
    vec_model= KeyedVectors.load_word2vec_format('../data/mini.wiki.'+lang+'.align.vec')
    return txt, corpus, txt_split, df_count_word,sent_len, vec_model

#Chargement des textes complet dans les 2 langues
full_txt_en, full_corpus_en, full_txt_split_en, full_df_count_word_en,full_sent_len_en, vec_model_en = load_all_preprocessed_data('en')
full_txt_fr, full_corpus_fr, full_txt_split_fr, full_df_count_word_fr,full_sent_len_fr, vec_model_fr = load_all_preprocessed_data('fr')


def plot_word_cloud(text, title, masque, stop_words, background_color = "white"):
    
    mask_coloring = np.array(Image.open(str(masque)))
    # Définir le calque du nuage des mots
    wc = WordCloud(background_color=background_color, max_words=200, 
                   stopwords=stop_words, mask = mask_coloring, 
                   max_font_size=50, random_state=42)
    # Générer et afficher le nuage de mots
    fig=plt.figure(figsize= (20,10))
    plt.title(title, fontsize=25, color="green")
    wc.generate(text)
    
    # getting current axes
    a = plt.gca()
 
    # set visibility of x-axis as False
    xax = a.axes.get_xaxis()
    xax = xax.set_visible(False)
 
    # set visibility of y-axis as False
    yax = a.axes.get_yaxis()
    yax = yax.set_visible(False)
    
    plt.imshow(wc)
    # plt.show()
    st.pyplot(fig)
 
def drop_df_null_col(df):
    # Check if all values in each column are 0
    columns_to_drop = df.columns[df.eq(0).all()]
    # Drop the columns with all values as 0
    return df.drop(columns=columns_to_drop)

def calcul_occurence(df_count_word):
    nb_occurences = pd.DataFrame(df_count_word.sum().sort_values(axis=0,ascending=False))
    nb_occurences.columns = ['occurences']
    nb_occurences.index.name = 'mot'
    nb_occurences['mots'] = nb_occurences.index
    return nb_occurences

def dist_frequence_mots(df_count_word):
    
    df_count_word = drop_df_null_col(df_count_word)
    nb_occurences = calcul_occurence(df_count_word)
    
    sns.set()
    fig = plt.figure() #figsize=(4,4)
    plt.title("Nombre d'apparitions des mots", fontsize=16)

    chart = sns.barplot(x='mots',y='occurences',data=nb_occurences.iloc[:40]); 
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right', size=8)
    st.pyplot(fig)
    
def dist_longueur_phrase(sent_len,sent_len2, lang1, lang2 ):
    '''
    fig = px.histogram(sent_len, nbins=16, range_x=[3, 18],labels={'count': 'Count', 'variable': 'Nb de mots'},
                       color_discrete_sequence=['rgb(200, 0, 0)'],  # Couleur des barres de l'histogramme
                       opacity=0.7)
    fig.update_traces(marker=dict(color='rgb(200, 0, 0)', line=dict(color='white', width=2)), showlegend=False,)
    fig.update_layout(
        title={'text': 'Distribution du nb de mots/phrase', 'y':1.0, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
        title_font=dict(size=28),  # Ajuste la taille de la police du titre
        xaxis_title=None,
        xaxis=dict(
            title_font=dict(size=30), # Ajuste la taille de la police de l'axe X
            tickfont=dict(size=22), 
            showgrid=True, gridcolor='white'
            ), 
        yaxis_title='Count',
        yaxis=dict(
            title_font= dict(size=30, color='black'), # Ajuste la taille de la police de l'axe Y
            title_standoff=10,  # Éloigne le label de l'axe X du graphique
            tickfont=dict(size=22), 
            showgrid=True, gridcolor='white'
            ), 
        margin=dict(l=20, r=20, t=40, b=20), # Ajustez les valeurs de 'r' pour déplacer les commandes à droite
        # legend=dict(x=1, y=1), # Position de la légende à droite en haut
        # width = 600
        height=600,  # Définir la hauteur de la figure
        plot_bgcolor='rgba(220, 220, 220, 0.6)',
    )
    st.plotly_chart(fig, use_container_width=True)
    '''
    df = pd.DataFrame({lang1:sent_len,lang2:sent_len2})
    sns.set()
    fig = plt.figure() # figsize=(12, 6*row_nb)

    fig.tight_layout()
    chart = sns.histplot(df, color=['r','b'], label=[lang1,lang2], binwidth=1, binrange=[2,22], element="step", 
                         common_norm=False, multiple="layer", discrete=True, stat='proportion')
    plt.xticks([2,4,6,8,10,12,14,16,18,20,22])
    chart.set(title='Distribution du nombre de mots sur '+str(len(sent_len))+' phrase(s)'); 
    st.pyplot(fig)

    '''
    # fig = ff.create_distplot([sent_len], ['Nb de mots'],bin_size=1, colors=['rgb(200, 0, 0)'])

    distribution = pd.DataFrame({'Nb mots':sent_len, 'Nb phrases':[1]*len(sent_len)})
    fig = px.histogram(distribution, x='Nb mots', y='Nb phrases', marginal="box",range_x=[3, 18], nbins=16, hover_data=distribution.columns)
    fig.update_layout(height=600,title={'text': 'Distribution du nb de mots/phrase', 'y':1.0, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'})
    fig.update_traces(marker=dict(color='rgb(200, 0, 0)', line=dict(color='white', width=2)), showlegend=False,)
    st.plotly_chart(fig, use_container_width=True)
    '''

def find_color(x,min_w,max_w):
    b_min = 0.0*(max_w-min_w)+min_w
    b_max = 0.05*(max_w-min_w)+min_w
    x = max(x,b_min)
    x = min(b_max, x)
    c = (x - b_min)/(b_max-b_min)
    return round(c)

def graphe_co_occurence(txt_split,corpus):

    dic = corpora.Dictionary(txt_split) # dictionnaire de tous les mots restant dans le token
    # Equivalent (ou presque) de la DTM : DFM, Document Feature Matrix
    dfm = [dic.doc2bow(tok) for tok in txt_split]

    mes_labels = [k for k, v in dic.token2id.items()]

    from gensim.matutils import corpus2csc
    term_matrice = corpus2csc(dfm)

    term_matrice = np.dot(term_matrice, term_matrice.T)

    for i in range(len(mes_labels)):
        term_matrice[i,i]= 0
    term_matrice.eliminate_zeros()

    G = nx.from_scipy_sparse_matrix(term_matrice)
    G.add_nodes = dic
    pos=nx.spring_layout(G, k=5)  # position des nodes

    importance = dict(nx.degree(G))
    importance = [round((v**1.3)) for v in importance.values()]
    edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
    max_w = max(weights)
    min_w = min(weights)
    edge_color = [find_color(weights[i],min_w,max_w)  for i in range(len(weights))]
    width = [(weights[i]-min_w)*3.4/(max_w-min_w)+0.2 for i in range(len(weights))]
    alpha = [(weights[i]-min_w)*0.3/(max_w-min_w)+0.3 for i in range(len(weights))]

    fig = plt.figure();

    nx.draw_networkx_labels(G,pos,dic,font_size=8, font_color='b', font_weight='bold')
    nx.draw_networkx_nodes(G,pos, dic, \
                           node_color= importance, # range(len(importance)), #"tab:red", \
                           node_size=importance, \
                           cmap=plt.cm.RdYlGn, #plt.cm.Reds_r, \
                           alpha=0.4);
    nx.draw_networkx_edges(G,pos,width=width,edge_color=edge_color, alpha=alpha,edge_cmap=plt.cm.RdYlGn)  # [1] * len(width)

    plt.axis("off");
    st.pyplot(fig)

def proximite():
    global vec_model_en,vec_model_fr

    # Creates and TSNE model and plots it"
    labels = []
    tokens = []

    nb_words = st.slider('Nombre de mots à afficher :',10,50, value=20)
    df = pd.read_csv('../data/dict_we_en_fr',header=0,index_col=0, encoding ="utf-8", keep_default_na=False)
    words_en = df.index.to_list()[:nb_words]
    words_fr = df['Francais'].to_list()[:nb_words]

    for word in words_en: 
        tokens.append(vec_model_en[word])
        labels.append(word)
    for word in words_fr: 
        tokens.append(vec_model_fr[word])
        labels.append(word)
    tokens = pd.DataFrame(tokens)

    tsne_model = TSNE(perplexity=10, n_components=2, init='pca', n_iter=2000, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    fig =plt.figure(figsize=(16, 16)) 
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    for i in range(len(x)):
        if i<nb_words  : color='green'
        else: color='blue'
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom',
                     color= color,
                     size=20)
    plt.title("Proximité des mots anglais avec leur traduction", fontsize=30, color="green")
    plt.legend(loc='best');
    st.pyplot(fig)
    

def run():
    
    global max_lines, first_line, Langue
    global full_txt_en, full_corpus_en, full_txt_split_en, full_df_count_word_en,full_sent_len_en, vec_model_en 
    global full_txt_fr, full_corpus_fr, full_txt_split_fr, full_df_count_word_fr,full_sent_len_fr, vec_model_fr 
    
    st.write("")
    st.title(title)

    # 
    st.write("## **Paramètres :**\n")
    Langue = st.radio('Langue:',('Anglais','Français'), horizontal=True)
    first_line = st.slider('No de la premiere ligne à analyser :',0,137859)
    max_lines = st.select_slider('Nombre de lignes à analyser :',
                              options=[1,5,10,15,100, 500, 1000,'Max'])
    if max_lines=='Max':
        max_lines=137860
    if ((first_line+max_lines)>137860):
        max_lines = max(137860-first_line,0)
     
    # Chargement des textes sélectionnés (max lignes = max_lines)
    last_line = first_line+max_lines
    if (Langue == 'Anglais'):
        txt_en = full_txt_en[first_line:last_line]
        corpus_en = full_corpus_en[first_line:last_line]
        txt_split_en = full_txt_split_en[first_line:last_line]
        df_count_word_en =full_df_count_word_en.loc[first_line:last_line-1]
        sent_len_en = full_sent_len_en[first_line:last_line]
        sent_len_fr = full_sent_len_fr[first_line:last_line]
    else:
        txt_fr = full_txt_fr[first_line:last_line]
        corpus_fr = full_corpus_fr[first_line:last_line]
        txt_split_fr = full_txt_split_fr[first_line:last_line]
        df_count_word_fr =full_df_count_word_fr.loc[first_line:last_line-1]
        sent_len_fr = full_sent_len_fr[first_line:last_line]
        sent_len_en = full_sent_len_en[first_line:last_line]
        
    if (Langue=='Anglais'):
        st.dataframe(pd.DataFrame(data=full_txt_en,columns=['Texte']).loc[first_line:last_line-1].head(max_lines_to_display), width=800)
    else:
        st.dataframe(pd.DataFrame(data=full_txt_fr,columns=['Texte']).loc[first_line:last_line-1].head(max_lines_to_display), width=800)
    st.write("")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["World Cloud", "Frequence","Distribution longueur", "Co-occurence", "Proximité"])

    with tab1:
        st.subheader("World Cloud")
        st.markdown(
            """
            On remarque, en changeant de langue, que certains mot de taille importante dans une langue,
            apparaissent avec une taille identique dans l'autre langue.
            La traduction mot à mot sera donc peut-être bonne.
            """
        )
        if (Langue == 'Anglais'):
            text = ""
            # Initialiser la variable des mots vides
            stop_words = set(stopwords.words('english'))
            for e in txt_en : text += e
            plot_word_cloud(text, "English words corpus", "../images/coeur.png", stop_words)
        else:
            text = ""
            # Initialiser la variable des mots vides
            stop_words = set(stopwords.words('french'))
            for e in txt_fr : text += e
            plot_word_cloud(text,"Mots français du corpus", "../images/coeur.png", stop_words)
            
    with tab2:
        st.subheader("Frequence d'apparition des mots")
        st.markdown(
            """
            On remarque, en changeant de langue, que certains mot fréquents dans une langue,
            apparaissent aussi fréquemment dans l'autre langue.
            Cela peut nous laisser penser que la traduction mot à mot sera peut-être bonne.
            """
        )
        if (Langue == 'Anglais'):
            dist_frequence_mots(df_count_word_en)
        else:
            dist_frequence_mots(df_count_word_fr)
    with tab3:
        st.subheader("Distribution des longueurs de phases")
        st.markdown(
            """
            Malgré quelques différences entre les 2 langues (les phrases anglaises sont généralement un peu plus courtes),
            on constate une certaine similitude dans les ditributions de longueur de phrases.
            Cela peut nous laisser penser que la traduction mot à mot ne sera pas si mauvaise.
            """
        )
        if (Langue == 'Anglais'):
            dist_longueur_phrase(sent_len_en, sent_len_fr, 'Anglais','Français')
        else:
            dist_longueur_phrase(sent_len_fr, sent_len_en, 'Français', 'Anglais')
    with tab4:
        st.subheader("Co-occurence des mots dans une phrase") 
        if (Langue == 'Anglais'):
            graphe_co_occurence(txt_split_en[:1000],corpus_en)
        else:
            graphe_co_occurence(txt_split_fr[:1000],corpus_fr)
    with tab5:
        st.subheader("Proximité sémantique des mots (Word Embedding)") 
        st.markdown(
            """
            MUSE est une bibliothèque Python pour l'intégration de mots multilingues, qui fournit
            notamment des "Word Embedding" multilingues  
            Facebook fournit des dictionnaires de référence. Ces embeddings sont des embeddings fastText Wikipedia pour 30 langues qui ont été alignés dans un espace espace vectoriel unique.
            Dans notre cas, nous avons utilisé 2 mini-dictionnaires d'environ 3000 mots (Français et Anglais).  
              
            En novembre 2015, l'équipe de recherche de Facebook a créé fastText qui est une extension de la bibliothèque word2vec. 
            Elle s'appuie sur Word2Vec en apprenant des représentations vectorielles pour chaque mot et les n-grammes trouvés dans chaque mot.  
            """
        )
        st.write("")
        proximite()
        