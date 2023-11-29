# Les 2 fonctions suivantes sont nécéssaires afin de sérialiser ces parametre de CountVectorizer
# et ainsi de sauvegarder le vectorizer pour un un usage ultérieur sans utiliser X_train pour  le réinitialiser
import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")

def custom_tokenizer(text):
    global tokenizer

    tokens = tokenizer.encode(text)  # Cela divise le texte en mots
    return tokens

def custom_preprocessor(text):
    return text