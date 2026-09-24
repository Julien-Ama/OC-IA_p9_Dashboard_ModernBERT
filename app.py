import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from collections import Counter
import re
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# ============================================================
# CONFIGURATION DE L'APPLICATION
# ============================================================

st.set_page_config(
    page_title="Dashboard P9",
    page_icon="🤖",
    layout="wide"
)

st.title("Dashboard P9 — Classification de produits")


# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/flipkart_com-ecommerce_sample_1050.csv"
    )

    # Création de la catégorie principale
    df["main_category"] = (
        df["product_category_tree"]
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .str.replace("'", "", regex=False)
        .str.split(">>")
        .str[0]
        .str.strip()
    )

    # Nombre de mots dans chaque description
    df["description_length"] = (
        df["description"]
        .fillna("")
        .str.split()
        .str.len()
    )

    return df


df = load_data()

# ============================================================
# MODERNBERT
# ============================================================

MODEL_PATH = "models/P9_ModernBERT_best"

LABELS = [
    "Baby Care",
    "Beauty and Personal Care",
    "Computers",
    "Home Decor & Festive Needs",
    "Home Furnishing",
    "Kitchen & Dining",
    "Watches"
]


@st.cache_resource
def load_modernbert():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.eval()

    return tokenizer, model


tokenizer, model = load_modernbert()

def predict_category(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():

        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )[0]

    predicted_id = torch.argmax(
        probabilities
    ).item()

    predicted_label = LABELS[predicted_id]

    confidence = probabilities[predicted_id].item()

    return (
        predicted_label,
        confidence,
        probabilities.numpy()
    )

# ============================================================
# NAVIGATION
# ============================================================

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Exploration des données",
        "🤖 Prédiction",
        "📈 BERT vs ModernBERT"
    ]
)


# ============================================================
# PAGE 1 — EXPLORATION DES DONNÉES
# ============================================================

if page == "📊 Exploration des données":

    st.header("Exploration des données")

    st.write(
        """
        Cette section présente une analyse exploratoire
        du dataset Flipkart utilisé pour la classification
        automatique des produits.
        """
    )

    # --------------------------------------------------------
    # INDICATEURS GÉNÉRAUX
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Produits",
        len(df)
    )

    col2.metric(
        "Catégories",
        df["main_category"].nunique()
    )

    col3.metric(
        "Longueur moyenne",
        f"{df['description_length'].mean():.0f} mots"
    )


    # ========================================================
    # 1 — RÉPARTITION DES CATÉGORIES
    # ========================================================

    st.subheader(
        "1. Répartition des produits par catégorie"
    )

    category_counts = (
        df["main_category"]
        .value_counts()
        .rename_axis("Catégorie")
        .reset_index(name="Nombre de produits")
    )

    fig_categories = px.bar(
        category_counts,
        x="Catégorie",
        y="Nombre de produits",
        text="Nombre de produits",
        title="Nombre de produits par catégorie"
    )

    fig_categories.update_layout(
        xaxis_title="Catégorie",
        yaxis_title="Nombre de produits"
    )

    st.plotly_chart(
        fig_categories,
        use_container_width=True
    )

    st.caption(
        """
        Le dataset est équilibré :
        chaque catégorie contient 150 produits.
        """
    )


    # ========================================================
    # 2 — LONGUEUR DES DESCRIPTIONS
    # ========================================================

    st.subheader(
        "2. Longueur des descriptions"
    )

    selected_category = st.selectbox(
        "Sélectionner une catégorie",
        ["Toutes les catégories"]
        + sorted(
            df["main_category"]
            .unique()
            .tolist()
        )
    )

    if selected_category == "Toutes les catégories":

        filtered_df = df

    else:

        filtered_df = df[
            df["main_category"]
            == selected_category
        ]


    fig_length = px.histogram(
        filtered_df,
        x="description_length",
        nbins=40,
        title=(
            "Distribution de la longueur des descriptions"
            f" — {selected_category}"
        )
    )

    fig_length.update_layout(
        xaxis_title="Nombre de mots",
        yaxis_title="Nombre de produits"
    )

    st.plotly_chart(
        fig_length,
        use_container_width=True
    )

    st.write(
        f"**Longueur moyenne :** "
        f"{filtered_df['description_length'].mean():.1f} mots"
    )


    # ========================================================
    # 3 — FRÉQUENCE DES MOTS
    # ========================================================

    st.subheader(
        "3. Mots les plus fréquents"
    )

    text = " ".join(
        filtered_df["description"]
        .fillna("")
        .astype(str)
    )

    # Extraction des mots
    words = re.findall(
        r"\b[a-zA-Z]{3,}\b",
        text.lower()
    )

    # Suppression de quelques mots anglais très fréquents
    stop_words = {
        "the",
        "and",
        "for",
        "with",
        "this",
        "that",
        "from",
        "your",
        "are",
        "you",
        "its",
        "has",
        "have",
        "will"
    }

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    word_counts = Counter(
        words
    ).most_common(20)

    words_df = pd.DataFrame(
        word_counts,
        columns=[
            "Mot",
            "Fréquence"
        ]
    )

    fig_words = px.bar(
        words_df,
        x="Fréquence",
        y="Mot",
        orientation="h",
        title=(
            "20 mots les plus fréquents"
            f" — {selected_category}"
        )
    )

    fig_words.update_layout(
        yaxis={
            "categoryorder":
            "total ascending"
        }
    )

    st.plotly_chart(
        fig_words,
        use_container_width=True
    )


    # ========================================================
    # 4 — WORDCLOUD
    # ========================================================

    st.subheader(
        "4. Nuage de mots"
    )

    clean_text = " ".join(words)

    if clean_text.strip():

        wordcloud = WordCloud(
            width=1200,
            height=500,
            background_color="white",
            collocations=False
        ).generate(
            clean_text
        )

        fig, ax = plt.subplots(
            figsize=(12, 5)
        )

        ax.imshow(
            wordcloud,
            interpolation="bilinear"
        )

        ax.axis("off")

        st.pyplot(fig)

    else:

        st.warning(
            """
            Aucun texte disponible
            pour cette catégorie.
            """
        )


# ============================================================
# PAGE 2 — PRÉDICTION
# ============================================================

elif page == "🤖 Prédiction":

    st.header("Classification d'un produit")

    st.write(
        """
        Utilisez ModernBERT pour prédire la catégorie d'un produit.
        Vous pouvez saisir votre propre description ou tester
        directement un produit du dataset Flipkart.
        """
    )

    tab1, tab2 = st.tabs([
        "✍️ Saisie libre",
        "🧪 Produit du dataset"
    ])


    # ========================================================
    # FONCTION D'AFFICHAGE DES RÉSULTATS
    # ========================================================

    def display_prediction(
        predicted_label,
        confidence,
        probabilities
    ):

        st.success(
            f"Catégorie prédite : **{predicted_label}**"
        )

        st.metric(
            "Confiance du modèle",
            f"{confidence * 100:.1f} %"
        )

        # Tableau des probabilités
        prob_df = pd.DataFrame({
            "Catégorie": LABELS,
            "Probabilité": probabilities
        })

        prob_df["Probabilité (%)"] = (
            prob_df["Probabilité"] * 100
        )

        prob_df = prob_df.sort_values(
            "Probabilité (%)",
            ascending=False
        )

        # Graphique
        fig_prob = px.bar(
            prob_df,
            x="Probabilité (%)",
            y="Catégorie",
            orientation="h",
            text="Probabilité (%)",
            title="Probabilité par catégorie"
        )

        fig_prob.update_traces(
            texttemplate="%{text:.1f} %"
        )

        fig_prob.update_layout(
            xaxis_title="Probabilité (%)",
            yaxis_title="Catégorie",
            yaxis={
                "categoryorder": "total ascending"
            }
        )

        st.plotly_chart(
            fig_prob,
            use_container_width=True
        )


    # ========================================================
    # ONGLET 1 — SAISIE LIBRE
    # ========================================================

    with tab1:

        st.subheader(
            "Tester une nouvelle description"
        )

        st.write(
            """
            Décrivez un produit en quelques mots ou quelques
            phrases. ModernBERT déterminera la catégorie
            qu'il considère comme la plus probable.
            """
        )

        description = st.text_area(
            "Description du produit à classifier",
            height=180,
            placeholder=(
                "Exemple : Men's analog wrist watch "
                "with leather strap and water resistant case."
            )
        )

        if st.button(
            "Classifier cette description",
            type="primary",
            key="predict_free"
        ):

            if not description.strip():

                st.warning(
                    "Veuillez saisir une description."
                )

            else:

                with st.spinner(
                    "Analyse avec ModernBERT..."
                ):

                    predicted_label, confidence, probabilities = (
                        predict_category(description)
                    )

                display_prediction(
                    predicted_label,
                    confidence,
                    probabilities
                )


    # ========================================================
    # ONGLET 2 — PRODUIT DU DATASET
    # ========================================================

    with tab2:

        st.subheader(
            "Tester ModernBERT sur un produit Flipkart"
        )

        st.write(
            """
            Sélectionnez un produit du dataset.
            Sa catégorie réelle sera comparée à la prédiction
            réalisée par ModernBERT.
            """
        )

        # On crée un libellé lisible pour la liste
        df_test_display = df.copy()

        df_test_display["display_name"] = (
            df_test_display["product_name"]
            .fillna("Produit sans nom")
            .astype(str)
        )

        selected_index = st.selectbox(
            "Choisir un produit",
            options=df_test_display.index,
            format_func=lambda i:
                df_test_display.loc[i, "display_name"]
        )

        selected_product = df_test_display.loc[
            selected_index
        ]

        st.write("**Description :**")

        st.write(
            selected_product["description"]
        )

        true_category = selected_product[
            "main_category"
        ]

        st.write(
            f"**Catégorie réelle :** {true_category}"
        )

        if st.button(
            "Tester ModernBERT",
            type="primary",
            key="predict_dataset"
        ):

            product_description = str(
                selected_product["description"]
            )

            with st.spinner(
                "Analyse avec ModernBERT..."
            ):

                predicted_label, confidence, probabilities = (
                    predict_category(
                        product_description
                    )
                )

            display_prediction(
                predicted_label,
                confidence,
                probabilities
            )

            # Comparaison avec la vraie catégorie
            if predicted_label == true_category:

                st.success(
                    "✅ Prédiction correcte"
                )

            else:

                st.error(
                    f"""
                    ❌ Prédiction incorrecte

                    Catégorie réelle : **{true_category}**

                    Catégorie prédite : **{predicted_label}**
                    """
                )


# ============================================================
# PAGE 3 — COMPARAISON BERT / MODERNBERT
# ============================================================

elif page == "📈 BERT vs ModernBERT":

    st.header(
        "Comparaison BERT vs ModernBERT"
    )

    st.write(
        """
        Cette section compare les performances de BERT
        et ModernBERT sur le dataset métier Flipkart,
        puis sur la tâche SST-2 du benchmark GLUE.
        """
    )


    # ========================================================
    # FLIPKART
    # ========================================================

    st.subheader(
        "Dataset métier — Flipkart"
    )

    flipkart_results = pd.DataFrame({

        "Modèle": [
            "BERT optimisé",
            "ModernBERT optimisé"
        ],

        "Accuracy": [
            0.9367,
            0.9367
        ],

        "Macro-F1": [
            0.9365,
            0.9361
        ]
    })

    st.dataframe(
        flipkart_results,
        use_container_width=True,
        hide_index=True
    )

    fig_flipkart = px.bar(
        flipkart_results,
        x="Modèle",
        y=[
            "Accuracy",
            "Macro-F1"
        ],
        barmode="group",
        title=(
            "Performances sur le dataset Flipkart"
        ),
        range_y=[0, 1]
    )

    fig_flipkart.update_layout(
        yaxis_title="Score",
        xaxis_title="Modèle"
    )

    st.plotly_chart(
        fig_flipkart,
        use_container_width=True
    )

    st.info(
        """
        Après optimisation équitable, BERT et ModernBERT
        obtiennent des performances pratiquement identiques
        sur le dataset Flipkart.
        """
    )


    # ========================================================
    # GLUE / SST-2
    # ========================================================

    st.subheader(
        "Benchmark externe — GLUE / SST-2"
    )

    glue_results = pd.DataFrame({

        "Modèle": [
            "BERT",
            "ModernBERT"
        ],

        "Accuracy": [
            0.9037,
            0.9209
        ],

        "F1": [
            0.9060,
            0.9220
        ]
    })

    st.dataframe(
        glue_results,
        use_container_width=True,
        hide_index=True
    )

    fig_glue = px.bar(
        glue_results,
        x="Modèle",
        y=[
            "Accuracy",
            "F1"
        ],
        barmode="group",
        title=(
            "Performances sur GLUE / SST-2"
        ),
        range_y=[0, 1]
    )

    fig_glue.update_layout(
        yaxis_title="Score",
        xaxis_title="Modèle"
    )

    st.plotly_chart(
        fig_glue,
        use_container_width=True
    )

    st.success(
        """
        Sur SST-2, ModernBERT obtient de meilleures
        performances : +1,72 point d'Accuracy et
        +1,60 point de F1 par rapport à BERT.
        """
    )