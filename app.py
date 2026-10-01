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

st.markdown(
    """
    Ce dashboard présente une preuve de concept de classification
    automatique de produits à partir de leur description textuelle.

    **Modèle étudié :** ModernBERT  
    **Modèle de référence :** BERT  
    **Nombre de catégories :** 7
    """
)

st.divider()

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

# ============================================================
# CHARGEMENT DES RÉSULTATS D'EXPÉRIENCES
# ============================================================

@st.cache_data
def load_experiment_data():

    tuning_df = pd.read_csv(
        "data/P9_tuning_ModernBERT_extended.csv"
    )

    glue_df = pd.read_csv(
        "data/P9_GLUE_ModernBERT_all_tasks.csv"
    )

    return tuning_df, glue_df

df = load_data()

tuning_df, glue_df = load_experiment_data()

st.sidebar.success(
    f"Tuning : {len(tuning_df)} résultats | "
    f"GLUE : {len(glue_df)} résultats"
)

# ============================================================
# MODERNBERT
# ============================================================

MODEL_PATH = "Julien-Ama/ModernBERT-Flipkart-P9"

LABELS = [
    "Baby Care",
    "Beauty and Personal Care",
    "Computers",
    "Home Decor & Festive Needs",
    "Home Furnishing",
    "Kitchen & Dining",
    "Watches"
]


MODEL_PATH = "Julien-Ama/ModernBERT-Flipkart-P9"
TOKENIZER_PATH = "answerdotai/ModernBERT-base"


@st.cache_resource
def load_modernbert():

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH
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

st.sidebar.title("Dashboard P9")

st.sidebar.write(
    """
    Sélectionnez une section pour explorer
    les données ou tester le modèle.
    """
)

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Exploration des données",
        "🤖 Prédiction",
        "📈 Expérimentations ModernBERT"
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
        yaxis_title="Nombre de produits",
        font=dict(size=14)
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
        yaxis_title="Nombre de produits",
        font=dict(size=14)
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

        st.caption(
            """
            La confiance correspond à la probabilité attribuée
            par le modèle à la catégorie sélectionnée.
            Une confiance élevée ne garantit pas que la prédiction
            soit correcte.
            """
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
            },
            font=dict(size=14)
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

        st.caption(
            """
            Conseil : utilisez une description suffisamment précise
            contenant le type de produit, ses caractéristiques
            et éventuellement son usage.
            """
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

elif page == "📈 Expérimentations ModernBERT":

    st.header("Expérimentations ModernBERT")

    st.write(
        """
        Cette section permet d'explorer les résultats des
        différentes expérimentations réalisées avec ModernBERT :
        optimisation des hyperparamètres et benchmark GLUE.
        """
    )

    tab_tuning, tab_glue = st.tabs(
        [
            "⚙️ Optimisation",
            "🧪 Benchmark GLUE"
        ]
    )

    # ========================================================
    # ONGLET 1 — OPTIMISATION
    # ========================================================

    with tab_tuning:

        st.subheader(
            "Optimisation des hyperparamètres"
        )

        st.write(
            """
            Le learning rate optimal identifié lors du premier
            tuning est fixé à **5e-5**.

            Les expériences suivantes étudient l'influence du
            **batch size**, du **weight decay** et du nombre
            d'**epochs** sur les performances de ModernBERT.
            """
        )

        # ----------------------------------------------------
        # Sélection Batch size
        # ----------------------------------------------------

        batch_values = sorted(
            tuning_df["batch_size"]
            .unique()
            .tolist()
        )

        selected_batch = st.selectbox(
            "Batch size",
            batch_values
        )

        # ----------------------------------------------------
        # Sélection Weight decay
        # ----------------------------------------------------

        available_wd = (
            tuning_df[
                tuning_df["batch_size"] == selected_batch
                ]["weight_decay"]
            .unique()
            .tolist()
        )

        available_wd = sorted(available_wd)

        selected_wd = st.selectbox(
            "Weight decay",
            available_wd
        )

        # ----------------------------------------------------
        # Filtrage de la configuration
        # ----------------------------------------------------

        tuning_filtered = tuning_df[
            (
                    tuning_df["batch_size"]
                    == selected_batch
            )
            &
            (
                    tuning_df["weight_decay"]
                    == selected_wd
            )
            ].sort_values("epoch")

        # ----------------------------------------------------
        # Sélection Epoch
        # ----------------------------------------------------

        epoch_values = (
            tuning_filtered["epoch"]
            .astype(int)
            .tolist()
        )

        selected_epoch = st.select_slider(
            "Epoch",
            options=epoch_values
        )

        result_epoch = tuning_filtered[
            tuning_filtered["epoch"].astype(int)
            == selected_epoch
            ].iloc[0]

        st.markdown("### Résultats")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Accuracy",
            f"{result_epoch['val_accuracy'] * 100:.2f} %"
        )

        col2.metric(
            "Macro-F1",
            f"{result_epoch['val_f1_macro'] * 100:.2f} %"
        )

        col3.metric(
            "Precision",
            f"{result_epoch['val_precision_macro'] * 100:.2f} %"
        )

        col4.metric(
            "Recall",
            f"{result_epoch['val_recall_macro'] * 100:.2f} %"
        )

        st.metric(
            "Validation Loss",
            f"{result_epoch['val_loss']:.4f}"
        )

        # st.markdown("### Résultats")
        # 
        # col1, col2, col3, col4 = st.columns(4)
        #
        # col1.metric(
        #     "Accuracy",
        #     f"{result_epoch['val_accuracy'] * 100:.2f} %"
        # )
        #
        # col2.metric(
        #     "Macro-F1",
        #     f"{result_epoch['val_f1_macro'] * 100:.2f} %"
        # )
        #
        # col3.metric(
        #     "Precision",
        #     f"{result_epoch['val_precision_macro'] * 100:.2f} %"
        # )
        #
        # col4.metric(
        #     "Recall",
        #     f"{result_epoch['val_recall_macro'] * 100:.2f} %"
        # )
        #
        # st.metric(
        #     "Validation Loss",
        #     f"{result_epoch['val_loss']:.4f}"
        # )

        # ========================================================
        # ONGLET 2 — BENCHMARK GLUE
        # ========================================================

        with tab_glue:

            st.subheader("Benchmark GLUE — ModernBERT")

            st.write(
                """
                Cette section présente les performances de ModernBERT
                sur plusieurs tâches du benchmark GLUE.

                Sélectionnez une tâche et un epoch pour observer
                l'évolution des performances du modèle.
                """
            )

            # ----------------------------------------------------
            # Description des tâches
            # ----------------------------------------------------

            glue_descriptions = {
                "SST-2": (
                    "Analyse de sentiment : déterminer si une phrase "
                    "exprime un sentiment positif ou négatif."
                ),
                "MRPC": (
                    "Détection de paraphrases : déterminer si deux "
                    "phrases ont le même sens."
                ),
                "RTE": (
                    "Reconnaissance d'implication textuelle : déterminer "
                    "si une phrase est impliquée par une autre."
                ),
                "CoLA": (
                    "Acceptabilité linguistique : déterminer si une "
                    "phrase anglaise est linguistiquement acceptable."
                )
            }

            # ----------------------------------------------------
            # Sélection de la tâche
            # ----------------------------------------------------

            tasks = [
                "SST-2",
                "MRPC",
                "RTE",
                "CoLA"
            ]

            selected_task = st.selectbox(
                "Tâche GLUE",
                tasks
            )

            st.info(
                glue_descriptions[selected_task]
            )

            # ----------------------------------------------------
            # Filtrage des résultats
            # ----------------------------------------------------

            glue_filtered = (
                glue_df[
                    glue_df["task"] == selected_task
                    ]
                .sort_values("epoch")
                .copy()
            )

            # ----------------------------------------------------
            # Sélection de l'epoch
            # ----------------------------------------------------

            glue_epochs = (
                glue_filtered["epoch"]
                .astype(int)
                .tolist()
            )

            selected_glue_epoch = st.select_slider(
                "Epoch",
                options=glue_epochs,
                key="glue_epoch"
            )

            glue_result = glue_filtered[
                glue_filtered["epoch"].astype(int)
                == selected_glue_epoch
                ].iloc[0]

            # ----------------------------------------------------
            # Résultats de l'epoch sélectionné
            # ----------------------------------------------------

            st.markdown("### Résultats")

            if selected_task == "CoLA":

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Accuracy",
                    f"{glue_result['accuracy'] * 100:.2f} %"
                )

                col2.metric(
                    "Macro-F1",
                    f"{glue_result['f1'] * 100:.2f} %"
                )

                col3.metric(
                    "MCC",
                    f"{glue_result['mcc']:.3f}"
                )

                col4.metric(
                    "Validation Loss",
                    f"{glue_result['loss']:.4f}"
                )

            else:

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Accuracy",
                    f"{glue_result['accuracy'] * 100:.2f} %"
                )

                col2.metric(
                    "F1",
                    f"{glue_result['f1'] * 100:.2f} %"
                )

                col3.metric(
                    "Validation Loss",
                    f"{glue_result['loss']:.4f}"
                )

            # ----------------------------------------------------
            # Évolution des performances
            # ----------------------------------------------------

            st.markdown("### Évolution selon les epochs")

            # Accuracy et F1 pour toutes les tâches
            metrics_to_plot = [
                "accuracy",
                "f1"
            ]

            # MCC uniquement pour CoLA
            if selected_task == "CoLA":
                metrics_to_plot.append("mcc")

            fig_glue = px.line(
                glue_filtered,
                x="epoch",
                y=metrics_to_plot,
                markers=True,
                title=(
                    f"Évolution des performances — "
                    f"{selected_task}"
                )
            )

            fig_glue.update_layout(
                xaxis_title="Epoch",
                yaxis_title="Score",
                yaxis_range=[0, 1]
            )

            st.plotly_chart(
                fig_glue,
                use_container_width=True
            )

            # ----------------------------------------------------
            # Évolution de la Validation Loss
            # ----------------------------------------------------

            fig_loss = px.line(
                glue_filtered,
                x="epoch",
                y="loss",
                markers=True,
                title=(
                    f"Évolution de la Validation Loss — "
                    f"{selected_task}"
                )
            )

            fig_loss.update_layout(
                xaxis_title="Epoch",
                yaxis_title="Validation Loss"
            )

            st.plotly_chart(
                fig_loss,
                use_container_width=True
            )

            # ----------------------------------------------------
            # Explication de la métrique principale
            # ----------------------------------------------------

            if selected_task == "CoLA":

                st.caption(
                    """
                    Pour CoLA, la métrique de référence est le
                    Matthews Correlation Coefficient (MCC).
                    Contrairement à l'Accuracy, le MCC prend en
                    compte la qualité des prédictions sur les
                    différentes classes.
                    """
                )

            elif selected_task == "MRPC":

                st.caption(
                    """
                    MRPC évalue la détection de paraphrases.
                    Accuracy et F1 sont utilisées pour analyser
                    les performances du modèle.
                    """
                )

            elif selected_task == "RTE":

                st.caption(
                    """
                    Pour RTE, l'Accuracy constitue la métrique
                    principale utilisée pour évaluer la tâche.
                    """
                )

            else:

                st.caption(
                    """
                    SST-2 est une tâche de classification binaire
                    de sentiment. L'Accuracy constitue la métrique
                    principale.
                    """
                )

            # ====================================================
            # SYNTHÈSE DES MEILLEURS RÉSULTATS GLUE
            # ====================================================

            st.divider()

            st.subheader("Synthèse des meilleurs résultats GLUE")

            st.write(
                """
                Le tableau ci-dessous présente le meilleur epoch
                obtenu par ModernBERT pour chaque tâche GLUE,
                selon la métrique principale associée à la tâche.
                """
            )

            # ----------------------------------------------------
            # Sélection du meilleur epoch pour chaque tâche
            # ----------------------------------------------------

            best_results = []

            for task in ["SST-2", "MRPC", "RTE", "CoLA"]:

                task_df = glue_df[
                    glue_df["task"] == task
                    ].copy()

                if task == "CoLA":

                    best_row = task_df.loc[
                        task_df["mcc"].idxmax()
                    ]

                    main_metric = "MCC"
                    main_score = best_row["mcc"]

                elif task == "MRPC":

                    best_row = task_df.loc[
                        task_df["f1"].idxmax()
                    ]

                    main_metric = "F1"
                    main_score = best_row["f1"]

                else:

                    best_row = task_df.loc[
                        task_df["accuracy"].idxmax()
                    ]

                    main_metric = "Accuracy"
                    main_score = best_row["accuracy"]

                best_results.append({
                    "Tâche": task,
                    "Meilleur epoch": int(best_row["epoch"]),
                    "Métrique principale": main_metric,
                    "Score principal": main_score,
                    "Accuracy": best_row["accuracy"],
                    "F1": best_row["f1"],
                    "MCC": best_row["mcc"]
                })

            best_glue_df = pd.DataFrame(best_results)

            # ----------------------------------------------------
            # Tableau récapitulatif
            # ----------------------------------------------------

            display_best_glue = best_glue_df.copy()

            display_best_glue["Score principal"] = (
                display_best_glue["Score principal"]
                .map(lambda x: f"{x:.3f}")
            )

            display_best_glue["Accuracy"] = (
                display_best_glue["Accuracy"]
                .map(lambda x: f"{x * 100:.2f} %")
            )

            display_best_glue["F1"] = (
                display_best_glue["F1"]
                .map(lambda x: f"{x * 100:.2f} %")
            )

            display_best_glue["MCC"] = (
                display_best_glue["MCC"]
                .apply(
                    lambda x:
                    "-" if pd.isna(x)
                    else f"{x:.3f}"
                )
            )

            st.dataframe(
                display_best_glue,
                use_container_width=True,
                hide_index=True
            )

            # ----------------------------------------------------
            # Graphique des scores principaux
            # ----------------------------------------------------

            fig_best_glue = px.bar(
                best_glue_df,
                x="Tâche",
                y="Score principal",
                text="Score principal",
                title=(
                    "Meilleur score obtenu par tâche GLUE"
                ),
                hover_data=[
                    "Meilleur epoch",
                    "Métrique principale"
                ]
            )

            fig_best_glue.update_traces(
                texttemplate="%{text:.3f}",
                textposition="outside"
            )

            fig_best_glue.update_layout(
                yaxis_title="Score",
                xaxis_title="Tâche",
                yaxis_range=[0, 1]
            )

            st.plotly_chart(
                fig_best_glue,
                use_container_width=True
            )

            st.info(
                """
                Attention : les scores des différentes tâches ne sont
                pas directement comparables entre eux.

                SST-2 et RTE utilisent principalement l'Accuracy,
                MRPC utilise ici le F1, tandis que CoLA utilise le MCC.
                Ce graphique synthétise donc les meilleurs résultats
                obtenus, mais ne constitue pas un classement direct
                des tâches.
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