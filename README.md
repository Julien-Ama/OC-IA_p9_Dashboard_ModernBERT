# P9 — Classification automatique de produits avec BERT et ModernBERT

Projet réalisé dans le cadre de la formation **Ingénieur IA — OpenClassrooms**.

L'objectif du projet est d'étudier et de comparer des modèles de classification
de texte basés sur les Large Language Models (LLM), puis de développer une
preuve de concept permettant de classifier automatiquement des produits à partir
de leur description textuelle.

Le projet utilise le dataset e-commerce **Flipkart** contenant 1 050 produits
répartis dans 7 catégories.

---

## Objectifs du projet

Les principales étapes du projet sont :

- exploration et préparation des données Flipkart ;
- création d'embeddings textuels ;
- expérimentation avec BERT et ModernBERT ;
- classification supervisée des produits ;
- optimisation des hyperparamètres ;
- comparaison des performances de BERT et ModernBERT ;
- évaluation externe avec le benchmark GLUE / SST-2 ;
- création d'un dashboard interactif avec Streamlit ;
- déploiement du modèle et de l'application dans le cloud.

---

## Dataset

Le dataset utilisé est :

`flipkart_com-ecommerce_sample_1050.csv`

Il contient **1 050 produits**, répartis équitablement dans **7 catégories** :

- Baby Care
- Beauty and Personal Care
- Computers
- Home Decor & Festive Needs
- Home Furnishing
- Kitchen & Dining
- Watches

La description textuelle des produits est utilisée comme entrée du modèle de
classification.

---

## Modèles étudiés

Deux architectures principales ont été comparées :

### BERT

BERT constitue le modèle de référence du projet.

Le modèle a été fine-tuné sur les descriptions des produits Flipkart afin de
réaliser une classification multiclasse parmi les 7 catégories.

### ModernBERT

ModernBERT est une architecture plus récente basée sur Transformer.

Le modèle utilisé pour le projet est basé sur :

`answerdotai/ModernBERT-base`

Il a ensuite été fine-tuné sur le dataset Flipkart.

---

## Résultats sur Flipkart

Après optimisation des deux modèles :

| Modèle | Accuracy | Macro-F1 |
|---|---:|---:|
| BERT optimisé | 0.9367 | 0.9365 |
| ModernBERT optimisé | 0.9367 | 0.9361 |

Sur le dataset métier Flipkart, les deux modèles obtiennent donc des
performances très proches.

---

## Benchmark GLUE — SST-2

BERT et ModernBERT ont également été comparés sur la tâche **SST-2** du
benchmark GLUE.

| Modèle | Accuracy | F1 |
|---|---:|---:|
| BERT | 0.9037 | 0.9060 |
| ModernBERT | 0.9209 | 0.9220 |

Sur SST-2, ModernBERT obtient de meilleures performances que BERT :

- **+1,72 point d'Accuracy**
- **+1,60 point de F1**

Ces résultats permettent de compléter la comparaison réalisée sur le dataset
métier avec un benchmark externe standardisé.

---

## Dashboard Streamlit

Une application interactive a été développée avec **Streamlit** afin de
présenter la preuve de concept.

Elle contient trois sections principales.

### Exploration des données

Cette partie permet notamment de visualiser :

- la répartition des produits par catégorie ;
- la longueur des descriptions ;
- les mots les plus fréquents ;
- des nuages de mots.

### Prédiction

L'utilisateur peut saisir la description d'un produit.

Le texte est ensuite traité par le tokenizer ModernBERT et envoyé au modèle
fine-tuné.

Le dashboard affiche :

- la catégorie prédite ;
- le niveau de confiance ;
- les probabilités associées aux différentes catégories.

Il est également possible de sélectionner directement un produit du dataset
Flipkart et de comparer la prédiction avec sa catégorie réelle.

### Comparaison BERT / ModernBERT

Cette section présente les performances obtenues par les deux modèles :

- sur Flipkart ;
- sur GLUE / SST-2.

---

## Architecture du POC

```text
Utilisateur
    |
    v
Dashboard Streamlit
    |
    v
Tokenizer ModernBERT
    |
    v
ModernBERT fine-tuné
    |
    v
Probabilités des 7 catégories
    |
    v
Catégorie prédite
```

L'architecture de déploiement est la suivante :

```text
GitHub
   |
   v
Streamlit Community Cloud
   |
   +------> Dataset Flipkart
   |
   v
Hugging Face Hub
   |
   v
ModernBERT fine-tuné
   |
   v
Prédiction
```

---

## Hébergement du modèle

Le modèle ModernBERT fine-tuné est hébergé sur Hugging Face :

`Julien-Ama/ModernBERT-Flipkart-P9`

Le tokenizer utilisé est celui du modèle original :

`answerdotai/ModernBERT-base`

Cette séparation permet de ne pas stocker les poids du modèle directement dans
le dépôt GitHub.

---

## Installation locale

Cloner le dépôt :

```bash
git clone <URL_DU_DEPOT>
```

Se placer dans le projet :

```bash
cd OC-IA_p9_Dashboard_ModernBERT
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Puis lancer l'application :

```bash
streamlit run app.py
```

L'application est alors accessible localement depuis le navigateur.

---

## Technologies utilisées

- Python
- pandas
- PyTorch
- Hugging Face Transformers
- BERT
- ModernBERT
- scikit-learn
- Streamlit
- Plotly
- Matplotlib
- WordCloud
- Hugging Face Hub
- Git / GitHub

---

## Accessibilité

Le dashboard a été conçu en prenant en compte plusieurs principes
d'accessibilité :

- navigation simple et explicite ;
- libellés textuels pour les éléments interactifs ;
- titres et axes explicites sur les graphiques ;
- taille de texte lisible ;
- messages de succès et d'erreur accompagnés d'un texte explicite ;
- information ne reposant pas uniquement sur la couleur.

L'objectif est de faciliter la compréhension et l'utilisation du dashboard
tout en respectant les bonnes pratiques d'accessibilité web.

---

## Structure du projet

```text
p9_dashboard/
|
|-- app.py
|-- README.md
|-- requirements.txt
|-- .gitignore
|
|-- data/
|   `-- flipkart_com-ecommerce_sample_1050.csv
|
`-- models/
    `-- P9_ModernBERT_best/
```

Le dossier `models/` est exclu du dépôt GitHub grâce au `.gitignore`.
Le modèle utilisé par l'application déployée est récupéré depuis Hugging Face.

---

## Limites du POC

Même avec une Accuracy supérieure à 93 % sur le jeu de test Flipkart, le modèle
peut produire des erreurs.

Les performances sont notamment dépendantes :

- de la qualité de la description du produit ;
- de sa proximité avec les données utilisées pour l'entraînement ;
- de la présence d'informations suffisamment discriminantes ;
- des 7 catégories disponibles dans le dataset.

Le niveau de confiance affiché correspond à la probabilité calculée par le
modèle et ne garantit pas que la prédiction soit correcte.

---

## Conclusion

BERT et ModernBERT obtiennent des performances très proches sur la tâche métier
Flipkart après optimisation.

ModernBERT montre cependant de meilleures performances sur le benchmark externe
GLUE / SST-2.

Le dashboard Streamlit permet de transformer cette expérimentation en une
preuve de concept interactive et déployée, capable de classifier de nouvelles
descriptions de produits avec le modèle ModernBERT fine-tuné.
