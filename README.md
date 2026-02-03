## Application d'analyse quotidienne des tickets

Cette application Streamlit permet d'automatiser une analyse quotidienne de tickets à partir d'un fichier Excel, en appliquant des règles déterministes (sans IA) avec **pandas**.

### 1. Fonctionnalités principales

- **Upload d'un fichier Excel** (une feuille principale).
- Sélection de **deux dates** : `Date 1 (j-1)` et `Date 2 (j)`.
- Application des règles métiers pour :
  - Identifier les **nouveaux tickets** (présents à j, absents à j-1).
  - Identifier les **tickets traités** (présents à j-1, absents à j).
  - Calculer la **synthèse** du jour j :
    - Nombre de cas traités à la date j.
    - Nombre de nouveaux cas à la date j.
    - Nombre total de tickets à la date j.
- Génération d'un **fichier Excel de sortie** avec 3 feuilles :
  1. `Synthèse`
  2. `Nouveaux tickets`
  3. `Tickets traités`
- **Dashboard** interactif avec :
  - Évolution du nombre total de tickets par date.
  - Nombre de tickets par date.
  - Répartition des exceptions (camembert).

### 2. Pré-requis

- Python 3.9+ (recommandé)
- pip (gestionnaire de paquets Python)

### 3. Installation

Depuis le dossier du projet (`Daily` dans ton cas) :

```bash
pip install -r requirements.txt
```

Si tu ne souhaites pas utiliser `requirements.txt`, installe manuellement :

```bash
pip install streamlit pandas openpyxl plotly
```

### 4. Lancer l'application

Depuis le dossier contenant `app.py` :

```bash
streamlit run app.py
```

Un onglet s'ouvrira dans ton navigateur par défaut (ou suis l'URL affichée dans le terminal).

### 5. Format du fichier Excel d'entrée

Le fichier doit contenir au minimum les colonnes suivantes (noms **exactement** identiques) :

- `Date` : date d’enregistrement du ticket (format Excel habituel ou texte convertible en date).
- `Référence du ticket` : identifiant unique du ticket.
- `Exception` : type d’exception associé au ticket.

Les autres colonnes sont conservées et restituées dans les feuilles `Nouveaux tickets` et `Tickets traités`.

### 6. Règles métiers implémentées

1. **Préparation des données**
   - Chargement avec `pandas.read_excel` + moteur `openpyxl`.
   - Conversion de la colonne `Date` en type datetime (les valeurs invalides sont supprimées).
   - Tri des données par `Date`.

2. **Définition des ensembles**
   - `tickets_j`  = tickets dont la colonne `Date` est égale à `Date 2 (j)`.
   - `tickets_j1` = tickets dont la colonne `Date` est égale à `Date 1 (j-1)`.

3. **Règles de calcul de la synthèse**
   - **Cas traités à la date j** : tickets présents dans `tickets_j1` et absents de `tickets_j`.
   - **Nouveaux cas à la date j** : tickets présents dans `tickets_j` et absents de `tickets_j1`.
   - **Nombre de tickets à la date j** : nombre d'identifiants uniques dans `tickets_j`.

4. **Cas particuliers pour j-1**

Ces règles sont codées de manière **explicite** dans la fonction `compute_j_minus_1` :

- Si `j = 16/12/2025` alors `j-1 = 13/12/2025`.
- Si `j = 01/02/2026` alors `j-1 = 29/01/2026`.
- Sinon : `j-1 = j - 1 jour`.

Une alerte est affichée si la date j-1 saisie par l'utilisateur ne correspond pas à la date calculée par ces règles.

5. **Feuilles Excel de sortie**

- **Synthèse**
  - Colonnes : `Date`, `Nombre des cas traités à la date j`, `Nombre des nouveaux cas à la date j`, `Nombre des tickets à la date j`.

- **Nouveaux tickets**
  - Tickets présents à j et absents à j-1.
  - Toutes les colonnes d'origine sont conservées.
  - Triés par **nombre d'occurrences de l'exception** (colonne `Exception`, ordre décroissant), puis par date.

- **Tickets traités**
  - Tickets présents à j-1 et absents à j.
  - Toutes les colonnes d'origine sont conservées.

### 7. Dashboard

L'onglet **Dashboard** affiche des graphiques Plotly à partir des données chargées :

- **Évolution du nombre total de tickets par date** : courbe du nombre de tickets (uniques par `Référence du ticket`) par jour.
- **Nombre de tickets par date** : histogramme du volume de tickets par jour.
- **Répartition des exceptions** : graphique en secteurs (`pie chart`) de la colonne `Exception`.

Les graphiques sont interactifs (zoom, survol, export d'image, etc.).

### 8. Messages d'erreur et validations

L'application gère plusieurs cas :

- Fichier non chargé ou invalide.
- Fichier vide après nettoyage.
- Colonnes obligatoires manquantes (`Date`, `Référence du ticket`, `Exception`).
- Dates fournies non présentes dans les données.
- Alerte si la date j-1 saisie ne correspond pas à la règle métier.

Les messages sont affichés directement dans l'interface Streamlit.

### 9. Dépendances principales

- `streamlit` : interface web.
- `pandas` : manipulation des données.
- `openpyxl` : lecture/écriture des fichiers Excel.
- `plotly` : graphiques interactifs.

