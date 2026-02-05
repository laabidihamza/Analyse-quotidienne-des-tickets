import io
from datetime import datetime, timedelta
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# =========================
# Constantes & configuration
# =========================

DATE_COL = "Date"
TICKET_ID_COL = "Référence du ticket"
EXCEPTION_COL = "Exception"

st.set_page_config(
    page_title="Analyse quotidienne des tickets",
    layout="wide",
)


# =========================
# Fonctions utilitaires
# =========================

def parse_date_input(date_input: datetime) -> pd.Timestamp:
    """
    Convertit un input Streamlit (datetime.date/datetime) en pd.Timestamp normalisé (sans heure).
    """
    return pd.to_datetime(date_input).normalize()


def compute_j_minus_1(j_date: pd.Timestamp) -> pd.Timestamp:
    """
    Calcule la date j-1 en tenant compte des cas particuliers imposés.

    Règles spéciales :
    - Si j = 16/12/2025 alors j-1 = 13/12/2025
    - Si j = 01/02/2026 alors j-1 = 29/01/2026
    Sinon : j-1 = j - 1 jour
    """
    # Cas particuliers (dates en format jour/mois/année)
    if j_date == pd.Timestamp("2025-12-16"):
        return pd.Timestamp("2025-12-13")
    if j_date == pd.Timestamp("2026-02-01"):
        return pd.Timestamp("2026-01-29")

    return j_date - pd.Timedelta(days=1)


def load_data_from_excel(uploaded_file) -> pd.DataFrame:
    """
    Charge les données depuis un fichier Excel uploadé dans Streamlit.
    Retourne un DataFrame pandas.
    """
    df = pd.read_excel(uploaded_file, engine="openpyxl")

    # Conversion de la colonne Date
    if DATE_COL not in df.columns:
        raise ValueError(f"La colonne obligatoire '{DATE_COL}' est manquante.")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    # Suppression des lignes où la date est invalide
    df = df.dropna(subset=[DATE_COL])

    # Tri par date
    df = df.sort_values(by=DATE_COL).reset_index(drop=True)
    return df


def validate_columns(df: pd.DataFrame) -> Optional[str]:
    """
    Vérifie la présence des colonnes essentielles.
    Retourne un message d'erreur si nécessaire, sinon None.
    """
    missing_cols = []
    for col in [DATE_COL, TICKET_ID_COL, EXCEPTION_COL]:
        if col not in df.columns:
            missing_cols.append(col)

    if missing_cols:
        return (
            "Les colonnes suivantes sont manquantes dans le fichier : "
            + ", ".join(missing_cols)
        )
    return None


# =========================
# Logique métier
# =========================

def compute_ticket_sets(
    df: pd.DataFrame, date_j1: pd.Timestamp, date_j: pd.Timestamp
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retourne deux DataFrames :
    - tickets_j1 : tickets présents à la date j-1
    - tickets_j  : tickets présents à la date j
    """
    tickets_j1 = df[df[DATE_COL] == date_j1].copy()
    tickets_j = df[df[DATE_COL] == date_j].copy()
    return tickets_j1, tickets_j


def compute_synthesis(
    tickets_j1: pd.DataFrame, tickets_j: pd.DataFrame, date_j1: pd.Timestamp, date_j: pd.Timestamp
) -> pd.DataFrame:
    """
    Calcule la synthèse pour la date j.

    Colonnes :
    - Date
    - Nombre des cas traités à la date j : présents à j-1 et absents à j
    - Nombre des nouveaux cas à la date j : présents à j et absents à j-1
    - Nombre des tickets à la date j
    """
    # Identifiants à j-1 et j
    set_j1 = set(tickets_j1[TICKET_ID_COL].astype(str))
    set_j = set(tickets_j[TICKET_ID_COL].astype(str))

    # Cas traités : présents à j-1 mais absents à j
    treated_ids = set_j1 - set_j
    # Nouveaux cas : présents à j mais absents à j-1
    new_ids = set_j - set_j1

    synthese_data = {
        "Date": [date_j.normalize()],
        "Nombre des cas traités à la date j": [len(treated_ids)],
        "Nombre des nouveaux cas à la date j": [len(new_ids)],
        "Nombre des tickets à la date j": [len(set_j)],
    }
    synthese_df = pd.DataFrame(synthese_data)
    return synthese_df


def compute_synthesis_all_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule la synthèse pour **toutes** les dates présentes dans le fichier d'entrée.

    Pour chaque date d :
    - Nombre des cas traités à la date d : tickets présents à d-1 et absents à d
    - Nombre des nouveaux cas à la date d : tickets présents à d et absents à d-1
    - Nombre des tickets à la date d

    La logique j-1 utilise compute_j_minus_1, ce qui applique aussi les cas particuliers métier.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "Date",
                "Nombre des cas traités à la date j",
                "Nombre des nouveaux cas à la date j",
                "Nombre des tickets à la date j",
            ]
        )

    data = df.copy()
    # Normalisation de la date pour travailler jour par jour
    data[DATE_COL] = pd.to_datetime(data[DATE_COL], errors="coerce").dt.normalize()
    data = data.dropna(subset=[DATE_COL])
    data[TICKET_ID_COL] = data[TICKET_ID_COL].astype(str)

    # Mapping date -> ensemble d'identifiants
    groups = (
        data.groupby(DATE_COL)[TICKET_ID_COL]
        .apply(lambda s: set(s.astype(str)))
        .to_dict()
    )

    all_dates = sorted(groups.keys())

    rows = []
    for date_j in all_dates:
        set_j = groups.get(date_j, set())

        # Calcul de j-1 selon les règles métier
        date_j1 = compute_j_minus_1(date_j)
        # On normalise pour être sûr d'utiliser la même clé que dans groups
        date_j1_norm = pd.to_datetime(date_j1).normalize()
        set_j1 = groups.get(date_j1_norm, set())

        treated_ids = set_j1 - set_j
        new_ids = set_j - set_j1

        rows.append(
            {
                "Date": date_j,
                "Nombre des cas traités à la date j": len(treated_ids),
                "Nombre des nouveaux cas à la date j": len(new_ids),
                "Nombre des tickets à la date j": len(set_j),
            }
        )

    synthese_all_df = pd.DataFrame(rows)
    # Tri par date pour garantir l'ordre
    synthese_all_df = synthese_all_df.sort_values(by="Date").reset_index(drop=True)
    return synthese_all_df


def compute_new_tickets(
    tickets_j1: pd.DataFrame, tickets_j: pd.DataFrame
) -> pd.DataFrame:
    """
    Retourne les nouveaux tickets (présents à j, absents à j-1),
    triés par fréquence de l'exception (décroissante) puis par date.
    """
    set_j1 = set(tickets_j1[TICKET_ID_COL].astype(str))
    tickets_j = tickets_j.copy()
    tickets_j[TICKET_ID_COL] = tickets_j[TICKET_ID_COL].astype(str)

    new_mask = ~tickets_j[TICKET_ID_COL].isin(set_j1)
    new_tickets = tickets_j[new_mask].copy()

    if new_tickets.empty:
        return new_tickets

    # Compte des occurrences par exception
    if EXCEPTION_COL in new_tickets.columns:
        exception_counts = (
            new_tickets[EXCEPTION_COL]
            .value_counts()
            .rename("Exception_Count")
            .to_frame()
        )
        new_tickets = new_tickets.merge(
            exception_counts,
            left_on=EXCEPTION_COL,
            right_index=True,
            how="left",
        )
        new_tickets = new_tickets.sort_values(
            by=["Exception_Count", DATE_COL], ascending=[False, True]
        )
        new_tickets = new_tickets.drop(columns=["Exception_Count"])
    else:
        new_tickets = new_tickets.sort_values(by=DATE_COL)

    return new_tickets.reset_index(drop=True)


def compute_treated_tickets(
    tickets_j1: pd.DataFrame, tickets_j: pd.DataFrame
) -> pd.DataFrame:
    """
    Retourne les tickets traités (présents à j-1, absents à j).
    """
    tickets_j1 = tickets_j1.copy()
    tickets_j = tickets_j.copy()

    tickets_j1[TICKET_ID_COL] = tickets_j1[TICKET_ID_COL].astype(str)
    tickets_j[TICKET_ID_COL] = tickets_j[TICKET_ID_COL].astype(str)

    set_j = set(tickets_j[TICKET_ID_COL])
    treated_mask = ~tickets_j1[TICKET_ID_COL].isin(set_j)
    treated_tickets = tickets_j1[treated_mask].copy()

    return treated_tickets.reset_index(drop=True)


def generate_excel_bytes(
    synthese_df: pd.DataFrame,
    nouveaux_df: pd.DataFrame,
    traites_df: pd.DataFrame,
) -> bytes:
    """
    Génère un fichier Excel en mémoire (bytes) avec 3 feuilles :
    1) Synthèse
    2) Nouveaux tickets
    3) Tickets traités
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        synthese_df.to_excel(writer, sheet_name="Synthèse", index=False)
        nouveaux_df.to_excel(writer, sheet_name="Nouveaux tickets", index=False)
        traites_df.to_excel(writer, sheet_name="Tickets traités", index=False)

    output.seek(0)
    return output.getvalue()


def generate_single_sheet_excel_bytes(df: pd.DataFrame, sheet_name: str) -> bytes:
    """
    Génère un fichier Excel en mémoire (bytes) contenant une seule feuille.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)
    return output.getvalue()


def generate_single_sheet_excel(df: pd.DataFrame, sheet_name: str = "Données") -> bytes:
    """
    Génère un fichier Excel en mémoire (bytes) avec une seule feuille.
    Utilisé notamment pour l'export des statistiques d'exceptions.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)
    return output.getvalue()


# =========================
# Dashboard / Visualisations
# =========================

def build_dashboard(df: pd.DataFrame):
    """
    Construit les graphiques principaux pour le dashboard :
    - Évolution du nombre total de tickets par date
    - Nombre de tickets par date (simplicité pour "nouveaux vs traités")
    - Répartition des exceptions
    """
    if df.empty:
        st.info("Aucune donnée disponible pour le dashboard.")
        return

    # Nombre de tickets (références uniques) par date
    df_daily = (
        df.groupby(DATE_COL)[TICKET_ID_COL]
        .nunique()
        .reset_index()
        .rename(columns={TICKET_ID_COL: "Nombre de tickets"})
    )

    st.subheader("Évolution du nombre total de tickets par date")
    fig_daily = px.line(
        df_daily,
        x=DATE_COL,
        y="Nombre de tickets",
        markers=True,
    )
    st.plotly_chart(fig_daily, use_container_width=True)

    # Histogramme nombre de tickets par date
    st.subheader("Nombre de tickets par date")
    fig_bar = px.bar(
        df_daily,
        x=DATE_COL,
        y="Nombre de tickets",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Graphique du nombre de tickets résolus vs non résolus
    st.subheader("Nombre de tickets résolus / non résolus")
    # Un ticket est résolu si sa référence n'apparaît qu'une seule fois
    # Comptage du nombre d'occurrences par référence de ticket
    counts_by_ref = (
        df[TICKET_ID_COL]
        .astype(str)
        .value_counts()
        .reset_index()
    )
    # Colonnes : [Référence du ticket, count]
    counts_by_ref.columns = [TICKET_ID_COL, "count"]

    # Statut : résolu si la référence apparaît une seule fois, sinon non résolu
    counts_by_ref["Statut"] = "Non résolu (référence dupliquée)"
    counts_by_ref.loc[counts_by_ref["count"] == 1, "Statut"] = (
        "Résolu (référence unique)"
    )
    resolved_stats = (
        counts_by_ref.groupby("Statut")["count"]
        .sum()
        .reset_index()
        .rename(columns={"count": "Nombre de tickets"})
    )
    if not resolved_stats.empty:
        fig_resolved = px.bar(
            resolved_stats,
            x="Statut",
            y="Nombre de tickets",
            text="Nombre de tickets",
        )
        fig_resolved.update_traces(textposition="outside")
        st.plotly_chart(fig_resolved, use_container_width=True)
    else:
        st.info("Impossible de calculer les tickets résolus / non résolus.")

    # Pie chart : top 10 exceptions les plus fréquentes
    st.subheader("Top 10 des exceptions (camembert)")
    if EXCEPTION_COL in df.columns:


        # 🔹 Copy dataset and drop duplicate ticket references
        df_pie = df.copy()
        df_pie[TICKET_ID_COL] = df_pie[TICKET_ID_COL].astype(str)
        df_pie = df_pie.drop_duplicates(subset=[TICKET_ID_COL], keep="last")



        exception_counts = (
            df_pie[EXCEPTION_COL]
            .value_counts()
            .reset_index(name="Nombre")
            .rename(columns={"index": EXCEPTION_COL})
        )
        if exception_counts.empty:
            st.info("Aucune exception à afficher.")
        else:
            top_n = 10
            top_exceptions = exception_counts.head(top_n).copy()

            # Raccourcir le texte de la légende pour plus de lisibilité
            max_len = 60
            top_exceptions["Exception_courte"] = (
                top_exceptions[EXCEPTION_COL]
                .astype(str)
                .str.slice(0, max_len)
            )
            # Ajouter "..." seulement si le texte a été tronqué
            mask_tronque = top_exceptions[EXCEPTION_COL].str.len() > max_len
            top_exceptions.loc[mask_tronque, "Exception_courte"] = (
                top_exceptions.loc[mask_tronque, "Exception_courte"] + "..."
            )

            fig_exceptions = px.pie(
                top_exceptions,
                names="Exception_courte",
                values="Nombre",
            )
            # Agrandir le camembert dans l'espace disponible
            fig_exceptions.update_layout(
                margin=dict(l=0, r=0, t=40, b=0),
                legend=dict(
                    font=dict(size=10),
                ),
            )
            st.plotly_chart(fig_exceptions, use_container_width=True)
    else:
        st.info(f"La colonne '{EXCEPTION_COL}' n'existe pas dans les données.")


# =========================
# Interface Streamlit
# =========================

def main():
    st.title("Analyse quotidienne des tickets")
    st.markdown(
        """
Cette application permet d'analyser quotidiennement les tickets à partir d'un fichier Excel.
        """
    )

    tab_analyse, tab_dashboard, tab_exceptions = st.tabs(
        ["Analyse & Résultats", "Dashboard", "Exceptions"]
    )

    # Stockage dans session_state pour réutiliser les données dans le dashboard
    if "df_source" not in st.session_state:
        st.session_state.df_source = None
    if "synthese_df" not in st.session_state:
        st.session_state.synthese_df = None
    if "nouveaux_df" not in st.session_state:
        st.session_state.nouveaux_df = None
    if "traites_df" not in st.session_state:
        st.session_state.traites_df = None
    if "excel_bytes" not in st.session_state:
        st.session_state.excel_bytes = None

    with tab_analyse:
        st.header("Analyse & Résultats")

        uploaded_file = st.file_uploader(
            "Uploader un fichier Excel", type=["xlsx", "xls"]
        )

        # Paramètres de dates
        today = datetime.today().date()
        default_j = today
        default_j1 = (today - timedelta(days=1))

        col_date1, col_date2 = st.columns(2)
        with col_date1:
            date_input_1 = st.date_input(
                "Date 1 (j-1)", value=default_j1, key="date_j1"
            )
        with col_date2:
            date_input_2 = st.date_input(
                "Date 2 (j)", value=default_j, key="date_j"
            )

        run_analysis = st.button("Lancer l’analyse")

        if uploaded_file is not None:
            st.subheader("Aperçu du fichier Excel")
            try:
                df = load_data_from_excel(uploaded_file)
                if df.empty:
                    st.error("Le fichier ne contient aucune donnée après nettoyage.")
                else:
                    st.dataframe(df.head(50))
                    # Stocker la source dès le chargement (utile pour Dashboard / Exceptions)
                    st.session_state.df_source = df
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier : {e}")
                df = None
        else:
            df = None

        if run_analysis:
            # Validation des dates
            try:
                date_j = parse_date_input(date_input_2)
                # Respecter explicitement la règle j-1 si l'utilisateur ne la suit pas
                computed_j1 = compute_j_minus_1(date_j)
                date_j1 = parse_date_input(date_input_1)
            except Exception:
                st.error("Les dates fournies sont invalides.")
                return

            if df is None:
                st.error("Aucun fichier valide n'a été chargé.")
                return

            # Vérification colonnes
            error_cols = validate_columns(df)
            if error_cols:
                st.error(error_cols)
                return

            # Alerte si la Date 1 saisie diffère de la règle j-1 calculée
            if date_j1 != computed_j1:
                st.warning(
                    f"Attention : selon les règles métiers, pour j = {date_j.date()}, "
                    f"la date j-1 attendue est {computed_j1.date()}. "
                    f"Vous avez saisi j-1 = {date_j1.date()}."
                )

            # Calcul des ensembles
            tickets_j1, tickets_j = compute_ticket_sets(df, date_j1, date_j)

            if tickets_j.empty and tickets_j1.empty:
                st.error(
                    "Aucun ticket trouvé pour les dates sélectionnées. "
                    "Vérifiez que les dates existent dans la colonne 'Date'."
                )
                return

            # Synthèse pour la date j saisie par l'utilisateur
            synthese_j_df = compute_synthesis(tickets_j1, tickets_j, date_j1, date_j)
            # Synthèse pour l'ensemble des dates présentes dans le fichier
            synthese_all_df = compute_synthesis_all_dates(df)
            nouveaux_df = compute_new_tickets(tickets_j1, tickets_j)
            traites_df = compute_treated_tickets(tickets_j1, tickets_j)

            st.session_state.df_source = df
            # On stocke la synthèse complète (toutes dates) pour d'éventuels usages futurs
            st.session_state.synthese_df = synthese_all_df
            st.session_state.nouveaux_df = nouveaux_df
            st.session_state.traites_df = traites_df

            # Synthèse sous forme de métriques (à partir de la ligne unique de synthese_df)
            st.subheader("Synthèse")
            if not synthese_j_df.empty:
                synth_row = synthese_j_df.iloc[0]
                # Toujours convertir en chaîne de caractères pour éviter les erreurs de type dans st.metric
                date_value = synth_row["Date"]
                try:
                    # pandas.Timestamp, datetime, date...
                    date_j_str = date_value.strftime("%Y-%m-%d")
                except Exception:
                    date_j_str = str(date_value)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Date j", date_j_str)
                with col2:
                    st.metric(
                        "Cas traités à la date j",
                        int(
                            synth_row[
                                "Nombre des cas traités à la date j"
                            ]
                        ),
                    )
                with col3:
                    st.metric(
                        "Nouveaux cas à la date j",
                        int(
                            synth_row[
                                "Nombre des nouveaux cas à la date j"
                            ]
                        ),
                    )
                with col4:
                    st.metric(
                        "Nombre total de tickets à la date j",
                        int(
                            synth_row[
                                "Nombre des tickets à la date j"
                            ]
                        ),
                    )

            st.subheader("Nouveaux tickets (présents à j, absents à j-1)")
            if nouveaux_df.empty:
                st.info("Aucun nouveau ticket pour la période sélectionnée.")
            else:
                st.dataframe(nouveaux_df)

            st.subheader("Tickets traités (présents à j-1, absents à j)")
            if traites_df.empty:
                st.info("Aucun ticket traité pour la période sélectionnée.")
            else:
                st.dataframe(traites_df)

            # Génération du fichier Excel en mémoire
            # La feuille 1 'Synthèse' doit contenir les statistiques pour toutes les dates
            excel_bytes = generate_excel_bytes(synthese_all_df, nouveaux_df, traites_df)
            st.session_state.excel_bytes = excel_bytes

            st.download_button(
                label="Télécharger le fichier Excel de résultats",
                data=excel_bytes,
                file_name=f"analyse_tickets_{date_j.date()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    with tab_dashboard:
        st.header("Dashboard")
        if st.session_state.df_source is None:
            st.info(
                "Veuillez d'abord charger un fichier et lancer une analyse dans l'onglet "
                "'Analyse & Résultats'."
            )
        else:
            build_dashboard(st.session_state.df_source)

    # =========================
    # Onglet Exceptions
    # =========================
    with tab_exceptions:
        st.header("Exceptions")
        if st.session_state.df_source is None:
            st.info(
                "Veuillez d'abord charger un fichier et lancer une analyse dans l'onglet "
                "'Analyse & Résultats' pour initialiser les données."
            )
        else:
            # 1) Copie du fichier d'entrée
            df_exc = st.session_state.df_source.copy()

            # Vérification des colonnes nécessaires
            missing_cols = []
            for col in [DATE_COL, TICKET_ID_COL, EXCEPTION_COL]:
                if col not in df_exc.columns:
                    missing_cols.append(col)
            if missing_cols:
                st.error(
                    "Les colonnes suivantes sont manquantes pour l'analyse des exceptions : "
                    + ", ".join(missing_cols)
                )
                return

            # Normalisation minimale
            df_exc[DATE_COL] = pd.to_datetime(df_exc[DATE_COL], errors="coerce")
            df_exc = df_exc.dropna(subset=[DATE_COL]).copy()
            df_exc[TICKET_ID_COL] = df_exc[TICKET_ID_COL].astype(str)

            # 2) Metrics : nombre de références distinctes et nombre de lignes doublons supprimées
            ref_counts = df_exc[TICKET_ID_COL].value_counts(dropna=False)
            nb_refs_distinctes = int(ref_counts.size)
            # Lignes qui seront supprimées lors du dédoublonnage (on garde la dernière occurrence)
            doublon_mask = df_exc.duplicated(subset=[TICKET_ID_COL], keep="last")
            nb_lignes_doublons_supprimees = int(doublon_mask.sum())

            m1, m2 = st.columns(2)
            with m1:
                st.metric("Nombre de références distinctes", nb_refs_distinctes)
            with m2:
                st.metric("Lignes doublons supprimées", nb_lignes_doublons_supprimees)

            # 3) Éliminer les doublons par référence du ticket (on garde la dernière occurrence par date)
            df_exc = (
                df_exc.sort_values(by=[DATE_COL])
                .drop_duplicates(subset=[TICKET_ID_COL], keep="last")
                .reset_index(drop=True)
            )

            st.markdown(
                "Sélectionnez une période pour analyser les exceptions distinctes "
                "et leur nombre total d'occurrences (après dédoublonnage par référence)."
            )

            # bornes pour la période après dédoublonnage
            min_date = df_exc[DATE_COL].min()
            max_date = df_exc[DATE_COL].max()

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                start_date = st.date_input(
                    "Date de début",
                    value=min_date.date() if hasattr(min_date, "date") else None,
                    min_value=min_date.date() if hasattr(min_date, "date") else None,
                    max_value=max_date.date() if hasattr(max_date, "date") else None,
                    key="exc_start_date",
                )
            with col_d2:
                end_date = st.date_input(
                    "Date de fin",
                    value=max_date.date() if hasattr(max_date, "date") else None,
                    min_value=min_date.date() if hasattr(min_date, "date") else None,
                    max_value=max_date.date() if hasattr(max_date, "date") else None,
                    key="exc_end_date",
                )

            # Conversion en Timestamp pour filtrage
            try:
                start_ts = pd.to_datetime(start_date).normalize()
                end_ts = pd.to_datetime(end_date).normalize()
            except Exception:
                st.error("Les dates sélectionnées pour l'analyse des exceptions sont invalides.")
                return

            if start_ts > end_ts:
                st.error("La date de début doit être inférieure ou égale à la date de fin.")
            else:
                # Filtre sur la période
                mask_period = (df_exc[DATE_COL] >= start_ts) & (df_exc[DATE_COL] <= end_ts)
                df_period = df_exc[mask_period].copy()

                if df_period.empty:
                    st.info(
                        "Aucune donnée d'exception pour la période sélectionnée. "
                        "Veuillez choisir une autre plage de dates."
                    )
                else:
                    # Tableau des exceptions distinctes et de leur nombre total
                    exceptions_stats = (
                        df_period[EXCEPTION_COL]
                        .value_counts()
                        .reset_index(name="Nombre d'occurrences")
                        .rename(columns={"index": EXCEPTION_COL})
                    )

                    st.subheader("Exceptions distinctes sur la période sélectionnée")
                    st.dataframe(exceptions_stats)

                    # Sélection d'une exception pour visualiser son évolution quotidienne
                    st.subheader(
                        "Évolution quotidienne d'une exception sur la période sélectionnée"
                    )
                    exceptions_list = (
                        exceptions_stats[EXCEPTION_COL]
                        .dropna()
                        .astype(str)
                        .tolist()
                    )
                    if exceptions_list:
                        selected_exception = st.selectbox(
                            "Choisissez une exception",
                            options=exceptions_list,
                            key="exc_selected_exception",
                        )

                        df_exc_selected = df_period[
                            df_period[EXCEPTION_COL] == selected_exception
                        ].copy()
                        daily_counts = (
                            df_exc_selected.groupby(DATE_COL)[EXCEPTION_COL]
                            .count()
                            .reset_index(name="Nombre d'occurrences")
                        )

                        if daily_counts.empty:
                            st.info(
                                "Aucune occurrence de cette exception sur la période sélectionnée."
                            )
                        else:
                            fig_exc_daily = px.line(
                                daily_counts,
                                x=DATE_COL,
                                y="Nombre d'occurrences",
                                markers=True,
                            )
                            st.plotly_chart(
                                fig_exc_daily, use_container_width=True
                            )
                    else:
                        st.info(
                            "Aucune exception distincte à afficher pour la période sélectionnée."
                        )

                    # Bouton de téléchargement Excel
                    excel_exc_bytes = generate_single_sheet_excel(
                        exceptions_stats,
                        sheet_name="Exceptions",
                    )
                    st.download_button(
                        label="Télécharger les exceptions (Excel)",
                        data=excel_exc_bytes,
                        file_name=(
                            f"exceptions_{start_ts.date()}_{end_ts.date()}.xlsx"
                        ),
                        mime=(
                            "application/vnd.openxmlformats-officedocument."
                            "spreadsheetml.sheet"
                        ),
                    )


if __name__ == "__main__":
    main()

