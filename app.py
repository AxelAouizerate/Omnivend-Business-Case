"""
Omnivend - Dashboard Business Case
Analyse des ventes brutes, marge et performance par magasin.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="Omnivend - Dashboard Ventes",
    page_icon="🛒",
    layout="wide",
)

# CSS : réduction des paddings et espaces verticaux de Streamlit par défaut
st.markdown(
    """
    <style>
    /* Réduction du padding global de la page */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    /* Séparateurs plus compacts */
    hr {
        margin: 0.3rem 0 !important;
    }
    /* Titres compacts mais lisibles */
    h1 { padding: 0.4rem 0 0.2rem 0 !important; margin: 0 0 0.4rem 0 !important; line-height: 1.3 !important; font-size: 1.95rem !important; }
    h2 { padding: 0.2rem 0 0.1rem 0 !important; margin: 0.3rem 0 0.2rem 0 !important; line-height: 1.3 !important; font-size: 1.5rem !important; }
    h3 { padding: 0.2rem 0 0.1rem 0 !important; margin: 0.3rem 0 0.2rem 0 !important; line-height: 1.3 !important; font-size: 1.35rem !important; }
    h4, h5 { padding: 0.15rem 0 0.05rem 0 !important; margin: 0.2rem 0 0.15rem 0 !important; line-height: 1.3 !important; font-size: 1.15rem !important; }
    /* Tableaux markdown plus compacts */
    .stMarkdown table { font-size: 0.85rem !important; }
    .stMarkdown table th, .stMarkdown table td {
        padding: 4px 8px !important;
        line-height: 1.25 !important;
    }
    /* KPI metrics plus compacts */
    [data-testid="stMetric"] {
        padding: 2px 4px;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
        line-height: 1.3 !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.75rem !important;
    }
    /* Captions plus rapprochées */
    [data-testid="stCaptionContainer"] {
        margin-top: -4px;
    }
    /* Espacement vertical des éléments réduit */
    [data-testid="stVerticalBlock"] > [data-testid="element-container"] {
        margin-bottom: 0.2rem;
    }
    /* Markdown plus tassé */
    .stMarkdown p { margin-bottom: 0.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Format compact (1k, 1M, 1B) sur l'axe Y uniquement
# (l'axe X est souvent catégoriel : mois, magasins…)
import plotly.io as pio
pio.templates["compact"] = go.layout.Template(
    layout=dict(yaxis=dict(tickformat="~s"))
)
pio.templates.default = "plotly+compact"

DATA_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Chargement & préparation des données
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    sales = pd.read_csv(DATA_DIR / "Sales.csv", sep=";", encoding="utf-8-sig")
    customers = pd.read_csv(DATA_DIR / "customer.csv", sep=";", encoding="utf-8-sig")
    geography = pd.read_csv(DATA_DIR / "geography.csv", sep=";", encoding="utf-8-sig")
    shops = pd.read_csv(DATA_DIR / "shops.csv", sep=";", encoding="utf-8-sig")

    # Colonnes parasites
    for d in (sales, customers):
        if "Unnamed: 0" in d.columns:
            d.drop(columns=["Unnamed: 0"], inplace=True)

    # Montants : virgule -> point
    sales["Ventes bruts"] = (
        sales["Ventes bruts"].astype(str).str.replace(",", ".").astype(float)
    )
    sales["Ventes nets"] = (
        sales["Ventes nets"].astype(str).str.replace(",", ".").astype(float)
    )

    # ------------------------------------------------------------------
    # Dates : règles métier
    # Les 3 catégories ci-dessous sont éparpillées sur toute la plage
    # SalesID (médianes ~5M, voisins en 2022/2023) → ce sont des
    # erreurs de saisie sur l'année, pas des ventes hors période.
    #   - année 1900    -> date du SalesID précédent
    #   - année 2042    -> date du SalesID précédent
    #   - date invalide -> date du SalesID précédent
    # ------------------------------------------------------------------
    sales = sales.sort_values("SalesID").reset_index(drop=True)
    sales["Date"] = pd.to_datetime(sales["Date"], format="%m/%d/%Y", errors="coerce")

    bad_year = sales["Date"].dt.year.isin([1900, 2042])
    sales.loc[bad_year, "Date"] = pd.NaT

    sales["Date"] = sales["Date"].ffill().bfill()

    sales["Year"] = sales["Date"].dt.year
    sales["Month"] = sales["Date"].dt.month
    sales["MonthName"] = sales["Date"].dt.strftime("%Y-%m")

    # ------------------------------------------------------------------
    # Outliers Ventes bruts : 7 lignes à 999 999 999 € (sentinelle/erreur),
    # toutes datées du 30/12/2023, avec un ratio net/brut anormalement bas
    # (~6e-9 vs 0.30 ± 0.008 sur le reste).
    # On les exclut.
    # ------------------------------------------------------------------
    ratio = sales["Ventes nets"] / sales["Ventes bruts"].replace(0, pd.NA)
    sales = sales[(sales["Ventes bruts"] < 10_000) & (ratio > 0.05)].copy()

    # Coût des ventes (coûts directs) = brut - net
    sales["Cout_ventes"] = sales["Ventes bruts"] - sales["Ventes nets"]

    # ------------------------------------------------------------------
    # CustomerID manquants : on crée un client "non défini" (ID 0)
    # ------------------------------------------------------------------
    sales["CustomerID"] = pd.to_numeric(sales["CustomerID"], errors="coerce")
    sales["CustomerID"] = sales["CustomerID"].fillna(0).astype("int64")

    # Normalisation des coquilles CustomerType : "B toC", "Bto C" -> "B2C"
    def _norm_ctype(x):
        s = str(x).replace(" ", "")
        if "BtoC" in s or "B2C" in s:
            return "B2C"
        if "BtoB" in s or "B2B" in s:
            return "B2B"
        return s
    customers["CustomerType"] = customers["CustomerType"].apply(_norm_ctype)

    if 0 not in customers["CustomerID"].values:
        customers = pd.concat(
            [
                pd.DataFrame(
                    [
                        {
                            "CustomerID": 0,
                            "CustomerType": "non défini",
                            "MemberCard": "FAUX",
                            "GeographyID": pd.NA,
                        }
                    ]
                ),
                customers,
            ],
            ignore_index=True,
        )

    # Geography : doublons (Nantes / Nante)
    geography = geography.drop_duplicates(subset=["GeographyID"], keep="first")

    # ZipCode : forçage texte sur 5 caractères (Nice 6000 -> 06000)
    geography["ZipCode"] = (
        geography["ZipCode"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    )

    # Jointures
    shops_geo = shops.merge(geography, on="GeographyID", how="left")
    df = sales.merge(
        shops_geo[["shop_id", "Name", "Costs", "City", "Adress"]],
        on="shop_id",
        how="left",
    )
    df = df.rename(columns={"Name": "Magasin", "Adress": "Region"})

    return df, sales, customers, geography, shops, shops_geo


df, sales_clean, customers, geography, shops, shops_geo = load_data()


# ---------------------------------------------------------------------------
# Sidebar : navigation + filtres
# ---------------------------------------------------------------------------
st.sidebar.title("🛒 Omnivend")
page = st.sidebar.radio(
    "Navigation",
    [
        "1. Préparation des données",
        "2. Dashboard",
        "3. Analyses complémentaires",
        "4. Recommandations",
        "5. Annexe - Détail technique",
        "6. Annexe - Recommandations détaillées",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filtres")
years = sorted(df["Year"].unique())
sel_years = st.sidebar.multiselect("Année", years, default=years)
shops_list = sorted(df["Magasin"].dropna().unique())
sel_shops = st.sidebar.multiselect("Magasin", shops_list, default=shops_list)
regions_list = sorted(df["Region"].dropna().unique())
sel_regions = st.sidebar.multiselect("Région", regions_list, default=regions_list)

mask = (
    df["Year"].isin(sel_years)
    & df["Magasin"].isin(sel_shops)
    & df["Region"].isin(sel_regions)
)
fdf = df[mask].copy()


def fmt_eur(x):
    """Formatage compact : 73 M€, 1,2 Md€, 850 k€, 42 €."""
    if x is None or pd.isna(x):
        return "—"
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e9:
        return f"{sign}{x / 1e9:.1f} Md€".replace(".", ",")
    if x >= 1e6:
        return f"{sign}{x / 1e6:.1f} M€".replace(".", ",")
    if x >= 1e3:
        return f"{sign}{x / 1e3:.0f} k€"
    return f"{sign}{x:.0f} €"


# ---------------------------------------------------------------------------
# PAGE 1 — Préparation des données
# ---------------------------------------------------------------------------
if page.startswith("1"):
    st.title("1. Préparation des données")

    st.markdown(
        """
        Les données fournies couvrent **2 années (2022-2023)** et regroupent
        4 sources : les **ventes**, les **magasins**, les **clients** et
        une référence **géographique**. Elles ont été consolidées dans un
        modèle unique pour permettre une analyse croisée.
        """
    )

    st.subheader("Modèle de données")
    st.markdown(
        """
        ```
        ┌──────────────┐         ┌────────────────────┐         ┌──────────────┐
        │   CLIENTS    │         │       VENTES       │         │   MAGASINS   │
        │──────────────│         │────────────────────│         │──────────────│
        │ CustomerID   │◄────────│ SalesID            │────────►│ shop_id      │
        │ Type B2B/B2C │         │ Date               │         │ Nom          │
        │ Carte fidé.  │         │ Magasin            │         │ Coûts indir. │
        └──────────────┘         │ Client             │         │ Région       │
                                 │ Ventes brutes      │         └──────────────┘
                                 │ Ventes nettes      │
                                 │ Coût des ventes ✦  │
                                 │ Mode de paiement   │
                                 └────────────────────┘
                                          │
                                          ▼
                                ┌──────────────────────┐
                                │      GÉOGRAPHIE      │
                                │──────────────────────│
                                │ Ville · CP · Région  │
                                └──────────────────────┘
        ```
        ✦ *Coût des ventes = Ventes brutes − Ventes nettes (champ calculé)*
        """
    )

    st.markdown("---")

    col1, col2 = st.columns([1, 1.4])
    with col1:
        st.subheader("Volumétrie")
        st.metric(
            "Ventes analysées",
            f"{len(sales_clean):,}".replace(",", " "),
        )
        c1, c2 = st.columns(2)
        c1.metric("Magasins", f"{len(shops)}")
        c2.metric("Régions", f"{geography['Adress'].nunique()}")
        st.caption(
            f"Période : {sales_clean['Date'].min():%d/%m/%Y} → "
            f"{sales_clean['Date'].max():%d/%m/%Y}"
        )

    with col2:
        st.subheader("Anomalies traitées")
        st.markdown(
            """
            La donnée brute contenait quelques incohérences que nous avons
            corrigées avant analyse :

            - **Ventes au montant manifestement faux** *(quelques milliards d'€
              artificiels)* → **exclues**.
            - **Dates erronées** *(années 1900, 2042, illisibles)*
              → **reconstituées** à partir de la chronologie des transactions.
            - **Clients non identifiés** → regroupés sous une catégorie
              **« non défini »**.
            - **Doublons et coquilles de référentiel** *(ville en double,
              code postal de Nice tronqué)* → **corrigés**.
            """
        )
        st.caption("📎 Détail technique disponible dans l'**Annexe**.")


# ---------------------------------------------------------------------------
# PAGE 2 — Dashboard principal
# ---------------------------------------------------------------------------
elif page.startswith("2"):
    st.title("2. Dashboard - Performance Omnivend")

    if fdf.empty:
        st.warning("Aucune donnée — élargis les filtres dans la barre latérale.")
        st.stop()

    # --- KPIs --------------------------------------------------------------
    ventes_brutes = fdf["Ventes bruts"].sum()
    ventes_nettes = fdf["Ventes nets"].sum()
    cout_ventes = fdf["Cout_ventes"].sum()
    taux_marge = ventes_nettes / ventes_brutes * 100 if ventes_brutes else 0

    # Coûts indirects : 1 fois par couple (magasin, année)
    couts_indirects = (
        fdf.groupby(["shop_id", "Year"])["Costs"].first().sum()
    )
    revenu_net = ventes_nettes - couts_indirects
    nb_transactions = len(fdf)
    panier_moyen = ventes_brutes / nb_transactions if nb_transactions else 0

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Ventes brutes", fmt_eur(ventes_brutes))
    k2.metric("Marge (ventes nettes)", fmt_eur(ventes_nettes), f"{taux_marge:.1f} %")
    k3.metric("Coût des ventes", fmt_eur(cout_ventes))
    k4.metric("Coûts indirects", fmt_eur(couts_indirects))
    k5.metric("Revenu net", fmt_eur(revenu_net))
    k6.metric("Panier moyen", fmt_eur(panier_moyen))

    st.markdown("---")

    # ====================================================================
    # ANALYSE PRINCIPALE — Revenu net par magasin (segmenté en 4 groupes)
    # ====================================================================
    st.subheader("⭐ Revenu net annuel par magasin")
    st.caption(
        "Magasins triés du plus déficitaire au plus rentable. "
        "**La couleur indique le groupe stratégique** — chaque groupe appelle "
        "un type d'action différent (cf. Recommandations)."
    )

    nb_annees_d = fdf["Year"].nunique()
    perf = (
        fdf.groupby(["shop_id", "Magasin"])
        .agg(Marge=("Ventes nets", "sum"))
        .reset_index()
    )
    couts_a = (
        fdf.groupby(["shop_id", "Year"])["Costs"].first()
        .groupby("shop_id").sum()
        .reset_index().rename(columns={"Costs": "CoutsTotaux"})
    )
    perf = perf.merge(couts_a, on="shop_id")
    perf["MargeAn"] = perf["Marge"] / nb_annees_d
    perf["CoutsAn"] = perf["CoutsTotaux"] / nb_annees_d
    perf["RevenuNet"] = perf["MargeAn"] - perf["CoutsAn"]
    perf["Ratio"] = perf["MargeAn"] / perf["CoutsAn"]

    # Segmentation en 4 groupes stratégiques
    # - Champion       : gros volume + ratio > 1.5
    # - Géant          : gros volume + ratio entre 0.5 et 1.5 (à optimiser)
    # - Petit          : petit volume + ratio >= 0.5 (boost commercial)
    # - Critique       : ratio < 0.5 (intervention urgente)
    def assign_group(row):
        if row["Ratio"] < 0.5:
            return "🚨 Déficit critique"
        if row["MargeAn"] > 500_000 and row["Ratio"] > 1.5:
            return "🥇 Champion"
        if row["MargeAn"] > 500_000:
            return "🔧 Géant à optimiser"
        return "📍 Petit magasin"

    perf["Groupe"] = perf.apply(assign_group, axis=1)
    perf = perf.sort_values("RevenuNet", ascending=True).reset_index(drop=True)

    GROUP_COLORS = {
        "🥇 Champion": "#f5b800",
        "📍 Petit magasin": "#5cb85c",
        "🔧 Géant à optimiser": "#3498db",
        "🚨 Déficit critique": "#d62728",
    }

    fig_main = go.Figure()
    for grp, color in GROUP_COLORS.items():
        sub = perf[perf["Groupe"] == grp]
        if sub.empty:
            continue
        fig_main.add_trace(go.Bar(
            y=sub["Magasin"],
            x=sub["RevenuNet"],
            orientation="h",
            name=grp,
            marker_color=color,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Revenu net : %{customdata[0]}<br>"
                "Marge : %{customdata[1]}<br>"
                "Coûts indirects : %{customdata[2]}<br>"
                "Taux de couverture : %{customdata[3]:.2f}<extra></extra>"
            ),
            customdata=[
                [fmt_eur(r["RevenuNet"]), fmt_eur(r["MargeAn"]),
                 fmt_eur(r["CoutsAn"]), r["Ratio"]]
                for _, r in sub.iterrows()
            ],
        ))

    # Annotations : valeur au bout de chaque barre
    max_abs = perf["RevenuNet"].abs().max()
    for _, r in perf.iterrows():
        offset = max_abs * 0.015 * (1 if r["RevenuNet"] >= 0 else -1)
        fig_main.add_annotation(
            x=r["RevenuNet"] + offset,
            y=r["Magasin"],
            text=f"<b>{fmt_eur(r['RevenuNet'])}</b>",
            showarrow=False,
            xanchor="left" if r["RevenuNet"] >= 0 else "right",
            font=dict(size=11, color="#333"),
        )

    fig_main.add_vline(
        x=0, line_color="#222", line_width=2,
        line_dash="dot",
    )
    # Annotation "Seuil de rentabilité" placée nettement au-dessus du plot,
    # entre la légende et le haut du graphe
    fig_main.add_annotation(
        x=0, y=1.06, xref="x", yref="paper",
        text="<i>Seuil de rentabilité</i>",
        showarrow=False, font=dict(size=10, color="#666"),
    )

    fig_main.update_layout(
        height=640,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.14,
            xanchor="right", x=1,
            title=None,
        ),
        margin=dict(t=110, b=40, l=10, r=140),
        xaxis=dict(
            ticksuffix=" €",
            tickformat="~s",
            title="Revenu net annuel  (Marge − Coûts indirects)",
            zeroline=False,
        ),
        yaxis=dict(tickmode="linear", dtick=1, automargin=True),
        barmode="relative",
    )
    st.plotly_chart(fig_main, use_container_width=True)

    with st.expander("🔍 Voir le détail : marge vs coûts indirects par magasin"):
        fig_detail = go.Figure()
        fig_detail.add_trace(go.Bar(
            y=perf["Magasin"], x=-perf["CoutsAn"],
            orientation="h", name="Coûts indirects",
            marker_color="#ff7f0e",
            hovertemplate="<b>%{y}</b><br>Coûts : %{customdata}<extra></extra>",
            customdata=[fmt_eur(v) for v in perf["CoutsAn"]],
        ))
        fig_detail.add_trace(go.Bar(
            y=perf["Magasin"], x=perf["MargeAn"],
            orientation="h", name="Marge (ventes nettes)",
            marker_color="#2ca02c",
            hovertemplate="<b>%{y}</b><br>Marge : %{customdata}<extra></extra>",
            customdata=[fmt_eur(v) for v in perf["MargeAn"]],
        ))
        fig_detail.add_vline(x=0, line_color="#444", line_width=1)
        fig_detail.update_layout(
            height=620, barmode="overlay",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                       xanchor="right", x=1),
            margin=dict(t=50, b=40, l=10, r=20),
            xaxis=dict(
                ticksuffix=" €", tickformat="~s",
                title="← Coûts indirects   |   Marge →",
            ),
            yaxis=dict(tickmode="linear", dtick=1, automargin=True),
        )
        st.plotly_chart(fig_detail, use_container_width=True)
        st.caption(
            "Lecture : barre verte plus longue que l'orange = rentable. "
            "Sinon = déficitaire."
        )

    st.markdown("---")

    # --- Balance Marge / Coûts par magasin (pleine largeur) ----------------
    st.subheader("Balance Marge / Coûts par magasin")
    st.caption(
        "Pour chaque magasin : marge annuelle (vert) vs coûts indirects "
        "annuels (orange). Si la verte dépasse l'orange = rentable."
    )
    balance = (
        fdf.groupby(["shop_id", "Magasin"])
        .agg(Marge=("Ventes nets", "sum"))
        .reset_index()
    )
    balance_couts = (
        fdf.groupby(["shop_id", "Year"])["Costs"].first()
        .groupby("shop_id").sum()
        .reset_index().rename(columns={"Costs": "CoutsTotaux"})
    )
    balance = balance.merge(balance_couts, on="shop_id")
    nb_an = fdf["Year"].nunique()
    balance["MargeAn"] = balance["Marge"] / nb_an
    balance["CoutsAn"] = balance["CoutsTotaux"] / nb_an
    balance = balance.sort_values("MargeAn", ascending=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        y=balance["Magasin"], x=balance["MargeAn"],
        orientation="h", name="Marge",
        marker_color="#2ca02c",
        hovertemplate="<b>%{y}</b><br>Marge : %{customdata}<extra></extra>",
        customdata=[fmt_eur(v) for v in balance["MargeAn"]],
    ))
    fig2.add_trace(go.Bar(
        y=balance["Magasin"], x=balance["CoutsAn"],
        orientation="h", name="Coûts indirects",
        marker_color="#ff7f0e",
        hovertemplate="<b>%{y}</b><br>Coûts : %{customdata}<extra></extra>",
        customdata=[fmt_eur(v) for v in balance["CoutsAn"]],
    ))
    fig2.update_layout(
        height=520,
        barmode="group",
        xaxis=dict(ticksuffix=" €", tickformat="~s"),
        yaxis=dict(tickmode="linear", dtick=1, automargin=True),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, title=None,
        ),
        margin=dict(t=40, b=20, l=10, r=10),
    )
    st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE 3 — Analyses complémentaires
# ---------------------------------------------------------------------------
elif page.startswith("3"):
    st.title("3. Analyses complémentaires")

    if fdf.empty:
        st.warning("Aucune donnée.")
        st.stop()

    # ====================================================================
    # Évolution mensuelle - Ventes brutes (pleine largeur)
    # ====================================================================
    st.subheader("Évolution mensuelle - Ventes brutes")
    st.caption(
        "Le taux de marge étant uniforme (~30 %), la courbe de marge a "
        "exactement la même forme que les ventes brutes — un seul indicateur suffit."
    )
    monthly = (
        fdf.groupby("MonthName")["Ventes bruts"]
        .sum()
        .reset_index()
        .sort_values("MonthName")
    )
    mois_fr = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin",
               "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]
    years = monthly["MonthName"].str[:4].tolist()
    months = [mois_fr[int(m[5:7]) - 1] for m in monthly["MonthName"]]
    x_multi = [years, months]

    fig_evol = go.Figure()
    fig_evol.add_trace(go.Scatter(
        x=x_multi, y=monthly["Ventes bruts"],
        name="Ventes brutes", mode="lines+markers",
        line=dict(width=3, color="#1f77b4"),
        fill="tozeroy", fillcolor="rgba(31,119,180,0.1)",
    ))
    fig_evol.update_layout(
        height=380,
        xaxis=dict(tickfont=dict(size=11), showdividers=True,
                   dividercolor="#888", dividerwidth=1),
        yaxis=dict(title="", tickformat="~s", ticksuffix=" €"),
        hovermode="x unified",
        showlegend=False,
        margin=dict(t=30, b=60, l=20, r=20),
    )
    st.plotly_chart(fig_evol, use_container_width=True)

    st.markdown("---")

    # ====================================================================
    # Saisonnalité + Performance par région côte à côte
    # ====================================================================
    c_s, c_r = st.columns(2)

    with c_s:
        st.subheader("Saisonnalité - Ventes brutes par mois")
        st.caption("Ventes brutes = demande client effective.")
        season = (
            fdf.assign(MoisNum=fdf["Date"].dt.month)
            .groupby("MoisNum")["Ventes bruts"]
            .sum()
            .reset_index()
        )
        season["Mois"] = season["MoisNum"].apply(lambda m: mois_fr[m - 1])
        fig_sai = px.bar(
            season, x="Mois", y="Ventes bruts",
            color="Ventes bruts", color_continuous_scale="Blues",
        )
        fig_sai.update_layout(
            height=380, coloraxis_showscale=False,
            yaxis_title="", yaxis_ticksuffix=" €",
            margin=dict(t=20, b=20, l=10, r=10),
        )
        st.plotly_chart(fig_sai, use_container_width=True)

    with c_r:
        st.subheader("Performance par région")
        st.caption(
            "Ventes brutes vs marge agrégées par grande région "
            "(IDF, PACA, etc. — départements regroupés)."
        )
        region_perf = fdf.copy()
        region_perf["RegionGrande"] = region_perf["Region"].str.split(" - ").str[0]
        region_perf = (
            region_perf.groupby("RegionGrande")
            .agg(VentesBrutes=("Ventes bruts", "sum"),
                 Marge=("Ventes nets", "sum"),
                 NbVentes=("SalesID", "count"))
            .reset_index()
            .sort_values("VentesBrutes", ascending=False)
        )
        fig_reg = px.bar(
            region_perf, x="RegionGrande",
            y=["VentesBrutes", "Marge"], barmode="group",
            color_discrete_sequence=["#1f77b4", "#2ca02c"],
            labels={"value": "", "variable": ""},
        )
        fig_reg.update_layout(
            height=380, xaxis_tickangle=-30,
            yaxis_title="", yaxis_ticksuffix=" €", xaxis_title="",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1, title=None),
            margin=dict(t=20, b=20, l=10, r=10),
        )
        st.plotly_chart(fig_reg, use_container_width=True)

    st.markdown("---")

    # ====================================================================
    # FOCUS : meilleur magasin vs 3 plus déficitaires
    # ====================================================================
    st.subheader("🔬 Focus magasins extrêmes")
    st.caption(
        "Le meilleur magasin et les 3 plus déficitaires se détachent "
        "nettement du reste — cette section les analyse en détail."
    )

    nb_annees = sales_clean["Date"].dt.year.nunique()
    extr = sales_clean.groupby("shop_id")["Ventes nets"].sum().reset_index()
    extr = extr.merge(
        shops_geo[["shop_id", "Name", "Costs", "City"]], on="shop_id"
    )
    extr["VN_an"] = extr["Ventes nets"] / nb_annees
    extr["Taux"] = extr["VN_an"] / extr["Costs"]

    top_1 = extr.nlargest(1, "Taux")["Name"].tolist()
    bot_3 = extr.nsmallest(3, "Taux")["Name"].tolist()
    focus_shops = top_1 + bot_3

    focus_df = fdf[fdf["Magasin"].isin(focus_shops)].copy()

    summary = (
        focus_df.groupby(["shop_id", "Magasin", "City"])
        .agg(
            Brut=("Ventes bruts", "sum"),
            Marge=("Ventes nets", "sum"),
            CoutVentes=("Cout_ventes", "sum"),
            NbVentes=("SalesID", "count"),
            PanierMoyen=("Ventes bruts", "mean"),
        )
        .reset_index()
    )
    couts_f = (
        focus_df.groupby(["shop_id", "Year"])["Costs"].first()
        .groupby("shop_id").sum()
        .reset_index().rename(columns={"Costs": "CoutsTotaux"})
    )
    summary = summary.merge(couts_f, on="shop_id")
    summary["BrutAn"] = summary["Brut"] / nb_annees
    summary["MargeAn"] = summary["Marge"] / nb_annees
    summary["CoutVentesAn"] = summary["CoutVentes"] / nb_annees
    summary["CoutsAn"] = summary["CoutsTotaux"] / nb_annees
    summary["RevenuNetAn"] = summary["MargeAn"] - summary["CoutsAn"]
    summary["TauxCouv"] = summary["MargeAn"] / summary["CoutsAn"]
    summary["Categorie"] = summary["Magasin"].apply(
        lambda x: "🥇 Meilleur" if x in top_1 else "⚠️ Déficitaire"
    )
    summary = summary.sort_values("TauxCouv", ascending=False).reset_index(drop=True)

    # ---- Comparaison visuelle ----
    st.markdown("##### Décomposition annuelle (€ / an)")
    fig_focus = go.Figure()
    fig_focus.add_trace(go.Bar(
        x=summary["Magasin"], y=summary["BrutAn"],
        name="Ventes brutes", marker_color="#1f77b4",
    ))
    fig_focus.add_trace(go.Bar(
        x=summary["Magasin"], y=summary["CoutVentesAn"],
        name="Coût des ventes (directs)", marker_color="#9467bd",
    ))
    fig_focus.add_trace(go.Bar(
        x=summary["Magasin"], y=summary["MargeAn"],
        name="Marge", marker_color="#2ca02c",
    ))
    fig_focus.add_trace(go.Bar(
        x=summary["Magasin"], y=summary["CoutsAn"],
        name="Coûts indirects", marker_color="#ff7f0e",
    ))
    fig_focus.add_trace(go.Bar(
        x=summary["Magasin"], y=summary["RevenuNetAn"],
        name="Revenu net",
        marker_color=[
            "#2ca02c" if v >= 0 else "#d62728"
            for v in summary["RevenuNetAn"]
        ],
    ))
    fig_focus.update_layout(
        height=420, barmode="group",
        yaxis_ticksuffix=" €",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        ),
        margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig_focus, use_container_width=True)

    # ---- Lecture rapide ----
    moy_couts = sales_clean.groupby("shop_id").first()  # placeholder
    couts_moy_an = shops_geo["Costs"].mean()
    marge_moy_an = (
        sales_clean.groupby("shop_id")["Ventes nets"].sum().mean() / nb_annees
    )
    st.info(
        f"**Lecture clé** — La marge nette est uniforme (~30 % du brut) "
        f"sur tous les magasins. La rentabilité dépend donc presque "
        f"uniquement du **rapport volume de ventes / coûts indirects**. "
        f"Les 3 magasins en rouge ont des coûts indirects élevés "
        f"(moy. {fmt_eur(summary[summary['TauxCouv']<1]['CoutsAn'].mean())}/an) "
        f"sans le volume de ventes correspondant."
    )

    # ====================================================================
    # ANALYSE CLIENT — carte de fidélité & B2B vs B2C
    # ====================================================================
    st.markdown("---")
    st.subheader("👥 Analyse client — Carte de fidélité & B2B / B2C")
    st.caption(
        "⚠️ Anomalie probable du jeu fictif : la pénétration carte est "
        "**bimodale** — 15 magasins à 100 % et 4 à ~27 %, sans gradient "
        "intermédiaire. Dans des données réelles on s'attendrait à une "
        "distribution continue. Les ratios (× 3,1) restent toutefois "
        "interprétables comme un effet net de la carte."
    )

    # Jointure sales x customers, exclusion du client non identifié (id=0)
    cli = customers.rename(
        columns={"CustomerID": "CustomerID", "MemberCard": "Carte"}
    )[["CustomerID", "CustomerType", "Carte"]]
    cli["Carte"] = cli["Carte"].replace({"VRAI": "Avec carte", "FAUX": "Sans carte"})

    sales_cli = fdf.merge(cli, on="CustomerID", how="left")
    sales_cli = sales_cli[sales_cli["CustomerID"] != 0]

    def _agg_segment(df, col):
        g = df.groupby(col).agg(
            nb_trans=("SalesID", "count"),
            nb_clients=("CustomerID", "nunique"),
            ca=("Ventes bruts", "sum"),
            panier=("Ventes bruts", "mean"),
        ).reset_index()
        g["trans_par_client"] = g["nb_trans"] / g["nb_clients"]
        return g

    seg_carte = _agg_segment(sales_cli, "Carte")
    seg_type = _agg_segment(sales_cli, "CustomerType")

    cA, cB = st.columns(2)

    # ---------- A. Carte de fidélité ----------
    with cA:
        st.markdown("##### 💳 Avec carte vs Sans carte")

        # KPI : panier moyen + transactions par client
        kc1, kc2 = st.columns(2)
        avec = seg_carte[seg_carte["Carte"] == "Avec carte"].iloc[0]
        sans = seg_carte[seg_carte["Carte"] == "Sans carte"].iloc[0]
        delta_panier = (avec["panier"] - sans["panier"]) / sans["panier"] * 100
        delta_trans = (avec["trans_par_client"] - sans["trans_par_client"]) / sans["trans_par_client"] * 100
        kc1.metric(
            "Panier moyen — Carte",
            fmt_eur(avec["panier"]),
            f"{delta_panier:+.1f} % vs sans",
        )
        kc2.metric(
            "Transactions / client — Carte",
            f"{avec['trans_par_client']:.2f}",
            f"{delta_trans:+.1f} % vs sans",
        )

        # 2 charts simples côte à côte (1 axe Y chacun)
        gc1, gc2 = st.columns(2)
        with gc1:
            fig_panier = px.bar(
                seg_carte, x="Carte", y="panier",
                text=[fmt_eur(v) for v in seg_carte["panier"]],
                color="Carte",
                color_discrete_map={"Avec carte": "#2ca02c", "Sans carte": "#a8d8a8"},
            )
            fig_panier.update_traces(textposition="outside")
            fig_panier.update_layout(
                height=260, showlegend=False,
                title=dict(text="Panier moyen", font=dict(size=13)),
                yaxis=dict(title="", ticksuffix=" €"),
                xaxis=dict(title=""),
                margin=dict(t=40, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_panier, use_container_width=True)
        with gc2:
            fig_freq = px.bar(
                seg_carte, x="Carte", y="trans_par_client",
                text=[f"{v:.2f}" for v in seg_carte["trans_par_client"]],
                color="Carte",
                color_discrete_map={"Avec carte": "#1f77b4", "Sans carte": "#aac8e0"},
            )
            fig_freq.update_traces(textposition="outside")
            fig_freq.update_layout(
                height=260, showlegend=False,
                title=dict(text="Transactions / client", font=dict(size=13)),
                yaxis=dict(title=""),
                xaxis=dict(title=""),
                margin=dict(t=40, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_freq, use_container_width=True)

    # ---------- B. B2B vs B2C ----------
    with cB:
        st.markdown("##### 🏢 B2B vs B2C")

        kt1, kt2 = st.columns(2)
        b2b = seg_type[seg_type["CustomerType"] == "B2B"].iloc[0]
        b2c = seg_type[seg_type["CustomerType"] == "B2C"].iloc[0]
        delta_p = (b2b["panier"] - b2c["panier"]) / b2c["panier"] * 100
        delta_t = (b2b["trans_par_client"] - b2c["trans_par_client"]) / b2c["trans_par_client"] * 100
        kt1.metric(
            "Panier moyen — B2B",
            fmt_eur(b2b["panier"]),
            f"{delta_p:+.1f} % vs B2C",
        )
        kt2.metric(
            "Transactions / client — B2B",
            f"{b2b['trans_par_client']:.2f}",
            f"{delta_t:+.1f} % vs B2C",
        )

        gt1, gt2 = st.columns(2)
        with gt1:
            fig_pt = px.bar(
                seg_type, x="CustomerType", y="panier",
                text=[fmt_eur(v) for v in seg_type["panier"]],
                color="CustomerType",
                color_discrete_map={"B2C": "#2ca02c", "B2B": "#a8d8a8"},
            )
            fig_pt.update_traces(textposition="outside")
            fig_pt.update_layout(
                height=260, showlegend=False,
                title=dict(text="Panier moyen", font=dict(size=13)),
                yaxis=dict(title="", ticksuffix=" €"),
                xaxis=dict(title=""),
                margin=dict(t=40, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_pt, use_container_width=True)
        with gt2:
            fig_ft = px.bar(
                seg_type, x="CustomerType", y="trans_par_client",
                text=[f"{v:.2f}" for v in seg_type["trans_par_client"]],
                color="CustomerType",
                color_discrete_map={"B2C": "#1f77b4", "B2B": "#aac8e0"},
            )
            fig_ft.update_traces(textposition="outside")
            fig_ft.update_layout(
                height=260, showlegend=False,
                title=dict(text="Transactions / client", font=dict(size=13)),
                yaxis=dict(title=""),
                xaxis=dict(title=""),
                margin=dict(t=40, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_ft, use_container_width=True)

    # Lecture
    pen_carte = (
        sales_cli[sales_cli["Carte"] == "Avec carte"]["SalesID"].count()
        / len(sales_cli) * 100
    )
    st.caption(
        f"**Pénétration de la carte** : {pen_carte:.1f} % des transactions "
        f"sont réalisées par un porteur de carte. "
        f"**Conclusion** : la carte a un impact très net — les porteurs de "
        f"carte ont un **panier moyen ~3× supérieur** (142 € vs 46 €) et "
        f"**reviennent ~3× plus souvent** (6,21 vs 1,99 visites sur 2 ans). "
        f"La fidélisation fonctionne clairement, c'est un levier à amplifier."
    )

    st.markdown("---")
    st.subheader("Tableau récapitulatif par magasin")
    recap = (
        fdf.groupby(["shop_id", "Magasin", "City", "Region"])
        .agg(
            VentesBrutes=("Ventes bruts", "sum"),
            Marge=("Ventes nets", "sum"),
            NbVentes=("SalesID", "count"),
        )
        .reset_index()
    )
    couts = (
        fdf.groupby(["shop_id", "Year"])["Costs"]
        .first()
        .groupby("shop_id")
        .sum()
        .reset_index()
        .rename(columns={"Costs": "CoutsIndirects"})
    )
    recap = recap.merge(couts, on="shop_id")
    recap["RevenuNet"] = recap["Marge"] - recap["CoutsIndirects"]
    recap["TauxMarge%"] = (recap["Marge"] / recap["VentesBrutes"] * 100).round(1)
    recap = recap.sort_values("RevenuNet", ascending=False)
    st.dataframe(
        recap.style.format(
            {
                "VentesBrutes": fmt_eur,
                "Marge": fmt_eur,
                "CoutsIndirects": fmt_eur,
                "RevenuNet": fmt_eur,
                "TauxMarge%": "{:.1f} %",
            }
        ),
        use_container_width=True,
    )

    # ---- Info secondaire : mix paiement ----
    st.markdown("---")
    with st.expander("Mix de paiement (information secondaire)"):
        c_pay, _ = st.columns([1, 2])
        with c_pay:
            pay = fdf.groupby("PaymentType")["Ventes bruts"].sum().reset_index()
            fig_pay = px.pie(
                pay, names="PaymentType", values="Ventes bruts", hole=0.5,
            )
            fig_pay.update_layout(
                height=260, margin=dict(t=10, b=10, l=10, r=10),
                showlegend=True,
            )
            st.plotly_chart(fig_pay, use_container_width=True)
        st.caption(
            "Le mix carte / cash est relativement stable et n'a pas "
            "d'impact direct sur les recommandations stratégiques."
        )


# ---------------------------------------------------------------------------
# PAGE 4 — Recommandations
# ---------------------------------------------------------------------------
elif page.startswith("4"):
    st.title("4. Recommandations")

    st.markdown(
        """
        ### 🎯 Constats clés

        1. **Omnivend déficitaire de −1,15 M€/an** : marge **11,0 M€** vs coûts indirects **12,1 M€**.
        2. **8 / 19 magasins déficitaires** — les 3 critiques (Terminal Market, Jet Marché, Super-Miam) pèsent **−1,9 M€/an** → **sans eux, l'enseigne serait rentable**.
        3. **Taux de marge homogène (~30 %)** → la rentabilité dépend du **volume vs coûts indirects**, pas du mix produit.
        4. **Saisonnalité** : 🔝 Nov, Déc, Juin, Mai · 🔻 Fév, Mars, Avril, Sept.

        ---

        ### 💡 Recommandations par groupe *(volume × ratio marge/coûts)*

        | Groupe | Magasins | Action principale |
        |---|---|---|
        | 🥇 **Champion** (1) | Maxi Délices | **Étudier et répliquer** les bonnes pratiques |
        | 🔧 **Géants à optimiser** (6) | Galerie Gourmande, Hyper Saveurs, Escale Gourmande, Grand Hall, Centre Saveurs, AéroBoutique | **Réduire les coûts** (foncier, énergie, RH, achats) — −5 % sur Hyper Saveurs = +85 k€ |
        | 📍 **Petits magasins** (9) | Marché du Quartier, Le Petit Panier, etc. | **Booster le volume** (animation, horaires, événements) |
        | 🚨 **Déficits critiques** (3) | Terminal Market, Jet Marché, Super-Miam | **Diagnostic + restructuration**, sinon fermer/relocaliser |

        ---

        ### 🎯 Leviers transversaux

        - **💳 Carte de fidélité** : panier **× 3,1**, fréquence **× 3,1**, pénétration 80 % → **pousser sur les 20 % sans carte**, cibler les déficitaires.
        - **🏢 B2B** : panier −63 %, < 3 % du CA → **ne pas surinvestir**, focus B2C.
        - **🛒 Centrale d'achats** : +2 points de marge = **+1,46 M€/an** → bascule l'enseigne dans le vert à elle seule.

        ---

        ### 🌍 Fermeture / relocalisation *(plan B des critiques)*

        Les 3 critiques sont en concurrence interne dans leur région → si le redressement échoue :

        | À fermer | → À ouvrir | Région cible |
        |---|---|---|
        | **Super-Miam** (Paris) | **Bordeaux** | Nouvelle-Aquitaine |
        | **Jet Marché** (Marseille) | **Strasbourg** | Grand Est |
        | **Terminal Market** (Nice) | **Rennes** | Bretagne |

        6 / 13 régions couvertes → 3 régions vides + métropoles >200 k hab. = pas de cannibalisation, marché à conquérir.

        ---

        ### 🎯 Plan d'action

        | Horizon | Action |
        |---|---|
        | **0-3 mois** | Audit des 3 critiques + benchmark Maxi Délices |
        | **3-12 mois** | Redressement Géants (coûts) + boost Petits magasins + déploiement carte fidélité |
        | **> 1 an** | Décision fermeture/relocalisation + standardisation pratiques Champion |

        ---

        ### 🔍 Pistes d'approfondissement

        - **Analyse géomarketing** : croiser avec INSEE (population, revenus, concurrence) pour calibrer les objectifs par magasin.
        - **Qualité de la donnée** : pérenniser les contrôles automatiques (dates, montants aberrants, référentiels).

        📎 *Questions client, KPI de suivi, détail par groupe → cf. Annexe Recommandations.*
        """
    )


# ---------------------------------------------------------------------------
# PAGE 5 — Annexe : détail technique de la préparation
# ---------------------------------------------------------------------------
elif page.startswith("5"):
    st.title("5. Annexe — Détail technique de la préparation")
    st.caption(
        "Cette page documente précisément les opérations de nettoyage "
        "appliquées en amont de l'analyse. Elle est destinée aux questions "
        "techniques."
    )

    st.subheader("Sources brutes")
    st.markdown(
        """
        | Fichier         | Lignes  | Description                              |
        |-----------------|---------|------------------------------------------|
        | `Sales.csv`     | 593 358 | Ventes individuelles 2022-2023           |
        | `customer.csv`  | 206 693 | Référentiel clients (B2B/B2C, fidélité)  |
        | `shops.csv`     | 19      | Magasins + coûts indirects annuels       |
        | `geography.csv` | 11      | Référentiel ville / région               |
        """
    )

    st.markdown("---")
    st.subheader("Détail des règles de nettoyage")

    st.markdown(
        """
        **1. Doublon référentiel — `geography`**
        `GeographyID = 1004` apparaît deux fois (Nantes / *Nante* —
        coquille). Conservation de la première occurrence pour éviter de
        dupliquer les ventes du magasin Nantais lors de la jointure.

        **2. Code postal Nice**
        `6000` → `06000`. Le `0` initial avait disparu lors d'un import
        en format numérique. Tous les codes postaux ont été normalisés
        sur 5 caractères texte.

        **3. Outliers ventes brutes — 7 lignes**
        Toutes datées du 30/12/2023, avec `Ventes bruts = 999 999 999 €`
        (valeur sentinelle) et `Ventes nets` de 6-27 €. Le ratio net/brut
        observé sur le reste du jeu est très stable (≈ 30 % ± 0,8 %),
        ce qui rend ces 7 lignes facilement détectables. **Lignes
        exclues** — elles gonflaient le total de ~7 Md€ artificiels
        (sur ~73 M€ réels).

        **4. CustomerID manquants — 1 186 lignes**
        Création d'un client générique `CustomerID = 0`
        (`CustomerType = "non défini"`, `MemberCard = FAUX`) ajouté à
        la table `customer`. Les ventes orphelines pointent désormais
        vers ce client.

        **5. Dates aberrantes — 305 lignes**
        Les SalesID concernés sont **dispersés** sur toute la plage
        (médianes ~5 M, voisins en 2022/2023), ce qui invalide
        l'hypothèse "vraies ventes hors période" et confirme une
        **erreur de saisie sur l'année** :

        | Anomalie  | Lignes | Traitement                     |
        |-----------|--------|--------------------------------|
        | 1900      | 150    | Date du SalesID précédent      |
        | 2042      | 57     | Date du SalesID précédent      |
        | Illisible | 98     | Date du SalesID précédent      |

        L'imputation suppose que les SalesID sont attribués
        chronologiquement, ce qui est cohérent avec la donnée. L'erreur
        résiduelle sur 305/593 351 lignes (0,05 %) est négligeable sur
        les agrégats.

        **6. Champ calculé**
        `Coût des ventes = Ventes brutes − Ventes nettes` (coûts directs
        liés à la transaction).

        **7. Coûts indirects (table `shops`)**
        `Costs` représente les coûts indirects **annuels** par magasin
        (foncier, salaires, électricité, assurance). Pour le calcul du
        revenu net, on les compte une fois par couple (magasin × année).
        """
    )

    st.markdown("---")
    st.subheader("Aperçu des tables après nettoyage")
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Ventes", "Magasins", "Géographie", "Clients"]
    )

    with tab1:
        ventes_view = sales_clean.rename(
            columns={
                "SalesID": "ID Vente",
                "Date": "Date",
                "Ventes bruts": "Ventes brutes",
                "Ventes nets": "Ventes nettes",
                "Cout_ventes": "Coût des ventes",
                "PaymentType": "Mode de paiement",
                "CustomerID": "ID Client",
                "shop_id": "ID Magasin",
            }
        )[
            [
                "ID Vente", "Date", "ID Magasin", "ID Client",
                "Ventes brutes", "Ventes nettes", "Coût des ventes",
                "Mode de paiement",
            ]
        ]
        st.dataframe(ventes_view.head(20), use_container_width=True, hide_index=True)
        st.caption(
            f"{len(sales_clean):,} lignes · "
            f"{sales_clean['Date'].min():%d/%m/%Y} → "
            f"{sales_clean['Date'].max():%d/%m/%Y}"
        )

    with tab2:
        # Tout en base annuelle : ventes nettes moyennes / an et coûts annuels
        nb_annees = sales_clean["Date"].dt.year.nunique()
        ventes_par_magasin = (
            sales_clean.groupby("shop_id")["Ventes nets"].sum().reset_index()
        )
        ventes_par_magasin["Ventes nettes (annuelles)"] = (
            ventes_par_magasin["Ventes nets"] / nb_annees
        )
        ventes_par_magasin = ventes_par_magasin.drop(columns=["Ventes nets"])

        magasins_view = shops_geo.merge(ventes_par_magasin, on="shop_id", how="left")
        magasins_view["Taux de couverture"] = (
            magasins_view["Ventes nettes (annuelles)"] / magasins_view["Costs"]
        )
        magasins_view = magasins_view.rename(
            columns={
                "shop_id": "ID Magasin",
                "Name": "Magasin",
                "Costs": "Coûts indirects (annuels)",
                "City": "Ville",
                "ZipCode": "Code postal",
                "Adress": "Région - Département",
            }
        )[
            [
                "ID Magasin", "Magasin", "Ville", "Code postal",
                "Région - Département",
                "Ventes nettes (annuelles)", "Coûts indirects (annuels)",
                "Taux de couverture",
            ]
        ].sort_values("Taux de couverture", ascending=False)

        st.dataframe(
            magasins_view.style.format(
                {
                    "Ventes nettes (annuelles)": fmt_eur,
                    "Coûts indirects (annuels)": fmt_eur,
                    "Taux de couverture": "{:.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "**Taux de couverture** = Ventes nettes annuelles (marge) / "
            f"Coûts indirects annuels. Moyenne sur {nb_annees} ans (2022-2023). "
            "> 1 → magasin rentable · < 1 → marge insuffisante pour couvrir les coûts."
        )

    with tab3:
        geo_view = geography.rename(
            columns={
                "GeographyID": "ID Géographie",
                "Country": "Pays",
                "City": "Ville",
                "ZipCode": "Code postal",
                "Adress": "Région - Département",
            }
        )
        st.dataframe(geo_view, use_container_width=True, hide_index=True)

    with tab4:
        clients_view = customers.rename(
            columns={
                "CustomerID": "ID Client",
                "CustomerType": "Type",
                "MemberCard": "Carte fidélité",
                "GeographyID": "ID Géographie",
            }
        )
        st.dataframe(clients_view.head(20), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# PAGE 6 — Annexe : recommandations détaillées
# ---------------------------------------------------------------------------
elif page.startswith("6"):
    st.title("6. Annexe — Recommandations détaillées")
    st.caption(
        "Cette page documente les éléments d'approfondissement des "
        "recommandations : justifications par groupe, questions à poser "
        "au client, KPI de suivi et pistes d'approfondissement."
    )

    st.markdown(
        """
        ### 💡 Détail des recommandations par groupe

        #### 🥇 Champion — *Maxi Délices*
        1,0 M€ de marge/an, ratio 1,94, +488 k€ de revenu net.
        Seul magasin qui combine **gros volume ET excellent ratio**.

        **Action** : comprendre **pourquoi ça marche** (emplacement, équipe,
        mix produits, politique commerciale locale) et **transposer** ces
        bonnes pratiques aux autres magasins.

        #### 🔧 Géants à optimiser — *6 magasins*
        Gros volume (> 500 k€ marge/an) mais ratio proche de 1 → ces
        magasins génèrent beaucoup de business mais leur structure de coûts
        engloutit la marge.

        **Action** : **réduire les coûts**. Leviers : renégociation
        foncière, optimisation énergétique, productivité RH, mutualisation
        des achats. À 5 % de coûts économisés sur Hyper Saveurs (1,7 M€/an),
        on gagne 85 k€ — bien plus efficace que d'augmenter les volumes.

        #### 📍 Petits magasins — *9 magasins*
        Volume modeste, tous proches de l'équilibre (−130 k€ à +80 k€).
        Profil de risque faible.

        **Action** : **booster le volume** (animation commerciale, horaires,
        événements, communication locale). Pour ceux à ratio < 1, vérifier
        en parallèle que les coûts sont alignés avec la zone (un magasin
        de quartier ne doit pas porter une structure de coûts de centre-ville).

        #### 🚨 Déficits critiques — *3 magasins*
        Ratio < 0,5 → marge couvre 26-41 % des coûts. **−1,9 M€/an** cumulés.

        **Action** : **diagnostic + restructuration** dans l'ordre suivant :
        1. **Auditer les coûts** (loyer, salaires, énergie).
        2. **Renégocier ou résilier** les contrats pesants.
        3. **Plan de relance commerciale** ciblé — *uniquement après* baisse des coûts.
        4. **Fermer ou relocaliser** si le plan échoue (cf. analyse géographique).

        ---

        ### ❓ Questions à poser au responsable des ventes

        Notre analyse repose sur des coûts indirects agrégés. Pour calibrer
        les recommandations, nous avons besoin d'éclaircir :

        **Sur la structure des coûts**
        - **Part de coûts fixes vs variables** dans les coûts indirects par
          magasin ? Un magasin à 80 % fixe (loyer + permanents) ne réagit
          pas pareil à un à 50 % variable (intérim, énergie indexée).
        - **Élasticité des coûts à l'activité** : si le CA double, les
          coûts suivent à 10, 30 ou 60 % ? Détermine si "booster le volume"
          est rentable.
        - **Marges de manœuvre contractuelles** : durée des baux (sortie
          triennale ?), ancienneté des contrats énergie, conventions
          collectives. Sur Hyper Saveurs, −5 % = 85 k€ — **mais seulement
          si les contrats le permettent**.

        **Sur le potentiel commercial**
        - **Plafond physique** dans les Petits magasins (surface, caisses,
          parking) qui plafonnerait le gain de volume ?
        - **Mode d'exploitation** des 3 critiques (transit / aéroport /
          gare) qui justifierait des coûts plus élevés ? Si oui, le format
          est-il **adapté à la zone** ?

        **Sur la stratégie**
        - **Synergies inter-magasins** (centrale d'achats, logistique
          mutualisée) en place ou à mettre en place ?
        - **Plan d'animation de la carte de fidélité** structuré et piloté,
          ou résultat subi ?

        ---

        ### 🌍 Cas géographiques secondaires (non critiques)

        - **Lyon** : Maxi Délices coexiste avec Tosco (−130 k€) et
          AéroBoutique (+36 k€). Tester si Tosco subit la concurrence
          de Maxi Délices.
        - **Nantes** : Le Comptoir Local rentable face à Joyeux marché
          et Départ Express légèrement déficitaires → possible
          cannibalisation à vérifier.
        - **Toulouse** : 3 magasins tous rentables → **modèle de
          répartition à étudier** pour le reste du réseau.

        ---

        ### 📊 KPI à suivre

        **Pilotage de la rentabilité**
        - **Revenu net par magasin** = Marge − Coûts indirects. Suivi
          mensuel cumulé sur les 8 magasins en plan de redressement.
        - **Revenu net global Omnivend** : passage du négatif au positif
          comme indicateur ultime de réussite.

        **Pilotage commercial**
        - **Nombre de transactions / jour** (flux client) et **évolution
          mensuelle vs N-1** (neutralise la saisonnalité).

        **Pilotage fidélité**
        - **Taux de pénétration carte de fidélité** = % de transactions
          avec carte.
        - **Fréquence d'achat des porteurs de carte** = mesure de la
          rétention dans le temps.

        **Pilotage segmentation**
        - **Saisonnalité résiduelle** = écart-type des ventes mensuelles
          / moyenne → dépendance aux pics ponctuels.

        ---

        ### 🔍 Pistes d'approfondissement

        - **Analyse géomarketing** : croiser avec données INSEE
          (population, revenus médians, concurrence) sur les zones de
          chalandise pour calibrer les objectifs réalistes par magasin.
        - **Qualité de la donnée** : pérenniser les contrôles automatiques
          (anomalies de dates, montants aberrants, référentiels) découverts
          lors de cette mission.
        """
    )
