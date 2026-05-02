import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
import json
import os
import re

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Luggage Brand Intelligence",
    page_icon="🧳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border: 1px solid #2d3250;
        border-radius: 12px;
        padding: 18px 22px;
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-card .label { color: #8b8fa8; font-size: 13px; font-weight: 500; letter-spacing: 0.5px; }
    .metric-card .value { color: #e2e8f0; font-size: 28px; font-weight: 700; margin: 6px 0 2px; }
    .metric-card .delta { font-size: 12px; color: #7dd87d; }
    .insight-card {
        background: linear-gradient(135deg, #1a1f35, #1e2445);
        border-left: 3px solid #6366f1;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
        color: #c8cce8;
        font-size: 14px;
        line-height: 1.6;
    }
    .brand-tag {
        display: inline-block;
        background: #2d3250;
        color: #818cf8;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px;
    }
    .section-header {
        color: #e2e8f0;
        font-size: 20px;
        font-weight: 700;
        border-bottom: 2px solid #2d3250;
        padding-bottom: 8px;
        margin: 24px 0 16px;
    }
    .stTabs [data-baseweb="tab"] { color: #8b8fa8; font-weight: 500; }
    .stTabs [aria-selected="true"] { color: #818cf8 !important; }
    div[data-testid="stSidebarContent"] { background: #0d1020; }
    .pos-pill {
        background: #1a3a2a; color: #6ee77a; border-radius: 20px;
        padding: 3px 10px; font-size: 12px; display: inline-block; margin: 2px;
    }
    .neg-pill {
        background: #3a1a1a; color: #f87171; border-radius: 20px;
        padding: 3px 10px; font-size: 12px; display: inline-block; margin: 2px;
    }
    .aspect-bar-label { font-size: 12px; color: #8b8fa8; }
</style>
""", unsafe_allow_html=True)

BRANDS = ["Safari", "Skybags", "American Tourister", "Aristocrat", "Nasher Miles"]
BRAND_COLORS = {
    "Safari": "#6366f1",
    "Skybags": "#f59e0b",
    "American Tourister": "#10b981",
    "Aristocrat": "#f43f5e",
    "Nasher Miles": "#38bdf8",
}
ASPECTS = ["wheels", "zipper", "handle", "material", "durability", "size"]

# ─── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(__file__)
    products = pd.read_csv(os.path.join(base, "luggage_cleaned.csv"))
    reviews = pd.read_csv(os.path.join(base, "reviews_dataset.csv"))

    # Assign brand from product name
    def assign_brand(name):
        for b in BRANDS:
            if b.lower() in str(name).lower():
                return b
        return "Unknown"

    reviews["Brand"] = reviews["Product"].apply(assign_brand)
    reviews = reviews[reviews["Brand"] != "Unknown"]

    # Clean prices
    products["Price"] = pd.to_numeric(products["Price"], errors="coerce")
    products["MRP"] = pd.to_numeric(products["MRP"], errors="coerce")
    products["Discount_Percent"] = pd.to_numeric(products["Discount_Percent"], errors="coerce")
    products["Rating"] = pd.to_numeric(products["Rating"], errors="coerce")
    products["Review_Count"] = pd.to_numeric(products["Review_Count"], errors="coerce")

    return products, reviews

products_df, reviews_df = load_data()

# ─── Groq Client ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_groq_client(api_key):
    return Groq(api_key=api_key)

def call_groq(client, prompt, system="You are a market intelligence analyst. Respond only in valid JSON.", max_tokens=1500):
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return json.dumps({"error": str(e)})

def safe_json(text):
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except:
        return {}

# ─── Sentiment Analysis ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def analyze_brand_sentiment(_client_key, brand):
    client = get_groq_client(_client_key)
    brand_reviews = reviews_df[reviews_df["Brand"] == brand]["Review_Text"].dropna().tolist()
    if not brand_reviews:
        return {}

    sample = brand_reviews[:40]
    combined = "\n---\n".join(sample)

    prompt = f"""Analyze these {len(sample)} customer reviews for {brand} luggage on Amazon India.
Return ONLY a JSON object with this exact structure:
{{
  "sentiment_score": <float 0-10>,
  "positive_themes": [<5 short phrases>],
  "negative_themes": [<5 short phrases>],
  "summary": "<2 sentence brand summary>",
  "aspect_scores": {{
    "wheels": <0-10>,
    "zipper": <0-10>,
    "handle": <0-10>,
    "material": <0-10>,
    "durability": <0-10>,
    "size": <0-10>
  }},
  "value_for_money": <0-10>,
  "recommendation_rate": <percentage 0-100>
}}

Reviews:
{combined[:4000]}"""

    raw = call_groq(client, prompt, max_tokens=600)
    result = safe_json(raw)
    result["brand"] = brand
    result["review_count"] = len(brand_reviews)
    return result

@st.cache_data(show_spinner=False)
def analyze_product_sentiment(_client_key, product_name):
    client = get_groq_client(_client_key)
    prod_reviews = reviews_df[reviews_df["Product"].str.contains(product_name[:40], na=False)]["Review_Text"].dropna().tolist()
    if not prod_reviews:
        return {}

    sample = prod_reviews[:20]
    combined = "\n---\n".join(sample)

    prompt = f"""Analyze these customer reviews for this luggage product.
Return ONLY a JSON object:
{{
  "sentiment_score": <float 0-10>,
  "summary": "<2 sentence product summary>",
  "top_praises": [<3 short phrases>],
  "top_complaints": [<3 short phrases>],
  "aspect_scores": {{
    "wheels": <0-10>, "zipper": <0-10>, "handle": <0-10>,
    "material": <0-10>, "durability": <0-10>, "size": <0-10>
  }}
}}

Reviews:
{combined[:3000]}"""

    return safe_json(call_groq(client, prompt, max_tokens=400))

@st.cache_data(show_spinner=False)
def generate_agent_insights(_client_key, brand_summaries):
    client = get_groq_client(_client_key)
    summary_text = json.dumps(brand_summaries, indent=2)

    prompt = f"""You are a senior market intelligence analyst. Based on this competitive data for luggage brands on Amazon India, generate exactly 5 non-obvious, actionable insights that a brand manager would find valuable.

Data:
{summary_text[:4000]}

Return ONLY a JSON object:
{{
  "insights": [
    {{
      "title": "<short bold title>",
      "insight": "<2-3 sentence insight explaining what the data reveals and why it matters>",
      "brand_relevance": "<brand name most affected or 'All Brands'>",
      "type": "<Opportunity|Risk|Anomaly|Strategy>"
    }}
  ]
}}"""

    return safe_json(call_groq(client, prompt, max_tokens=1000))

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧳 Luggage Intel")
    st.markdown("---")
    groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    st.caption("Get your free key at [console.groq.com](https://console.groq.com)")
    st.markdown("---")

    st.markdown("### Filters")
    selected_brands = st.multiselect("Brands", BRANDS, default=BRANDS)
    price_range = st.slider(
        "Price Range (₹)",
        min_value=int(products_df["Price"].min()),
        max_value=int(products_df["Price"].max()),
        value=(int(products_df["Price"].min()), int(products_df["Price"].max())),
    )
    min_rating = st.slider("Minimum Rating", 1.0, 5.0, 1.0, 0.1)
    selected_sizes = st.multiselect(
        "Luggage Size",
        options=sorted(products_df["Size"].dropna().unique().tolist()),
        default=[],
    )

    st.markdown("---")
    st.markdown("### About")
    st.caption("Dashboard built for Moonshot AI Agent Internship Assignment. Data from Amazon India.")

# ─── Filter Data ──────────────────────────────────────────────────────────────
filtered_products = products_df[
    (products_df["Brand"].isin(selected_brands)) &
    (products_df["Price"].between(price_range[0], price_range[1])) &
    (products_df["Rating"] >= min_rating)
]
if selected_sizes:
    filtered_products = filtered_products[filtered_products["Size"].isin(selected_sizes)]

filtered_reviews = reviews_df[reviews_df["Brand"].isin(selected_brands)]

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background: linear-gradient(135deg, #1a1f35 0%, #252840 100%);
     border-radius: 16px; padding: 28px 32px; margin-bottom: 24px;
     border: 1px solid #2d3250;'>
  <h1 style='color: #e2e8f0; margin: 0 0 6px; font-size: 32px;'>
    🧳 Luggage Brand Intelligence
  </h1>
  <p style='color: #8b8fa8; margin: 0; font-size: 15px;'>
    Competitive analysis of luggage brands on Amazon India — Pricing · Sentiment · Insights
  </p>
</div>
""", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "⚔️ Brand Comparison", "🔍 Product Drilldown", "🤖 Agent Insights"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    # KPI Row
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, "Brands Tracked", len(selected_brands), ""),
        (c2, "Products Analyzed", len(filtered_products), ""),
        (c3, "Reviews Analyzed", len(filtered_reviews), ""),
        (c4, "Avg. Selling Price", f"₹{filtered_products['Price'].mean():,.0f}", ""),
        (c5, "Avg. Discount", f"{filtered_products['Discount_Percent'].mean():.1f}%", ""),
    ]
    for col, label, val, delta in kpis:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
              <div class='label'>{label}</div>
              <div class='value'>{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 💰 Average Price by Brand")
        avg_price = filtered_products.groupby("Brand")["Price"].mean().reset_index().sort_values("Price", ascending=True)
        fig = px.bar(
            avg_price, x="Price", y="Brand", orientation="h",
            color="Brand", color_discrete_map=BRAND_COLORS,
            text=avg_price["Price"].apply(lambda x: f"₹{x:,.0f}"),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c8cce8", showlegend=False,
            xaxis=dict(gridcolor="#2d3250", title="Avg. Price (₹)"),
            yaxis=dict(title=""),
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("#### 🏷️ Discount Distribution by Brand")
        disc_data = filtered_products.dropna(subset=["Discount_Percent"])
        fig2 = px.box(
            disc_data, x="Brand", y="Discount_Percent",
            color="Brand", color_discrete_map=BRAND_COLORS,
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c8cce8", showlegend=False,
            xaxis=dict(title="", gridcolor="#2d3250"),
            yaxis=dict(title="Discount %", gridcolor="#2d3250"),
            height=320,
        )
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### ⭐ Rating vs Price (Bubble = Review Count)")
        scatter_data = filtered_products.dropna(subset=["Rating", "Price", "Review_Count"])
        fig3 = px.scatter(
            scatter_data, x="Price", y="Rating", size="Review_Count",
            color="Brand", color_discrete_map=BRAND_COLORS,
            hover_name="Title", size_max=40,
        )
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c8cce8",
            xaxis=dict(gridcolor="#2d3250", title="Price (₹)"),
            yaxis=dict(gridcolor="#2d3250", title="Rating"),
            height=350, legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("#### 📦 Product Count by Size & Brand")
        size_brand = filtered_products.groupby(["Brand", "Size"]).size().reset_index(name="Count")
        fig4 = px.bar(
            size_brand, x="Size", y="Count", color="Brand",
            color_discrete_map=BRAND_COLORS, barmode="group",
        )
        fig4.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c8cce8",
            xaxis=dict(title="", gridcolor="#2d3250"),
            yaxis=dict(title="Products", gridcolor="#2d3250"),
            height=350, legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig4, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BRAND COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    if not groq_key:
        st.info("🔑 Enter your Groq API key in the sidebar to enable AI-powered sentiment analysis.")

    # Brand metrics table
    st.markdown("#### 📋 Brand Scorecard")
    brand_stats = []
    for brand in selected_brands:
        bdf = filtered_products[filtered_products["Brand"] == brand]
        stat = {
            "Brand": brand,
            "Products": len(bdf),
            "Avg Price (₹)": f"₹{bdf['Price'].mean():,.0f}" if len(bdf) else "—",
            "Avg Discount %": f"{bdf['Discount_Percent'].mean():.1f}%" if not bdf['Discount_Percent'].isna().all() else "—",
            "Avg Rating": f"{bdf['Rating'].mean():.2f} ⭐" if len(bdf) else "—",
            "Total Reviews": f"{bdf['Review_Count'].sum():,}" if len(bdf) else "—",
            "Price Range": f"₹{bdf['Price'].min():,.0f} – ₹{bdf['Price'].max():,.0f}" if len(bdf) else "—",
        }
        brand_stats.append(stat)

    st.dataframe(
        pd.DataFrame(brand_stats),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    # Radar Chart
    st.markdown("#### 🕸️ Multi-Metric Radar Comparison")
    radar_metrics = ["Avg Price (norm)", "Avg Discount", "Avg Rating", "Review Volume (norm)"]
    radar_df = filtered_products.groupby("Brand").agg(
        avg_price=("Price", "mean"),
        avg_discount=("Discount_Percent", "mean"),
        avg_rating=("Rating", "mean"),
        total_reviews=("Review_Count", "sum"),
    ).reset_index()

    # Normalize 0-10
    for col in ["avg_price", "avg_discount", "avg_rating", "total_reviews"]:
        mn, mx = radar_df[col].min(), radar_df[col].max()
        radar_df[col + "_n"] = ((radar_df[col] - mn) / (mx - mn + 1e-9)) * 10

    fig_radar = go.Figure()
    categories = ["Avg Price", "Avg Discount", "Avg Rating", "Review Volume"]
    for _, row in radar_df.iterrows():
        if row["Brand"] not in selected_brands:
            continue
        vals = [row["avg_price_n"], row["avg_discount_n"], row["avg_rating_n"], row["total_reviews_n"]]
        vals += [vals[0]]  # close polygon
        fig_radar.add_trace(go.Scatterpolar(
            r=vals,
            theta=categories + [categories[0]],
            fill="toself",
            name=row["Brand"],
            line_color=BRAND_COLORS.get(row["Brand"], "#818cf8"),
            opacity=0.7,
        ))

    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 10], gridcolor="#2d3250", color="#8b8fa8"),
            angularaxis=dict(gridcolor="#2d3250", color="#c8cce8"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#c8cce8",
        height=420,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # AI Sentiment Comparison
    if groq_key:
        st.markdown("#### 🧠 AI Sentiment Analysis by Brand")
        sentiment_results = {}
        cols_s = st.columns(len(selected_brands))
        for i, brand in enumerate(selected_brands):
            with st.spinner(f"Analyzing {brand}..."):
                result = analyze_brand_sentiment(groq_key, brand)
                sentiment_results[brand] = result

        # Sentiment scores bar
        if sentiment_results:
            scores_data = [
                {"Brand": b, "Sentiment Score": v.get("sentiment_score", 0),
                 "Value for Money": v.get("value_for_money", 0)}
                for b, v in sentiment_results.items() if v
            ]
            if scores_data:
                sdf = pd.DataFrame(scores_data)
                fig_sent = px.bar(
                    sdf.melt(id_vars="Brand", var_name="Metric", value_name="Score"),
                    x="Brand", y="Score", color="Metric", barmode="group",
                    color_discrete_sequence=["#6366f1", "#f59e0b"],
                )
                fig_sent.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#c8cce8",
                    xaxis=dict(gridcolor="#2d3250", title=""),
                    yaxis=dict(gridcolor="#2d3250", title="Score (0–10)", range=[0, 11]),
                    height=320, legend=dict(bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig_sent, use_container_width=True)

            # Aspect Radar
            st.markdown("#### 🔩 Aspect-Level Sentiment (Wheels, Zipper, Handle, Material, Durability, Size)")
            fig_aspect = go.Figure()
            aspect_labels = ["Wheels", "Zipper", "Handle", "Material", "Durability", "Size"]
            for brand, res in sentiment_results.items():
                asp = res.get("aspect_scores", {})
                if asp:
                    vals = [asp.get(a.lower(), 5) for a in aspect_labels] + [asp.get("wheels", 5)]
                    fig_aspect.add_trace(go.Scatterpolar(
                        r=vals,
                        theta=aspect_labels + [aspect_labels[0]],
                        fill="toself",
                        name=brand,
                        line_color=BRAND_COLORS.get(brand, "#818cf8"),
                        opacity=0.65,
                    ))
            fig_aspect.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 10], gridcolor="#2d3250", color="#8b8fa8"),
                    angularaxis=dict(gridcolor="#2d3250", color="#c8cce8"),
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#c8cce8",
                height=420,
                legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_aspect, use_container_width=True)

            # Pros & Cons per brand
            st.markdown("#### ✅ Top Pros & ❌ Top Cons by Brand")
            pros_cols = st.columns(len(selected_brands))
            for i, brand in enumerate(selected_brands):
                res = sentiment_results.get(brand, {})
                with pros_cols[i]:
                    st.markdown(f"**{brand}**")
                    score = res.get("sentiment_score", "—")
                    st.markdown(f"Sentiment: **{score}/10**")
                    for p in res.get("positive_themes", []):
                        st.markdown(f"<span class='pos-pill'>✓ {p}</span>", unsafe_allow_html=True)
                    for n in res.get("negative_themes", []):
                        st.markdown(f"<span class='neg-pill'>✗ {n}</span>", unsafe_allow_html=True)
                    summary = res.get("summary", "")
                    if summary:
                        st.caption(summary)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PRODUCT DRILLDOWN
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### 🔍 Product Drilldown")

    brand_filter = st.selectbox("Select Brand", selected_brands, key="drilldown_brand")
    brand_products = filtered_products[filtered_products["Brand"] == brand_filter]

    product_names = brand_products["Title"].str[:80].tolist()
    selected_product_title = st.selectbox("Select Product", product_names, key="drilldown_product")

    if selected_product_title:
        full_product = brand_products[brand_products["Title"].str.startswith(selected_product_title[:40])].iloc[0]

        # Product Info Cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class='metric-card'>
                <div class='label'>Selling Price</div>
                <div class='value'>₹{full_product['Price']:,.0f}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            mrp = full_product['MRP'] if not pd.isna(full_product['MRP']) else full_product['Price']
            st.markdown(f"""<div class='metric-card'>
                <div class='label'>MRP</div>
                <div class='value'>₹{mrp:,.0f}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            disc = full_product['Discount_Percent'] if not pd.isna(full_product['Discount_Percent']) else 0
            st.markdown(f"""<div class='metric-card'>
                <div class='label'>Discount</div>
                <div class='value'>{disc:.0f}%</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class='metric-card'>
                <div class='label'>Rating</div>
                <div class='value'>{full_product['Rating']:.1f} ⭐</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"**Product:** {full_product['Title']}")
        st.markdown(f"**Size:** {full_product['Size']}  |  **Reviews:** {full_product['Review_Count']:,}")
        if pd.notna(full_product.get("URL")):
            st.markdown(f"[View on Amazon ↗]({full_product['URL']})")

        # Reviews for this product
        prod_reviews = reviews_df[reviews_df["Product"].str.contains(
            str(full_product["Title"])[:40], na=False, regex=False
        )]
        st.markdown(f"**{len(prod_reviews)} scraped reviews available**")

        if len(prod_reviews) > 0:
            # Aspect scores
            if groq_key:
                with st.spinner("Generating AI review synthesis..."):
                    prod_analysis = analyze_product_sentiment(groq_key, full_product["Title"])

                if prod_analysis:
                    col_l, col_r = st.columns(2)
                    with col_l:
                        st.markdown("**Review Synthesis**")
                        st.info(prod_analysis.get("summary", ""))
                        st.markdown("**Top Praises:**")
                        for p in prod_analysis.get("top_praises", []):
                            st.markdown(f"<span class='pos-pill'>✓ {p}</span>", unsafe_allow_html=True)
                        st.markdown("**Top Complaints:**")
                        for c_ in prod_analysis.get("top_complaints", []):
                            st.markdown(f"<span class='neg-pill'>✗ {c_}</span>", unsafe_allow_html=True)

                    with col_r:
                        asp = prod_analysis.get("aspect_scores", {})
                        if asp:
                            aspect_df = pd.DataFrame([
                                {"Aspect": a.capitalize(), "Score": asp.get(a, 5)}
                                for a in ASPECTS
                            ])
                            fig_asp = px.bar(
                                aspect_df, x="Score", y="Aspect", orientation="h",
                                color="Score", color_continuous_scale="RdYlGn",
                                range_color=[0, 10], text="Score",
                            )
                            fig_asp.update_traces(texttemplate="%{text:.1f}", textposition="outside")
                            fig_asp.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font_color="#c8cce8", showlegend=False,
                                xaxis=dict(gridcolor="#2d3250", range=[0, 12], title="Score"),
                                yaxis=dict(title=""),
                                height=280, coloraxis_showscale=False,
                            )
                            st.plotly_chart(fig_asp, use_container_width=True)

            # Raw Reviews
            st.markdown("#### 📝 Customer Reviews")
            st.dataframe(
                prod_reviews[["Reviewer", "Review_Title", "Review_Text", "Review_Date"]].reset_index(drop=True),
                use_container_width=True,
                height=300,
            )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — AGENT INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("#### 🤖 Agent Insights")
    st.markdown("AI-generated non-obvious conclusions from the competitive data.")

    if not groq_key:
        st.info("🔑 Enter your Groq API key in the sidebar to generate Agent Insights.")
    else:
        if st.button("🚀 Generate Agent Insights", type="primary"):
            with st.spinner("Analyzing all brand data and generating insights..."):
                # Collect brand summaries
                brand_summaries = {}
                for brand in selected_brands:
                    bdf = filtered_products[filtered_products["Brand"] == brand]
                    brand_summaries[brand] = {
                        "avg_price": round(bdf["Price"].mean(), 0),
                        "avg_discount": round(bdf["Discount_Percent"].mean(), 1),
                        "avg_rating": round(bdf["Rating"].mean(), 2),
                        "total_review_count": int(bdf["Review_Count"].sum()),
                        "product_count": len(bdf),
                        "price_range": [round(bdf["Price"].min()), round(bdf["Price"].max())],
                        "sizes_available": bdf["Size"].dropna().unique().tolist(),
                    }
                    # Add sentiment
                    sent = analyze_brand_sentiment(groq_key, brand)
                    if sent:
                        brand_summaries[brand]["sentiment_score"] = sent.get("sentiment_score")
                        brand_summaries[brand]["value_for_money"] = sent.get("value_for_money")
                        brand_summaries[brand]["positive_themes"] = sent.get("positive_themes", [])
                        brand_summaries[brand]["negative_themes"] = sent.get("negative_themes", [])

                insights_data = generate_agent_insights(groq_key, brand_summaries)

            insights = insights_data.get("insights", [])
            if insights:
                type_colors = {
                    "Opportunity": "#10b981",
                    "Risk": "#f43f5e",
                    "Anomaly": "#f59e0b",
                    "Strategy": "#6366f1",
                }
                for i, ins in enumerate(insights, 1):
                    itype = ins.get("type", "Insight")
                    color = type_colors.get(itype, "#818cf8")
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #1a1f35, #1e2445);
                         border-left: 4px solid {color}; border-radius: 10px;
                         padding: 18px 22px; margin: 12px 0;'>
                      <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;'>
                        <span style='color:{color}; font-weight:700; font-size:16px;'>{i}. {ins.get("title", "")}</span>
                        <span style='background:{color}22; color:{color}; border-radius:20px;
                              padding:3px 12px; font-size:12px; font-weight:600;'>{itype}</span>
                      </div>
                      <p style='color:#c8cce8; margin:0 0 8px; line-height:1.7;'>{ins.get("insight", "")}</p>
                      <span style='background:#2d3250; color:#818cf8; border-radius:20px;
                            padding:3px 12px; font-size:12px;'>📌 {ins.get("brand_relevance", "")}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("Could not generate insights. Please try again.")

        # Always show pricing position map
        st.markdown("---")
        st.markdown("#### 🗺️ Price vs Sentiment Positioning Map")
        if groq_key:
            pos_data = []
            for brand in selected_brands:
                bdf = filtered_products[filtered_products["Brand"] == brand]
                sent = analyze_brand_sentiment(groq_key, brand)
                pos_data.append({
                    "Brand": brand,
                    "Avg Price": bdf["Price"].mean(),
                    "Sentiment": sent.get("sentiment_score", 5) if sent else 5,
                    "Reviews": bdf["Review_Count"].sum(),
                    "Avg Discount": bdf["Discount_Percent"].mean(),
                })
            pos_df = pd.DataFrame(pos_data)
            fig_pos = px.scatter(
                pos_df, x="Avg Price", y="Sentiment",
                size="Reviews", color="Brand",
                color_discrete_map=BRAND_COLORS,
                hover_name="Brand", size_max=60,
                text="Brand",
            )
            fig_pos.update_traces(textposition="top center")
            # Quadrant lines
            mid_price = pos_df["Avg Price"].mean()
            mid_sent = 5
            fig_pos.add_hline(y=mid_sent, line_dash="dash", line_color="#2d3250")
            fig_pos.add_vline(x=mid_price, line_dash="dash", line_color="#2d3250")
            # Quadrant labels
            fig_pos.add_annotation(x=mid_price * 1.3, y=9, text="Premium Leader", font_color="#6ee77a", showarrow=False)
            fig_pos.add_annotation(x=mid_price * 0.5, y=9, text="Value Champion", font_color="#38bdf8", showarrow=False)
            fig_pos.add_annotation(x=mid_price * 1.3, y=2, text="Overpriced", font_color="#f87171", showarrow=False)
            fig_pos.add_annotation(x=mid_price * 0.5, y=2, text="Low Value", font_color="#f59e0b", showarrow=False)
            fig_pos.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,17,23,0.8)",
                font_color="#c8cce8",
                xaxis=dict(gridcolor="#2d3250", title="Average Price (₹)"),
                yaxis=dict(gridcolor="#2d3250", title="Sentiment Score (0–10)", range=[0, 11]),
                height=480, legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_pos, use_container_width=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; color:#4a4f6a; font-size:12px; margin-top:40px; padding:16px;
     border-top: 1px solid #1e2130;'>
  Moonshot AI Agent Internship · Data sourced from Amazon India · Powered by Groq LLaMA3
</div>
""", unsafe_allow_html=True)
