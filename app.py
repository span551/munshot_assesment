import streamlit as st
import pandas as pd
import altair as alt
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
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"] { background: #0d1020; }
.metric-card {
    background: linear-gradient(135deg, #1e2130, #252840);
    border: 1px solid #2d3250; border-radius: 12px;
    padding: 18px 22px; text-align: center; margin-bottom: 12px;
}
.metric-card .label { color: #8b8fa8; font-size: 13px; font-weight: 500; }
.metric-card .value { color: #e2e8f0; font-size: 26px; font-weight: 700; margin: 6px 0 2px; }
.pos-pill {
    background: #1a3a2a; color: #6ee77a; border-radius: 20px;
    padding: 3px 10px; font-size: 12px; display: inline-block; margin: 2px;
}
.neg-pill {
    background: #3a1a1a; color: #f87171; border-radius: 20px;
    padding: 3px 10px; font-size: 12px; display: inline-block; margin: 2px;
}
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

color_scale = alt.Scale(
    domain=BRANDS,
    range=[BRAND_COLORS[b] for b in BRANDS],
)

# ─── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    products = pd.read_csv(os.path.join(base, "luggage_cleaned.csv"))
    reviews = pd.read_csv(os.path.join(base, "reviews_dataset.csv"))

    def assign_brand(name):
        for b in BRANDS:
            if b.lower() in str(name).lower():
                return b
        return "Unknown"

    reviews["Brand"] = reviews["Product"].apply(assign_brand)
    reviews = reviews[reviews["Brand"] != "Unknown"]
    for col in ["Price", "MRP", "Discount_Percent", "Rating", "Review_Count"]:
        products[col] = pd.to_numeric(products[col], errors="coerce")
    return products, reviews

products_df, reviews_df = load_data()

# ─── Groq Helpers ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_groq_client(api_key):
    return Groq(api_key=api_key)

def call_groq(client, prompt, max_tokens=1200):
    try:
        resp = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a market intelligence analyst. Respond ONLY in valid JSON with no markdown fences, no explanation, no preamble."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return json.dumps({"error": str(e)})

def safe_json(text):
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        return {}

@st.cache_data(show_spinner=False)
def analyze_brand_sentiment(_key, brand):
    client = get_groq_client(_key)
    texts = reviews_df[reviews_df["Brand"] == brand]["Review_Text"].dropna().tolist()[:40]
    if not texts:
        return {}
    combined = "\n---\n".join(texts)[:4000]
    prompt = f"""Analyze {len(texts)} Amazon India customer reviews for {brand} luggage.
Return ONLY this JSON (no extra keys, no markdown):
{{
  "sentiment_score": 7.5,
  "positive_themes": ["theme1","theme2","theme3","theme4","theme5"],
  "negative_themes": ["theme1","theme2","theme3","theme4","theme5"],
  "summary": "Two sentence summary here.",
  "aspect_scores": {{"wheels":7,"zipper":6,"handle":7,"material":8,"durability":7,"size":8}},
  "value_for_money": 7.5,
  "recommendation_rate": 75
}}
Reviews:
{combined}"""
    result = safe_json(call_groq(client, prompt, max_tokens=600))
    result["brand"] = brand
    return result

@st.cache_data(show_spinner=False)
def analyze_product_sentiment(_key, product_key):
    client = get_groq_client(_key)
    texts = reviews_df[reviews_df["Product"].str.contains(product_key, na=False, regex=False)]["Review_Text"].dropna().tolist()[:20]
    if not texts:
        return {}
    combined = "\n---\n".join(texts)[:3000]
    prompt = f"""Analyze these Amazon customer reviews for a luggage product.
Return ONLY this JSON (no markdown):
{{
  "sentiment_score": 7.5,
  "summary": "Two sentence summary.",
  "top_praises": ["praise1","praise2","praise3"],
  "top_complaints": ["complaint1","complaint2","complaint3"],
  "aspect_scores": {{"wheels":7,"zipper":6,"handle":7,"material":8,"durability":7,"size":8}}
}}
Reviews:
{combined}"""
    return safe_json(call_groq(client, prompt, max_tokens=400))

@st.cache_data(show_spinner=False)
def generate_agent_insights(_key, summaries_json):
    client = get_groq_client(_key)
    prompt = f"""You are a senior market intelligence analyst for luggage brands on Amazon India.
Based on this competitive data, generate exactly 5 non-obvious actionable insights.
Return ONLY this JSON (no markdown):
{{
  "insights": [
    {{"title":"Short Title","insight":"2-3 sentence insight.","brand_relevance":"Brand Name or All Brands","type":"Opportunity"}}
  ]
}}
Types must be one of: Opportunity, Risk, Anomaly, Strategy
Data:
{summaries_json[:4000]}"""
    return safe_json(call_groq(client, prompt, max_tokens=1000))

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧳 Luggage Intel")
    st.markdown("---")
    groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    st.caption("Free key at [console.groq.com](https://console.groq.com)")
    st.markdown("---")
    st.markdown("### Filters")
    selected_brands = st.multiselect("Brands", BRANDS, default=BRANDS)
    p_min = int(products_df["Price"].min())
    p_max = int(products_df["Price"].max())
    price_range = st.slider("Price Range (₹)", p_min, p_max, (p_min, p_max))
    min_rating = st.slider("Min Rating", 1.0, 5.0, 1.0, 0.1)
    sizes = sorted(products_df["Size"].dropna().unique().tolist())
    selected_sizes = st.multiselect("Luggage Size", sizes, default=[])
    st.markdown("---")
    st.caption("Moonshot AI Agent Internship · Amazon India")

# ─── Filter Data ──────────────────────────────────────────────────────────────
fp = products_df[
    products_df["Brand"].isin(selected_brands) &
    products_df["Price"].between(price_range[0], price_range[1]) &
    (products_df["Rating"] >= min_rating)
].copy()
if selected_sizes:
    fp = fp[fp["Size"].isin(selected_sizes)]
fr = reviews_df[reviews_df["Brand"].isin(selected_brands)].copy()

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#1a1f35,#252840);border-radius:16px;
padding:28px 32px;margin-bottom:24px;border:1px solid #2d3250;'>
<h1 style='color:#e2e8f0;margin:0 0 6px;font-size:30px;'>🧳 Luggage Brand Intelligence</h1>
<p style='color:#8b8fa8;margin:0;font-size:14px;'>Competitive analysis · Amazon India · Pricing · Sentiment · Insights</p>
</div>""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "⚔️ Brand Comparison", "🔍 Product Drilldown", "🤖 Agent Insights"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, "Brands", len(selected_brands)),
        (c2, "Products", len(fp)),
        (c3, "Reviews", len(fr)),
        (c4, "Avg Price", f"₹{fp['Price'].mean():,.0f}"),
        (c5, "Avg Discount", f"{fp['Discount_Percent'].mean():.1f}%"),
    ]
    for col, label, val in kpis:
        with col:
            st.markdown(f"<div class='metric-card'><div class='label'>{label}</div><div class='value'>{val}</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 💰 Avg Selling Price by Brand")
        avg_p = fp.groupby("Brand")["Price"].mean().reset_index().sort_values("Price")
        chart1 = alt.Chart(avg_p).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
            x=alt.X("Price:Q", title="Avg Price (₹)"),
            y=alt.Y("Brand:N", sort="-x", title=""),
            color=alt.Color("Brand:N", scale=color_scale, legend=None),
            tooltip=["Brand", alt.Tooltip("Price:Q", format=",.0f", title="Avg ₹")],
        ).properties(height=260, background="transparent")
        st.altair_chart(chart1, use_container_width=True)

    with col2:
        st.markdown("#### 🏷️ Avg Discount % by Brand")
        avg_d = fp.dropna(subset=["Discount_Percent"]).groupby("Brand")["Discount_Percent"].mean().reset_index().sort_values("Discount_Percent")
        chart2 = alt.Chart(avg_d).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
            x=alt.X("Discount_Percent:Q", title="Avg Discount %"),
            y=alt.Y("Brand:N", sort="-x", title=""),
            color=alt.Color("Brand:N", scale=color_scale, legend=None),
            tooltip=["Brand", alt.Tooltip("Discount_Percent:Q", format=".1f", title="Discount %")],
        ).properties(height=260, background="transparent")
        st.altair_chart(chart2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### ⭐ Rating vs Price (bubble = review count)")
        scatter = fp.dropna(subset=["Rating", "Price", "Review_Count"]).copy()
        chart3 = alt.Chart(scatter).mark_circle(opacity=0.8).encode(
            x=alt.X("Price:Q", title="Price (₹)"),
            y=alt.Y("Rating:Q", title="Rating", scale=alt.Scale(domain=[3, 5])),
            color=alt.Color("Brand:N", scale=color_scale),
            size=alt.Size("Review_Count:Q", scale=alt.Scale(range=[60, 600]), legend=None),
            tooltip=["Brand", alt.Tooltip("Price:Q", format=",.0f"), "Rating", alt.Tooltip("Review_Count:Q", format=",")],
        ).properties(height=300, background="transparent")
        st.altair_chart(chart3, use_container_width=True)

    with col4:
        st.markdown("#### 📦 Products by Size & Brand")
        size_brand = fp.groupby(["Brand", "Size"]).size().reset_index(name="Count")
        chart4 = alt.Chart(size_brand).mark_bar().encode(
            x=alt.X("Size:N", title=""),
            y=alt.Y("Count:Q", title="Products"),
            color=alt.Color("Brand:N", scale=color_scale),
            xOffset="Brand:N",
            tooltip=["Brand", "Size", "Count"],
        ).properties(height=300, background="transparent")
        st.altair_chart(chart4, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BRAND COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### 📋 Brand Scorecard")
    rows = []
    for b in selected_brands:
        bdf = fp[fp["Brand"] == b]
        rows.append({
            "Brand": b,
            "Products": len(bdf),
            "Avg Price": f"₹{bdf['Price'].mean():,.0f}" if len(bdf) else "—",
            "Avg Discount": f"{bdf['Discount_Percent'].mean():.1f}%" if not bdf['Discount_Percent'].isna().all() else "—",
            "Avg Rating": f"{bdf['Rating'].mean():.2f} ⭐" if len(bdf) else "—",
            "Total Reviews": f"{int(bdf['Review_Count'].sum()):,}" if len(bdf) else "—",
            "Price Range": f"₹{bdf['Price'].min():,.0f} – ₹{bdf['Price'].max():,.0f}" if len(bdf) else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 📊 Normalized Metrics Comparison (0–10 scale)")
    rdf = fp.groupby("Brand").agg(
        avg_price=("Price", "mean"),
        avg_discount=("Discount_Percent", "mean"),
        avg_rating=("Rating", "mean"),
        total_reviews=("Review_Count", "sum"),
    ).reset_index()
    for col in ["avg_price", "avg_discount", "avg_rating", "total_reviews"]:
        mn, mx = rdf[col].min(), rdf[col].max()
        rdf[col + "_n"] = ((rdf[col] - mn) / (mx - mn + 1e-9)) * 10

    melt = rdf[["Brand", "avg_price_n", "avg_discount_n", "avg_rating_n", "total_reviews_n"]].melt(id_vars="Brand", var_name="Metric", value_name="Score")
    melt["Metric"] = melt["Metric"].map({
        "avg_price_n": "Price", "avg_discount_n": "Discount",
        "avg_rating_n": "Rating", "total_reviews_n": "Review Volume",
    })
    bar_cmp = alt.Chart(melt).mark_bar().encode(
        x=alt.X("Metric:N", title=""),
        y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 10]), title="Score (0–10)"),
        color=alt.Color("Brand:N", scale=color_scale),
        xOffset="Brand:N",
        tooltip=["Brand", "Metric", alt.Tooltip("Score:Q", format=".1f")],
    ).properties(height=320, background="transparent")
    st.altair_chart(bar_cmp, use_container_width=True)

    if not groq_key:
        st.info("🔑 Enter Groq API key in the sidebar to enable AI sentiment analysis.")
    else:
        st.markdown("---")
        st.markdown("#### 🧠 AI Sentiment Scores")
        sentiment_results = {}
        for brand in selected_brands:
            with st.spinner(f"Analyzing {brand}..."):
                sentiment_results[brand] = analyze_brand_sentiment(groq_key, brand)

        sent_rows = []
        for b, v in sentiment_results.items():
            if v:
                sent_rows.append({"Brand": b, "Metric": "Sentiment", "Score": v.get("sentiment_score", 0)})
                sent_rows.append({"Brand": b, "Metric": "Value for Money", "Score": v.get("value_for_money", 0)})
        if sent_rows:
            sdf = pd.DataFrame(sent_rows)
            sc = alt.Chart(sdf).mark_bar().encode(
                x=alt.X("Brand:N", title=""),
                y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 10]), title="Score (0–10)"),
                color=alt.Color("Metric:N", scale=alt.Scale(range=["#6366f1", "#f59e0b"])),
                xOffset="Metric:N",
                tooltip=["Brand", "Metric", alt.Tooltip("Score:Q", format=".1f")],
            ).properties(height=300, background="transparent")
            st.altair_chart(sc, use_container_width=True)

        st.markdown("#### 🔩 Aspect Scores — Wheels · Zipper · Handle · Material · Durability · Size")
        aspect_rows = []
        for b, v in sentiment_results.items():
            asp = v.get("aspect_scores", {}) if v else {}
            for a in ASPECTS:
                aspect_rows.append({"Brand": b, "Aspect": a.capitalize(), "Score": asp.get(a, 0)})
        if aspect_rows:
            adf = pd.DataFrame(aspect_rows)
            ac = alt.Chart(adf).mark_bar().encode(
                x=alt.X("Aspect:N", title=""),
                y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 10]), title="Score (0–10)"),
                color=alt.Color("Brand:N", scale=color_scale),
                xOffset="Brand:N",
                tooltip=["Brand", "Aspect", alt.Tooltip("Score:Q", format=".1f")],
            ).properties(height=320, background="transparent")
            st.altair_chart(ac, use_container_width=True)

        st.markdown("#### ✅ Pros & ❌ Cons by Brand")
        cols_pc = st.columns(len(selected_brands))
        for i, brand in enumerate(selected_brands):
            res = sentiment_results.get(brand, {})
            with cols_pc[i]:
                score = res.get("sentiment_score", "—") if res else "—"
                st.markdown(f"**{brand}**  \n`{score}/10`")
                for p in (res.get("positive_themes", []) if res else []):
                    st.markdown(f"<span class='pos-pill'>✓ {p}</span>", unsafe_allow_html=True)
                for n in (res.get("negative_themes", []) if res else []):
                    st.markdown(f"<span class='neg-pill'>✗ {n}</span>", unsafe_allow_html=True)
                summary = res.get("summary", "") if res else ""
                if summary:
                    st.caption(summary)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PRODUCT DRILLDOWN
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### 🔍 Product Drilldown")
    brand_sel = st.selectbox("Brand", selected_brands, key="dd_brand")
    brand_prods = fp[fp["Brand"] == brand_sel]
    prod_options = brand_prods["Title"].str[:80].tolist()

    if not prod_options:
        st.warning("No products match current filters.")
    else:
        sel_title = st.selectbox("Product", prod_options, key="dd_prod")
        row = brand_prods[brand_prods["Title"].str.startswith(sel_title[:40])].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        for col, label, val in [
            (c1, "Selling Price", f"₹{row['Price']:,.0f}"),
            (c2, "MRP", f"₹{row['MRP']:,.0f}" if not pd.isna(row['MRP']) else "—"),
            (c3, "Discount", f"{row['Discount_Percent']:.0f}%" if not pd.isna(row['Discount_Percent']) else "—"),
            (c4, "Rating", f"{row['Rating']:.1f} ⭐"),
        ]:
            with col:
                st.markdown(f"<div class='metric-card'><div class='label'>{label}</div><div class='value'>{val}</div></div>", unsafe_allow_html=True)

        st.markdown(f"**{row['Title']}**")
        st.markdown(f"Size: `{row['Size']}` | Amazon Reviews: `{int(row['Review_Count']):,}`")
        if pd.notna(row.get("URL")):
            st.markdown(f"[View on Amazon ↗]({row['URL']})")

        prod_reviews = reviews_df[reviews_df["Product"].str.contains(str(row["Title"])[:40], na=False, regex=False)]
        st.markdown(f"**{len(prod_reviews)} scraped reviews available**")

        if groq_key and len(prod_reviews) > 0:
            with st.spinner("AI synthesis..."):
                pa = analyze_product_sentiment(groq_key, str(row["Title"])[:40])
            if pa:
                cl, cr = st.columns(2)
                with cl:
                    st.markdown("**AI Summary**")
                    st.info(pa.get("summary", ""))
                    st.markdown("**Top Praises:**")
                    for p in pa.get("top_praises", []):
                        st.markdown(f"<span class='pos-pill'>✓ {p}</span>", unsafe_allow_html=True)
                    st.markdown("**Top Complaints:**")
                    for c_ in pa.get("top_complaints", []):
                        st.markdown(f"<span class='neg-pill'>✗ {c_}</span>", unsafe_allow_html=True)
                with cr:
                    asp = pa.get("aspect_scores", {})
                    if asp:
                        adf = pd.DataFrame([{"Aspect": a.capitalize(), "Score": asp.get(a, 0)} for a in ASPECTS])
                        ac = alt.Chart(adf).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
                            x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 10]), title="Score"),
                            y=alt.Y("Aspect:N", sort="-x", title=""),
                            color=alt.Color("Score:Q", scale=alt.Scale(scheme="redyellowgreen", domain=[0, 10]), legend=None),
                            tooltip=["Aspect", alt.Tooltip("Score:Q", format=".1f")],
                        ).properties(height=240, background="transparent")
                        st.altair_chart(ac, use_container_width=True)

        st.markdown("#### 📝 Customer Reviews")
        st.dataframe(
            prod_reviews[["Reviewer", "Review_Title", "Review_Text", "Review_Date"]].reset_index(drop=True),
            use_container_width=True, height=300,
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — AGENT INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("#### 🤖 Agent Insights")
    st.markdown("AI-generated non-obvious conclusions from the full competitive dataset.")

    if not groq_key:
        st.info("🔑 Enter your Groq API key in the sidebar to generate insights.")
    else:
        if st.button("🚀 Generate Agent Insights", type="primary"):
            with st.spinner("Gathering brand data and generating insights..."):
                summaries = {}
                for brand in selected_brands:
                    bdf = fp[fp["Brand"] == brand]
                    summaries[brand] = {
                        "avg_price": round(float(bdf["Price"].mean()), 0),
                        "avg_discount": round(float(bdf["Discount_Percent"].mean()), 1),
                        "avg_rating": round(float(bdf["Rating"].mean()), 2),
                        "total_reviews": int(bdf["Review_Count"].sum()),
                        "products": len(bdf),
                        "price_range": [round(float(bdf["Price"].min())), round(float(bdf["Price"].max()))],
                        "sizes": bdf["Size"].dropna().unique().tolist(),
                    }
                    sent = analyze_brand_sentiment(groq_key, brand)
                    if sent:
                        summaries[brand]["sentiment_score"] = sent.get("sentiment_score")
                        summaries[brand]["value_for_money"] = sent.get("value_for_money")
                        summaries[brand]["positive_themes"] = sent.get("positive_themes", [])
                        summaries[brand]["negative_themes"] = sent.get("negative_themes", [])
                        summaries[brand]["aspect_scores"] = sent.get("aspect_scores", {})

                result = generate_agent_insights(groq_key, json.dumps(summaries, indent=2))

            insights = result.get("insights", [])
            type_colors = {"Opportunity": "#10b981", "Risk": "#f43f5e", "Anomaly": "#f59e0b", "Strategy": "#6366f1"}
            if insights:
                for i, ins in enumerate(insights, 1):
                    t = ins.get("type", "Insight")
                    c = type_colors.get(t, "#818cf8")
                    st.markdown(f"""
                    <div style='background:linear-gradient(135deg,#1a1f35,#1e2445);
                         border-left:4px solid {c};border-radius:10px;
                         padding:16px 20px;margin:10px 0;'>
                      <div style='display:flex;justify-content:space-between;margin-bottom:8px;'>
                        <span style='color:{c};font-weight:700;font-size:15px;'>{i}. {ins.get("title","")}</span>
                        <span style='background:{c}22;color:{c};border-radius:20px;padding:2px 10px;font-size:11px;font-weight:600;'>{t}</span>
                      </div>
                      <p style='color:#c8cce8;margin:0 0 8px;line-height:1.7;font-size:14px;'>{ins.get("insight","")}</p>
                      <span style='background:#2d3250;color:#818cf8;border-radius:20px;padding:2px 10px;font-size:11px;'>📌 {ins.get("brand_relevance","")}</span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.error("Could not generate insights. Try again.")

        st.markdown("---")
        st.markdown("#### 🗺️ Price vs Sentiment Positioning Map")
        pos_rows = []
        for brand in selected_brands:
            bdf = fp[fp["Brand"] == brand]
            sent = analyze_brand_sentiment(groq_key, brand)
            pos_rows.append({
                "Brand": brand,
                "Avg Price": float(bdf["Price"].mean()),
                "Sentiment": float(sent.get("sentiment_score", 5)) if sent else 5.0,
                "Reviews": int(bdf["Review_Count"].sum()),
            })
        pos_df = pd.DataFrame(pos_rows)
        mid_price = float(pos_df["Avg Price"].mean())

        circles = alt.Chart(pos_df).mark_circle(opacity=0.85).encode(
            x=alt.X("Avg Price:Q", title="Average Price (₹)"),
            y=alt.Y("Sentiment:Q", title="Sentiment Score (0–10)", scale=alt.Scale(domain=[0, 10])),
            size=alt.Size("Reviews:Q", scale=alt.Scale(range=[300, 1400]), legend=None),
            color=alt.Color("Brand:N", scale=color_scale),
            tooltip=["Brand", alt.Tooltip("Avg Price:Q", format=",.0f"), alt.Tooltip("Sentiment:Q", format=".1f"), "Reviews"],
        )
        labels = alt.Chart(pos_df).mark_text(dy=-20, fontSize=12, fontWeight="bold").encode(
            x="Avg Price:Q",
            y="Sentiment:Q",
            text="Brand:N",
            color=alt.Color("Brand:N", scale=color_scale),
        )
        hline = alt.Chart(pd.DataFrame({"y": [5.0]})).mark_rule(color="#3d4270", strokeDash=[4, 4]).encode(y="y:Q")
        vline = alt.Chart(pd.DataFrame({"x": [mid_price]})).mark_rule(color="#3d4270", strokeDash=[4, 4]).encode(x="x:Q")
        chart_pos = (hline + vline + circles + labels).properties(height=420, background="transparent")
        st.altair_chart(chart_pos, use_container_width=True)
        st.caption("Bubble size = total review count. Dashed lines = avg price & neutral sentiment (5.0).")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;color:#4a4f6a;font-size:12px;margin-top:40px;
padding:16px;border-top:1px solid #1e2130;'>
Moonshot AI Agent Internship · Amazon India Data · Powered by Groq LLaMA3-8B
</div>""", unsafe_allow_html=True)
