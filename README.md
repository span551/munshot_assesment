# 🧳 Luggage Brand Intelligence Dashboard

Competitive intelligence dashboard for luggage brands on Amazon India.
Built for the **Moonshot AI Agent Internship Assignment**.

## Live Demo
Deploy on Streamlit Community Cloud — instructions below.

## Features

| Feature | Details |
|---|---|
| Brands | Safari, Skybags, American Tourister, Aristocrat, Nasher Miles |
| Products | 45 products scraped from Amazon India |
| Reviews | 360+ customer reviews |
| AI Engine | Groq (LLaMA3-8B) for sentiment & insights |

### Dashboard Views
1. **Overview** — KPIs, pricing charts, rating scatter, size breakdown
2. **Brand Comparison** — Scorecard table, radar chart, aspect-level sentiment, pros/cons
3. **Product Drilldown** — Per-product sentiment, aspect scores, raw reviews
4. **Agent Insights** — 5 non-obvious AI-generated conclusions + Price vs Sentiment positioning map

### Bonus Features Implemented
- ✅ Aspect-level sentiment (wheels, zipper, handle, material, durability, size)
- ✅ Value-for-money analysis
- ✅ Price vs Sentiment positioning quadrant map
- ✅ Agent Insights with Opportunity/Risk/Anomaly/Strategy tags
- ✅ Brand color-coded visualizations throughout

## Setup

### Prerequisites
- Python 3.9+
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Local Setup
```bash
git clone <your-repo>
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

### Deploy to Streamlit Cloud
1. Push this repo to GitHub (include `app.py`, `requirements.txt`, `luggage_cleaned.csv`, `reviews_dataset.csv`)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file as `app.py`
5. Click Deploy

### Using the Dashboard
1. Open the app
2. Enter your Groq API key in the sidebar (never committed to code)
3. Use sidebar filters to slice by brand, price, rating, size
4. Navigate tabs for different views
5. Click "Generate Agent Insights" for AI conclusions

## Dataset

### `luggage_cleaned.csv`
| Column | Description |
|---|---|
| Brand | Brand name |
| Title | Full product title |
| Price | Selling price (₹) |
| MRP | List price (₹) |
| Discount_Percent | Discount % |
| Rating | Star rating (1–5) |
| Review_Count | Number of reviews on Amazon |
| Size | Cabin / Medium / Large |
| URL | Amazon product URL |

### `reviews_dataset.csv`
| Column | Description |
|---|---|
| Product | Product name |
| Reviewer | Reviewer name |
| Review_Title | Review headline |
| Review_Text | Full review text |
| Review_Rating | Star rating |
| Review_Date | Date of review |
| Product_URL | Amazon URL |

## Sentiment Methodology
- Model: Groq `llama3-8b-8192`
- Reviews are batched (up to 40 per brand, 20 per product)
- Output: sentiment score (0–10), aspect scores, themes, summaries
- Results are cached in Streamlit's session cache to minimize API calls
- Prompts are structured to return JSON only for reliable parsing

## Limitations
- Skybags has fewer scraped products (5 vs 10 for other brands)
- Review ratings were not available in the scraped data (NaN) — sentiment is inferred from review text only
- Groq free tier has rate limits; heavy usage may trigger delays

## Stack
- **Frontend**: Streamlit + Plotly
- **AI**: Groq (LLaMA3-8B-8192)
- **Data**: Pandas
- **Scraping**: Pre-scraped data provided

## Architecture
```
app.py
├── Data Loading (cached)
│   ├── luggage_cleaned.csv → products_df
│   └── reviews_dataset.csv → reviews_df (brand assigned from product name)
├── Sidebar (filters + API key)
├── Tab 1: Overview (charts, KPIs)
├── Tab 2: Brand Comparison (table + radar + Groq sentiment)
├── Tab 3: Product Drilldown (product selector + Groq per-product analysis)
└── Tab 4: Agent Insights (Groq brand summaries → 5 insights + positioning map)
```
