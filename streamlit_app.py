import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

# ---------- Supabase REST config ----------
SUPABASE_URL = "https://qkonktvwcbfkehznugum.supabase.co"  # <- your project
SUPABASE_KEY = st.secrets['SUPABASE_KEY']      # service role or anon w/ RLS read policy
TABLE = "quote_history"
REST_ENDPOINT = f"{SUPABASE_URL}/rest/v1/{TABLE}"
HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
}

# ---------- Trade size buckets per underlying asset ----------
TRADE_SIZE_BUCKETS = {
    "USDC": {
        "small": [10, 100],
        "medium": [1_000, 10_000],
        "big": [50_000],
    },
    "ETH": {
        "small": [0.01, 0.1],
        "medium": [1, 5],
        "big": [10, 25],
    },
    "wstETH": {
        "small": [0.01, 0.1],
        "medium": [1, 5],
        "big": [10, 25],
    },
}

def classify_trade_size(amount: float, src_token: str) -> str:
    buckets = TRADE_SIZE_BUCKETS.get(src_token, {})
    for label, values in buckets.items():
        # exact membership, matching how the batch job queried
        if amount in values:
            return label
    return "other"

# ---------- Data fetch ----------
def fetch_quotes(chain: str | None, days: int, limit: int = 20000) -> pd.DataFrame:
    """
    Pulls raw rows from Supabase REST. We'll do all grouping client-side.
    """
    params = {"limit": limit, "order": "time.desc"}
    # Time filter
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    params["time"] = f"gte.{since}"
    # Chain filter (filter on src_chain; you can also OR dst_chain if needed)
    if chain:
        params["src_chain"] = f"eq.{chain}"
        params["dst_chain"] = f"eq.{chain}"
        # NOTE: Using both filters makes it intra-chain only. If you want either side to match, remove dst_chain or fetch-all-then-filter.

    resp = requests.get(REST_ENDPOINT, headers=HEADERS, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data)
    if not df.empty:
        # Ensure types
        df["input_amount"] = pd.to_numeric(df["input_amount"], errors="coerce")
        df["output_amount"] = pd.to_numeric(df["output_amount"], errors="coerce")
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    return df

# ---------- Stats ----------
def compute_provider_stats(
    df: pd.DataFrame,
    time_granularity: str = "H",  # "H" for hourly, "D" for daily
) -> pd.DataFrame:
    """
    Winner = provider with highest output_amount per group.
    Group = route + time bucket + input_amount.
    """
    if df.empty:
        return pd.DataFrame(columns=["provider","answered","win_rate","avg_loss_vs_winner","median_loss_vs_winner"])

    # Ensure correct dtypes
    df = df.dropna(subset=["output_amount","provider","time","input_amount"]).copy()
    df["output_amount"] = pd.to_numeric(df["output_amount"], errors="coerce")
    df["input_amount"] = pd.to_numeric(df["input_amount"], errors="coerce")
    df["time_bucket"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.floor(time_granularity)

    # Define grouping keys
    group_cols = ["src_chain","dst_chain","src_token","dst_token","input_amount","time_bucket"]

    # Per-group: compute winner baseline and diffs
    def per_group_calc(g: pd.DataFrame) -> pd.DataFrame:
        winner_output = g["output_amount"].max()
        g = g.copy()
        g["is_winner"] = (g["output_amount"] == winner_output).astype(int)
        g["loss_vs_winner"] = (
            g["output_amount"] / winner_output - 1.0
            if winner_output not in (None,0) else 0.0
        )
        return g

    df_calc = df.groupby(group_cols, dropna=False, as_index=False, group_keys=False).apply(per_group_calc)

    total_groups = df_calc[group_cols].drop_duplicates().shape[0]
    if total_groups == 0:
        return pd.DataFrame(columns=["provider","answered","win_rate","avg_loss_vs_winner","median_loss_vs_winner"])

    provider_agg = (
        df_calc.groupby("provider")
               .agg(
                   answered=("provider","count"),
                   wins=("is_winner","sum"),
                   avg_loss_vs_winner=("loss_vs_winner","mean"),
                   median_loss_vs_winner=("loss_vs_winner","median"),
               )
               .reset_index()
    )
    provider_agg["win_rate"] = provider_agg["wins"] / provider_agg["answered"]

    return provider_agg.drop(columns=["wins"]).sort_values(["win_rate","avg_loss_vs_winner"], ascending=[False,True])


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Quote Provider Performance", layout="wide")
st.title("Quote Provider Performance")

# Filters
col1, col2, col3 = st.columns([1,1,1])
with col1:
    chain_sel = st.selectbox("Chain", ["All","Ethereum","Arbitrum","Base"])
    chain = None if chain_sel == "All" else chain_sel
with col2:
    trade_size_sel = st.selectbox("Trade Size", ["All","small","medium","big"])
with col3:
    time_window_sel = st.selectbox("Time Window", ["Last 1 day","Last 7 days"])
    days = 1 if time_window_sel == "Last 1 day" else 7

compute = st.button("Compute")

if compute:
    df = fetch_quotes(chain=chain, days=days)

    if df.empty:
        st.warning("No quotes found for the selected filters.")
    else:
        # Classify trade size bin using underlying source token
        df["trade_size_bin"] = df.apply(
            lambda r: classify_trade_size(float(r["input_amount"]), str(r["src_token"])),
            axis=1
        )
        # Optional: exclude unexpected trade sizes
        if trade_size_sel != "All":
            df = df[df["trade_size_bin"] == trade_size_sel]

        # Derive hour bucket and compute stats
        df["time_hour"] = pd.to_datetime(df["time"], errors="coerce", utc=True).dt.floor("H")

        # Provider stats (winner by output_amount)
        stats_df = compute_provider_stats(df)

        st.subheader("Provider Statistics")
        # Nice formatting for rates
        if not stats_df.empty:
            stats_fmt = stats_df.copy()
            stats_fmt["win_rate"] = (stats_fmt["win_rate"] * 100).round(2).astype(str) + "%"
            stats_fmt["avg_loss_vs_winner"] = (stats_fmt["avg_loss_vs_winner"] * 100).round(3).astype(str) + "%"
            stats_fmt["median_loss_vs_winner"] = (stats_fmt["median_loss_vs_winner"] * 100).round(3).astype(str) + "%"
            st.dataframe(stats_fmt, use_container_width=True)
        else:
            st.info("No stats to display with the selected filters.")

        # Raw quotes table that fed the calc
        st.subheader("Filtered Quotes")
        show_cols = [
            "time","src_chain","dst_chain","src_token","dst_token",
            "input_amount","output_amount","provider","trade_size_bin","time_hour"
        ]
        existing_cols = [c for c in show_cols if c in df.columns]
        st.dataframe(df[existing_cols].sort_values("time", ascending=False), use_container_width=True)