import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import joblib

# This module contains lightweight ML utilities for churn risk scoring
# It trains a logistic regression model using pseudo-labels derived from activity recency
# and opportunity interactions. The model is persisted to disk and refreshed periodically.

# NOTE: This is a bootstrap model. For production, replace pseudo-labeling with real churn labels
# from your historical data and consider more robust features.

MODEL_DIRNAME = "models"
MODEL_FILENAME = "churn_model.pkl"
MODEL_MAX_AGE_SEC = 24 * 3600  # refresh daily


def _now_utc() -> datetime:
    return datetime.utcnow()


def _model_path(base_dir: str) -> str:
    d = os.path.join(base_dir, MODEL_DIRNAME)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, MODEL_FILENAME)


def _safe_parse_iso(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(ts) if ts else None
    except Exception:
        return None


def _build_customer_features(db) -> pd.DataFrame:
    """Aggregate per-customer features from sqlite3 connection.
    Returns a DataFrame with columns:
      ['customer_id','days_since_update','days_since_created','lead_count',
       'opp_open_cnt','opp_won_cnt','opp_lost_cnt','opp_open_amt','opp_won_amt_180d']
    """
    # Customers
    customers = db.execute("SELECT id, created_at, updated_at FROM customers").fetchall()
    if not customers:
        # empty DF with required columns
        cols = [
            'customer_id','days_since_update','days_since_created','lead_count',
            'opp_open_cnt','opp_won_cnt','opp_lost_cnt','opp_open_amt','opp_won_amt_180d'
        ]
        return pd.DataFrame(columns=cols)

    now = _now_utc()

    cust_df = pd.DataFrame([
        {
            'customer_id': r['id'],
            'days_since_update': (now - (_safe_parse_iso(r['updated_at']) or now)).days,
            'days_since_created': (now - (_safe_parse_iso(r['created_at']) or now)).days,
        }
        for r in customers
    ])

    # Leads per customer
    leads = db.execute("SELECT customer_id, COUNT(*) as c FROM leads WHERE customer_id IS NOT NULL GROUP BY customer_id").fetchall()
    lead_df = pd.DataFrame([{'customer_id': r['customer_id'], 'lead_count': r['c']} for r in leads]) if leads else pd.DataFrame(columns=['customer_id','lead_count'])

    # Opportunities per customer with breakdowns
    opp_rows = db.execute(
        """
        SELECT customer_id, stage, status, amount, close_date
        FROM opportunities
        WHERE customer_id IS NOT NULL
        """
    ).fetchall()

    opp_df = pd.DataFrame([dict(r) for r in opp_rows]) if opp_rows else pd.DataFrame(columns=['customer_id','stage','status','amount','close_date'])

    if not opp_df.empty:
        opp_df['amount'] = pd.to_numeric(opp_df['amount'], errors='coerce').fillna(0.0)
        opp_df['stage_l'] = opp_df['stage'].str.lower().fillna('')
        opp_df['status_l'] = opp_df['status'].str.lower().fillna('')
        opp_df['close_date_dt'] = pd.to_datetime(opp_df['close_date'], errors='coerce')

        # Aggregations
        open_mask = (opp_df['stage_l'].isin(['prospecting','qualification','proposal','negotiation']) | ((opp_df['status_l'] == 'open') & (~opp_df['stage_l'].isin(['won','lost']))))
        won_mask = opp_df['stage_l'] == 'won'
        lost_mask = opp_df['stage_l'] == 'lost'

        agg = opp_df.groupby('customer_id').apply(lambda g: pd.Series({
            'opp_open_cnt': int(open_mask.loc[g.index].sum()),
            'opp_won_cnt': int(won_mask.loc[g.index].sum()),
            'opp_lost_cnt': int(lost_mask.loc[g.index].sum()),
            'opp_open_amt': float(g.loc[open_mask.loc[g.index], 'amount'].sum()),
            'opp_won_amt_180d': float(g.loc[won_mask.loc[g.index] & (g['close_date_dt'] >= (pd.Timestamp.now(tz=None) - pd.Timedelta(days=180))), 'amount'].sum()),
        })).reset_index()
    else:
        agg = pd.DataFrame(columns=['customer_id','opp_open_cnt','opp_won_cnt','opp_lost_cnt','opp_open_amt','opp_won_amt_180d'])

    # Merge
    df = cust_df.merge(lead_df, on='customer_id', how='left') \
                 .merge(agg, on='customer_id', how='left')
    for col in ['lead_count','opp_open_cnt','opp_won_cnt','opp_lost_cnt','opp_open_amt','opp_won_amt_180d']:
        df[col] = df[col].fillna(0)

    # Ensure types
    int_cols = ['lead_count','opp_open_cnt','opp_won_cnt','opp_lost_cnt']
    for c in int_cols:
        df[c] = df[c].astype(int)
    for c in ['opp_open_amt','opp_won_amt_180d']:
        df[c] = df[c].astype(float)

    return df


def _pseudo_labels(df: pd.DataFrame) -> np.ndarray:
    """Create churn pseudo-labels using simple rules.
    1 if likely churned based on inactivity and no engagement; else 0.
    """
    if df.empty:
        return np.array([], dtype=int)
    # Heuristics
    inactive = df['days_since_update'] > 90
    no_open = df['opp_open_cnt'] == 0
    no_recent_wins = df['opp_won_amt_180d'] <= 0.0
    low_leads = df['lead_count'] == 0
    churn = (inactive & no_open & no_recent_wins) | (inactive & low_leads)
    return churn.astype(int).values


def train_churn_model(db, base_dir: str) -> Tuple[Pipeline, List[str]]:
    df = _build_customer_features(db)
    feature_cols = [
        'days_since_update','days_since_created','lead_count',
        'opp_open_cnt','opp_won_cnt','opp_lost_cnt','opp_open_amt','opp_won_amt_180d'
    ]
    if df.empty:
        return None, feature_cols

    y = _pseudo_labels(df)
    X = df[feature_cols].fillna(0.0).values

    # If only one class present, model can't train -> return None to use heuristic
    if len(np.unique(y)) < 2:
        return None, feature_cols

    # Class weights to balance
    classes = np.array([0, 1])
    weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=500, class_weight=class_weight, n_jobs=1))
    ])

    pipeline.fit(X, y)

    # Persist
    path = _model_path(base_dir)
    joblib.dump({'model': pipeline, 'features': feature_cols, 'trained_at': _now_utc().isoformat()}, path)

    return pipeline, feature_cols


def load_churn_model(base_dir: str):
    path = _model_path(base_dir)
    if not os.path.exists(path):
        return None
    try:
        bundle = joblib.load(path)
        return bundle
    except Exception:
        return None


def ensure_churn_model(db, base_dir: str):
    path = _model_path(base_dir)
    if not os.path.exists(path):
        train_churn_model(db, base_dir)
        return
    # Refresh if older than threshold
    try:
        mtime = os.path.getmtime(path)
        if (time.time() - mtime) > MODEL_MAX_AGE_SEC:
            train_churn_model(db, base_dir)
    except Exception:
        # If anything goes wrong, attempt retrain
        train_churn_model(db, base_dir)


def predict_churn_probs(db, base_dir: str) -> pd.DataFrame:
    df = _build_customer_features(db)
    if df.empty:
        return df.assign(prob=[])  # empty
    bundle = load_churn_model(base_dir)
    feature_cols = [
        'days_since_update','days_since_created','lead_count',
        'opp_open_cnt','opp_won_cnt','opp_lost_cnt','opp_open_amt','opp_won_amt_180d'
    ]

    X = df[feature_cols].fillna(0.0).values
    if bundle and bundle.get('model') is not None:
        model: Pipeline = bundle['model']
        probs = model.predict_proba(X)[:, 1]
    else:
        # Heuristic fallback: normalized linear risk
        z = (
            0.03 * df['days_since_update'] +
            0.02 * df['days_since_created'] -
            0.10 * df['opp_won_cnt'] -
            0.06 * df['opp_open_cnt'] -
            0.00002 * df['opp_open_amt'] -
            0.00005 * df['opp_won_amt_180d'] -
            0.05 * df['lead_count'] +
            0.04 * df['opp_lost_cnt']
        )
        # Sigmoid
        probs = 1 / (1 + np.exp(-z.clip(-10, 10)))

    out = df.copy()
    out['prob'] = probs
    return out


def _runtime_data_dir() -> str:
    """Return a user-writable data directory for models, esp. on Windows frozen builds."""
    try:
        import sys
        if os.name == 'nt':
            appdata = os.getenv('LOCALAPPDATA') or os.path.expanduser('~')
            d = os.path.join(appdata, 'NIVARA')
            os.makedirs(d, exist_ok=True)
            return d
        # For non-Windows, prefer MEIPASS (read-only) for assets but models should be next to executable dir
        base = getattr(sys, 'frozen', False)
        if base:
            return os.path.dirname(getattr(sys, 'executable', sys.argv[0]))
        return os.path.dirname(__file__)
    except Exception:
        return os.path.dirname(__file__)


def churn_summary(db) -> Dict:
    """Compute churn KPIs and top at-risk customers.
    Returns keys: avg_risk, high_risk_percent, expected_churn_amount, top_customers
    """
    # Use a writable data directory for model caching
    base_dir = _runtime_data_dir()

    probs_df = predict_churn_probs(db, base_dir)
    if probs_df.empty:
        return {
            'avg_risk': 0.0,
            'high_risk_percent': 0.0,
            'expected_churn_amount': 0.0,
            'top_customers': []
        }

    # Expected churn amount = sum over customers (risk * open_pipeline_amount)
    total_open = probs_df['opp_open_amt'].sum()
    expected_churn = float((probs_df['prob'] * probs_df['opp_open_amt']).sum())

    # Build top list: need names
    # Fetch names for top N
    top_df = probs_df.sort_values('prob', ascending=False).head(5)
    ids = tuple(int(x) for x in top_df['customer_id'].tolist())
    name_map = {}
    if len(ids) > 0:
        placeholders = ",".join(["?"] * len(ids))
        rows = db.execute(f"SELECT id, name FROM customers WHERE id IN ({placeholders})", ids).fetchall()
        name_map = {r['id']: r['name'] for r in rows}

    top_customers = [
        {
            'id': int(row.customer_id),
            'name': name_map.get(int(row.customer_id), f"Customer {int(row.customer_id)}"),
            'risk': float(row.prob),
            'open_pipeline': float(row.opp_open_amt),
        }
        for row in top_df.itertuples(index=False)
    ]

    return {
        'avg_risk': float(probs_df['prob'].mean()),
        'high_risk_percent': float((probs_df['prob'] >= 0.60).mean()),
        'expected_churn_amount': expected_churn,
        'top_customers': top_customers,
        'total_open_pipeline': float(total_open),
    }


def ensure_ready(db):
    """Public helper to ensure model exists/refreshed."""
    try:
        import sys
        base_dir = getattr(sys, '_MEIPASS', None) or os.path.dirname(__file__)
    except Exception:
        base_dir = os.path.dirname(__file__)
    ensure_churn_model(db, base_dir)