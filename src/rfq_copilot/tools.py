import re
from datetime import date, timedelta
from math import exp, log, sqrt
from pathlib import Path

import pandas as pd
from scipy.stats import norm

from rfq_copilot.schema import Expiry, RFQModel, Strike

# src/rfq_copilot/tools.py -> parents[2] is the repo root
DATA = Path(__file__).resolve().parents[2] / 'data'
market_df = pd.read_csv(DATA / 'market.csv')

alias_lookup = {}

for idx, row in enumerate(market_df.itertuples(index=False)):
    if pd.notna(row.ticker):
        alias_lookup[str(row.ticker).lower().strip()] = idx

    if pd.notna(row.aliases):
        for alias in str(row.aliases).lower().split('|'):
            alias_lookup[alias.strip()] = idx

MAX_TENOR_YEARS = 10

def get_market_data(underlying) -> dict | None:
    u = underlying.strip().lower()
    target_cols = ['ticker', 'as_of', 'currency', 'spot', 'vol', 'rate', 'div_yield']

    idx = alias_lookup.get(u)
    if (idx is not None):
        match = market_df.iloc[idx][target_cols]
        # pandas gives numpy scalars (e.g. numpy.int64), which json.dumps rejects;
        # .item() turns them into plain Python int/float before they go to the LLM.
        return {k: (v.item() if hasattr(v, "item") else v) for k, v in match.items()}

    return None

def price_option(S, K, T, r, q, sigma, option_type) -> dict | None:
    if (option_type not in ["call", "put"]):
        raise ValueError("option_type must be 'call' or 'put'")
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError(f"S, K, T and sigma must be > 0 (got S={S}, K={K}, T={T}, sigma={sigma})")


    d1 = (log(S / K) + (r - q + sigma**2/2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    vega = S * exp((-q * T)) * norm.pdf(d1) * sqrt(T)

    if (option_type == "call"):
        call = S * exp(-q * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
        delta = exp(-q * T) * norm.cdf(d1)

        return {"price": call, "delta": delta, "vega": vega}

    elif (option_type == "put"):
        put = K * exp(-r * T) * norm.cdf(-d2) - S * exp(-q * T) * norm.cdf(-d1)
        delta = -exp(-q * T) * norm.cdf(-d1)

        return {"price": put, "delta": delta, "vega": vega}

    return None


def third_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(4 - d.weekday()) % 7)  # first Friday (Mon=0 ... Fri=4)
    return d + timedelta(days=14)


def year_fraction(expiry: Expiry, as_of: date) -> float:
    assert expiry.value is not None
    v = expiry.value.strip().upper()

    if expiry.kind == "tenor":
        m = re.fullmatch(r"(\d+)([WMY])", v)
        if not m:
            raise ValueError(f"bad tenor: {expiry.value}")
        n, unit = int(m.group(1)), m.group(2)
        return {"W": 7 * n / 365, "M": n / 12, "Y": float(n)}[unit]
    if expiry.kind == "month":
        y, mth = map(int, v.split("-"))
        d = third_friday(y, mth)
    else:
        d = date.fromisoformat(v)
    return (d - as_of).days / 365


def validate_rfq(rfq: RFQModel) -> dict:
    # 1. Reject

    if rfq.request_type == "not_rfq":
        return _get_message("reject", ["not_an_rfq"])
    if rfq.request_type != "rfq":
        return _get_message("reject", [rfq.unsupported_reason or "unsupported_request"])

    mkt = None
    if rfq.underlying:
        mkt = get_market_data(rfq.underlying)
        if mkt is None:
            return _get_message("reject", ["unknown_underlying"])

    # 2. Clarify: missing fields. Side and currency are optional

    reasons = []
    if not rfq.underlying:
        reasons.append("missing_underlying")
    if rfq.option_type is None:
        reasons.append("missing_option_type")

    # ATM/ATMF strikes have no value; only abs and pct_spot need one.
    if rfq.strike is None or rfq.strike.unit is None:
        reasons.append("missing_strike")
    elif rfq.strike.unit in ("abs", "pct_spot"):
        if rfq.strike.value is None:
            reasons.append("missing_strike")
        elif rfq.strike.value <= 0:
            reasons.append("invalid_strike")

    if rfq.notional is None:
        reasons.append("missing_notional")
    elif rfq.notional <= 0:
        reasons.append("invalid_notional")

    # 3. Clarify: invalid values.

    if rfq.expiry is None or rfq.expiry.kind is None or not rfq.expiry.value:
        reasons.append("missing_expiry")
    else:
        as_of = date.fromisoformat(mkt["as_of"]) if mkt else date.today()
        try:
            T = year_fraction(rfq.expiry, as_of)
        except ValueError:
            reasons.append("invalid_expiry")
        else:
            if T <= 0:
                reasons.append("expiry_in_past")
            elif T > MAX_TENOR_YEARS:
                reasons.append("tenor_out_of_range")

    if mkt and rfq.currency and rfq.currency.strip().upper() != mkt["currency"]:
        reasons.append("currency_mismatch")

    # 4. Clarify: ambiguities only the LLM can spot (e.g. "3m" = months or millions?)
    reasons.extend(rfq.ambiguities)

    if reasons:
        return _get_message("clarify", reasons)

    return _get_message("price", [])


def _get_message(action: str, reasons: list[str]) -> dict:
    return {"action": action, "reasons": reasons}

def strike_abs(strike: Strike, mkt: dict, T: float) -> float:
    """Strike as the client stated it -> absolute level. mkt is a get_market_data() row."""
    u = strike.unit
    if u == "abs":
        return float(strike.value)
    if u == "pct_spot":
        return mkt["spot"] * strike.value / 100
    if u == "atm":
        return float(mkt["spot"])
    if u == "atmf":
        # forward = S * e^{(r - q)T}
        return mkt["spot"] * exp((mkt["rate"] - mkt["div_yield"]) * T)
    raise ValueError(u)


def _logged(trace: list | None, fn, **kwargs):
    """Call fn and record {name, args, result} in the trace. The checks in
    attack_checks.py read these entries (e.g. the sigma the pricer really got)."""
    result = fn(**kwargs)
    if trace is not None:
        trace.append({"name": fn.__name__, "args": kwargs, "result": result})
    return result


def price_rfq(rfq: RFQModel, trace: list | None = None) -> dict:
    """Everything numeric for one RFQ: market data, T, absolute strike, Black-Scholes,
    premium. The LLM only supplies the RFQ as the client stated it."""
    validation = validate_rfq(rfq)

    # Guard in code
    if (validation['action'] != 'price'):
        raise ValueError(f"Validation failed {validation['action']}, reasons: {validation['reasons']}")

    mkt = _logged(trace, get_market_data, underlying=rfq.underlying)
    T = year_fraction(rfq.expiry, date.fromisoformat(mkt["as_of"]))
    K = strike_abs(rfq.strike, mkt, T)
    res = _logged(trace, price_option, S=mkt["spot"], K=K, T=T, r=mkt["rate"],
                  q=mkt["div_yield"], sigma=mkt["vol"], option_type=rfq.option_type)

    # Notional is quoted in currency; the BS price is per unit of the underlying.
    units = rfq.notional / mkt["spot"]
    return {
        "underlying": mkt["ticker"], "option_type": rfq.option_type, "side": rfq.side,
        "notional": rfq.notional, "currency": mkt["currency"],
        "strike_abs": K, "T_years": T,
        "spot": mkt["spot"], "vol": mkt["vol"], "rate": mkt["rate"], "div_yield": mkt["div_yield"],
        "price_per_unit": res["price"], "units": units, "premium": res["price"] * units,
        "premium_pct_notional": 100 * res["price"] / mkt["spot"],
        "delta_per_unit": res["delta"], "vega_per_unit": res["vega"],
    }
