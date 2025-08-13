# clean_excel_app.py
import io
import json
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Excel Cleaner • FP&A Ready", layout="wide")

# ---------------------------
# Helpers
# ---------------------------
def parse_mapping(text: str) -> Dict[str, str]:
    """
    Parse a mapping typed like: "old1:new1, old2:new2".
    Whitespace around items is stripped.
    """
    mapping = {}
    if not text:
        return mapping
    for pair in text.split(","):
        if ":" in pair:
            k, v = pair.split(":", 1)
            mapping[k.strip()] = v.strip()
    return mapping

def safe_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(r"[^\d\.\-eE]", "", regex=True), errors="coerce")

def apply_simple_transform(series: pd.Series, op: str, **kwargs) -> pd.Series:
    if op == "strip":
        return series.astype(str).str.strip()
    if op == "lower":
        return series.astype(str).str.lower()
    if op == "upper":
        return series.astype(str).str.upper()
    if op == "title":
        return series.astype(str).str.title()
    if op == "currency_to_float":
        return safe_to_numeric(series)
    if op == "date_parse":
        fmt = kwargs.get("date_format") or None
        return pd.to_datetime(series, format=fmt, errors="coerce")
    if op == "multiply":
        factor = kwargs.get("factor", 1.0)
        return pd.to_numeric(series, errors="coerce") * factor
    if op == "add":
        c = kwargs.get("constant", 0.0)
        return pd.to_numeric(series, errors="coerce") + c
    if op == "divide":
        d = kwargs.get("divisor", 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return pd.to_numeric(series, errors="coerce") / d
    return series

def clean_dataframe(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()

    # 1) Drop Missing
    if cfg.get("drop_na", {}).get("enabled"):
        subset = cfg["drop_na"].get("subset") or None
        out = out.dropna(subset=subset)

    # 2) Fill Missing
    if cfg.get("fillna", {}).get("enabled"):
        cols = cfg["fillna"].get("columns") or out.columns.tolist()
        strategy = cfg["fillna"].get("strategy", "constant")
        value = cfg["fillna"].get("value")
        for c in cols:
            if strategy == "constant":
                out[c] = out[c].fillna(value)
            elif strategy == "median":
                out[c] = out[c].fillna(pd.to_numeric(out[c], errors="coerce").median())
            elif strategy == "mean":
                out[c] = out[c].fillna(pd.to_numeric(out[c], errors="coerce").mean())
            elif strategy == "zero":
                out[c] = out[c].fillna(0)
            else:
                out[c] = out[c].fillna(value)

    # 3) Replace Values
    if cfg.get("replace", {}).get("enabled"):
        c = cfg["replace"].get("column")
        mapping = cfg["replace"].get("mapping", {})
        if c and mapping:
            out[c] = out[c].replace(mapping)

    # 4) Rename Columns
    if cfg.get("rename", {}).get("enabled"):
        mapping = cfg["rename"].get("mapping", {})
        if mapping:
            out = out.rename(columns=mapping)

    # 5) Filter Rows by Profit Margin > 10%
    if cfg.get("filter_margin", {}).get("enabled"):
        margin_col = cfg["filter_margin"].get("margin_col")
        rev_col = cfg["filter_margin"].get("revenue_col", "Revenue")
        cogs_col = cfg["filter_margin"].get("cogs_col", "COGS")
        threshold = cfg["filter_margin"].get("threshold", 0.10)
        if margin_col and margin_col in out.columns:
            margin = pd.to_numeric(out[margin_col], errors="coerce")
        elif rev_col in out.columns and cogs_col in out.columns:
            revenue = pd.to_numeric(out[rev_col], errors="coerce")
            cogs = pd.to_numeric(out[cogs_col], errors="coerce")
            margin = (revenue - cogs) / revenue.replace(0, np.nan)
            out["ProfitMargin"] = margin
        else:
            margin = None

        if margin is not None:
            out = out[margin > threshold]

    # 6) Apply Function to a Column
    if cfg.get("transform", {}).get("enabled"):
        col = cfg["transform"].get("column")
        op = cfg["transform"].get("operation", "strip")
        kwargs = cfg["transform"].get("kwargs", {})
        if col in out.columns:
            out[col] = apply_simple_transform(out[col], op, **kwargs)

    # 7) Create a New Column via Expression
    if cfg.get("newcol", {}).get("enabled"):
        name = cfg["newcol"].get("name") or "NewColumn"
        expr = cfg["newcol"].get("expression")
        if expr:
            try:
                # Use pandas.eval for vectorized arithmetic on columns
                out[name] = pd.eval(expr, engine="python", parser="pandas", local_dict={c: out[c] for c in out.columns})
            except Exception:
                # Fall back to numeric coercion of common form: Revenue - COGS
                try:
                    out[name] = pd.to_numeric(out[cfg["newcol"]["left"]], errors="coerce") - pd.to_numeric(out[cfg["newcol"]["right"]], errors="coerce")
                except Exception:
                    pass

    # 8) Change Data Types
    if cfg.get("astype", {}).get("enabled"):
        mapping = cfg["astype"].get("mapping", {})
        for c, t in mapping.items():
            if c not in out.columns:
                continue
            if t == "int":
                out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
            elif t == "float":
                out[c] = pd.to_numeric(out[c], errors="coerce")
            elif t == "str":
                out[c] = out[c].astype(str)
            elif t == "datetime":
                out[c] = pd.to_datetime(out[c], errors="coerce")
            else:
                try:
                    out[c] = out[c].astype(t)
                except Exception:
                    pass

    # 9) Sort by Revenue (desc) or chosen column
    if cfg.get("sort", {}).get("enabled"):
        col = cfg["sort"].get("by", "Revenue")
        ascending = not cfg["sort"].get("descending", True)
        if col in out.columns:
            out = out.sort_values(by=col, ascending=ascending, kind="mergesort")

    # 10) Drop Duplicates
    if cfg.get("dedupe", {}).get("enabled"):
        subset = cfg["dedupe"].get("subset") or None
        keep = cfg["dedupe"].get("keep", "first")
        out = out.drop_duplicates(subset=subset, keep=keep)

    return out

# ---------------------------
# UI
# ---------------------------
st.title("Excel Cleaner — FP&A Ready Output")

with st.sidebar:
    st.header("How it works")
    st.markdown(
        """
        1. Upload an Excel/CSV file.
        2. Configure the cleaning steps you want (left→right).
        3. Click **Run Cleaning** to preview and download the cleaned file.
        """
    )

uploaded = st.file_uploader("Upload Excel/CSV", type=["xlsx", "xls", "csv"])
sheet = None
if uploaded is not None and uploaded.name.lower().endswith((".xlsx", ".xls")):
    xls = pd.ExcelFile(uploaded)
    sheet = st.selectbox("Select sheet", options=xls.sheet_names, index=0)
    df = pd.read_excel(xls, sheet_name=sheet)
elif uploaded is not None and uploaded.name.lower().endswith(".csv"):
    df = pd.read_csv(uploaded)
else:
    df = None

if df is not None:
    st.success(f"Loaded shape: {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(20), use_container_width=True)

    with st.form("cfg"):
        st.subheader("Configure Steps")

        # 1) Drop Missing
        drop_na_enabled = st.checkbox("Drop Missing Values", value=False)
        drop_subset = st.multiselect("Drop NA in subset (optional)", options=df.columns.tolist()) if drop_na_enabled else []

        # 2) Fill NA
        fill_enabled = st.checkbox("Fill Missing Values", value=False)
        fill_cols = st.multiselect("Columns to fill (blank = all)", options=df.columns.tolist()) if fill_enabled else []
        fill_strategy = st.selectbox("Fill strategy", ["constant", "median", "mean", "zero"]) if fill_enabled else "constant"
        fill_value = st.text_input("Constant fill value (used when strategy=constant)", value="") if fill_enabled else ""

        # 3) Replace Values
        rep_enabled = st.checkbox("Replace Values", value=False)
        rep_col = st.selectbox("Column for replacements", options=[""] + df.columns.tolist()) if rep_enabled else ""
        rep_map_text = st.text_input('Mapping "old:new, old2:new2"', value="") if rep_enabled else ""

        # 4) Rename Columns
        ren_enabled = st.checkbox("Rename Columns", value=False)
        ren_map_text = st.text_input('Rename map "Old:New, Old2:New2"', value="") if ren_enabled else ""

        # 5) Filter Rows by Profit Margin > threshold
        filt_enabled = st.checkbox("Filter Rows by Profit Margin", value=False)
        threshold = st.number_input("Margin threshold (e.g., 0.10 = 10%)", value=0.10, step=0.01) if filt_enabled else 0.10
        margin_col = st.selectbox("Use existing margin column (optional)", options=[""] + df.columns.tolist()) if filt_enabled else ""
        rev_col = st.selectbox("Revenue column (if computing margin)", options=["Revenue"] + df.columns.tolist()) if filt_enabled else "Revenue"
        cogs_col = st.selectbox("COGS column (if computing margin)", options=["COGS"] + df.columns.tolist()) if filt_enabled else "COGS"

        # 6) Apply Function to Column
        tr_enabled = st.checkbox("Apply Function to a Column", value=False)
        tr_col = st.selectbox("Column to transform", options=[""] + df.columns.tolist()) if tr_enabled else ""
        tr_op = st.selectbox("Operation", ["strip", "lower", "upper", "title", "currency_to_float", "date_parse", "multiply", "add", "divide"]) if tr_enabled else "strip"
        tr_kwargs = {}
        if tr_enabled:
            if tr_op == "date_parse":
                tr_kwargs["date_format"] = st.text_input("Date format (optional, e.g., %d/%m/%Y)", value="")
            if tr_op == "multiply":
                tr_kwargs["factor"] = st.number_input("Factor", value=1.0)
            if tr_op == "add":
                tr_kwargs["constant"] = st.number_input("Constant", value=0.0)
            if tr_op == "divide":
                tr_kwargs["divisor"] = st.number_input("Divisor", value=1.0, min_value=0.0000001)

        # 7) Create New Column
        new_enabled = st.checkbox("Create a New Column (expression)")
        new_name = st.text_input("New column name", value="NewColumn") if new_enabled else "NewColumn"
        example = "Revenue - COGS  # or (NetIncome / Revenue)"
        new_expr = st.text_area("Expression (use column names)", value=example) if new_enabled else ""

        # 8) Change Data Types
        astype_enabled = st.checkbox("Change Data Types", value=False)
        dtype_cols = st.multiselect("Columns to change type", options=df.columns.tolist()) if astype_enabled else []
        dtype_target = st.selectbox("Target dtype", ["int", "float", "str", "datetime"]) if astype_enabled else "str"

        # 9) Sort
        sort_enabled = st.checkbox("Sort Data", value=True)
        sort_by = st.selectbox("Sort by column", options=df.columns.tolist(), index=(list(df.columns).index("Revenue") if "Revenue" in df.columns else 0)) if sort_enabled else None
        sort_desc = st.checkbox("Descending", value=True) if sort_enabled else True

        # 10) De-duplicate
        dedupe_enabled = st.checkbox("Drop Duplicates", value=False)
        dedupe_subset = st.multiselect("Duplicate subset (leave empty = all columns)", options=df.columns.tolist()) if dedupe_enabled else []

        run = st.form_submit_button("Run Cleaning")

    if run:
        cfg = {
            "drop_na": {"enabled": drop_na_enabled, "subset": drop_subset or None},
            "fillna": {"enabled": fill_enabled, "columns": fill_cols or None, "strategy": fill_strategy, "value": None if fill_value == "" else fill_value},
            "replace": {"enabled": rep_enabled, "column": rep_col or None, "mapping": parse_mapping(rep_map_text)},
            "rename": {"enabled": ren_enabled, "mapping": parse_mapping(ren_map_text)},
            "filter_margin": {"enabled": filt_enabled, "margin_col": margin_col or None, "revenue_col": rev_col, "cogs_col": cogs_col, "threshold": threshold},
            "transform": {"enabled": tr_enabled, "column": tr_col or None, "operation": tr_op, "kwargs": tr_kwargs},
            "newcol": {"enabled": new_enabled, "name": new_name, "expression": new_expr},
            "astype": {"enabled": astype_enabled, "mapping": {c: dtype_target for c in dtype_cols}},
            "sort": {"enabled": sort_enabled, "by": sort_by, "descending": sort_desc},
            "dedupe": {"enabled": dedupe_enabled, "subset": dedupe_subset or None, "keep": "first"},
        }

        cleaned = clean_dataframe(df, cfg)
        st.success(f"Cleaned shape: {cleaned.shape[0]} rows × {cleaned.shape[1]} columns")
        st.dataframe(cleaned.head(50), use_container_width=True)

        # Downloads
        excel_bytes = io.BytesIO()
        with pd.ExcelWriter(excel_bytes, engine="xlsxwriter") as writer:
            cleaned.to_excel(writer, index=False, sheet_name="Cleaned")
            meta = pd.DataFrame([{ "RunAt": datetime.now().isoformat(timespec="seconds"), "SourceFile": uploaded.name }])
            meta.to_excel(writer, index=False, sheet_name="Meta")
        excel_bytes.seek(0)

        st.download_button(
            label="⬇️ Download Cleaned Excel",
            data=excel_bytes,
            file_name=f"cleaned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.download_button(
            label="⬇️ Download Cleaned CSV",
            data=cleaned.to_csv(index=False).encode("utf-8"),
            file_name=f"cleaned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

else:
    st.info("Upload an Excel/CSV file to begin.")
