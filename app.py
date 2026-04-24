import io
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from rapidfuzz import fuzz, process


APP_TITLE = "GTM Account Mapping Tool"
DEFAULT_SCORE_THRESHOLD = 85


@dataclass
class MatchResult:
    matched_name: Optional[str]
    score: float
    status: str


def normalize_text(value: object) -> str:
    """Normalize company names for robust matching."""
    if pd.isna(value):
        return ""

    text = str(value).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


@st.cache_data(show_spinner=False)
def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Load CSV or Excel input into a DataFrame."""
    if uploaded_file is None:
        return pd.DataFrame()

    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)

    raise ValueError("Unsupported file format. Please upload CSV, XLS, or XLSX files.")


def find_company_column(df: pd.DataFrame, preferred_names: List[str]) -> Optional[str]:
    """Find best-fit company column by common naming patterns."""
    normalized_column_map = {normalize_text(col): col for col in df.columns}

    for candidate in preferred_names:
        if candidate in normalized_column_map:
            return normalized_column_map[candidate]

    # Fallback: any column containing the word company/account/org/attendee company.
    keywords = ["company", "account", "organization", "organisation", "employer"]
    for norm_col, original_col in normalized_column_map.items():
        if any(keyword in norm_col for keyword in keywords):
            return original_col

    return None


def build_exact_lookup(customer_names: pd.Series) -> Dict[str, str]:
    """Map normalized customer company name -> original display name."""
    lookup: Dict[str, str] = {}
    for original in customer_names.dropna().astype(str):
        normalized = normalize_text(original)
        if normalized and normalized not in lookup:
            lookup[normalized] = original
    return lookup


def run_matching(
    customers_df: pd.DataFrame,
    attendees_df: pd.DataFrame,
    customer_col: str,
    attendee_col: str,
    threshold: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run exact + fuzzy matching and return processed DataFrames."""
    customers = customers_df.copy()
    attendees = attendees_df.copy()

    customers["_normalized_company"] = customers[customer_col].apply(normalize_text)
    attendees["_normalized_company"] = attendees[attendee_col].apply(normalize_text)

    exact_lookup = build_exact_lookup(customers[customer_col])
    customer_choice_keys = list(exact_lookup.keys())

    def match_one(attendee_name_norm: str) -> MatchResult:
        if not attendee_name_norm:
            return MatchResult(matched_name=None, score=0.0, status="NO")

        if attendee_name_norm in exact_lookup:
            return MatchResult(
                matched_name=exact_lookup[attendee_name_norm],
                score=100.0,
                status="YES",
            )

        if not customer_choice_keys:
            return MatchResult(matched_name=None, score=0.0, status="NO")

        best = process.extractOne(
            attendee_name_norm,
            customer_choice_keys,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold,
        )

        if best is None:
            return MatchResult(matched_name=None, score=0.0, status="NO")

        matched_norm_name, score, _ = best
        return MatchResult(
            matched_name=exact_lookup.get(matched_norm_name),
            score=float(score),
            status="YES",
        )

    match_records = attendees["_normalized_company"].apply(match_one)

    attendees["Matched Customer Account"] = match_records.apply(lambda r: r.matched_name)
    attendees["Match Score"] = match_records.apply(lambda r: round(r.score, 2))
    attendees["Match Status"] = match_records.apply(lambda r: r.status)
    attendees["Priority Level"] = attendees["Match Status"].map({"YES": "High", "NO": "Low"})

    matched_df = attendees[attendees["Match Status"] == "YES"].copy()
    unmatched_df = attendees[attendees["Match Status"] == "NO"].copy()

    export_df = attendees.drop(columns=["_normalized_company"])
    customers = customers.drop(columns=["_normalized_company"])

    return matched_df, unmatched_df, export_df


def highlight_matched_rows(row: pd.Series) -> List[str]:
    if row.get("Match Status") == "YES":
        return ["background-color: #e8f7ea"] * len(row)
    return [""] * len(row)


def build_excel_download(attendee_results: pd.DataFrame, customer_df: pd.DataFrame) -> bytes:
    """Create downloadable Excel bytes with source + output sheets."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        attendee_results.to_excel(writer, index=False, sheet_name="Attendee Mapping Results")
        customer_df.to_excel(writer, index=False, sheet_name="Customer Accounts")
    output.seek(0)
    return output.read()


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Map event attendees against your customer accounts to find high-priority overlaps.")

    st.markdown("## Upload")
    left, right = st.columns(2)

    with left:
        customer_file = st.file_uploader(
            "Upload Customer Accounts (CSV/XLS/XLSX)",
            type=["csv", "xls", "xlsx"],
            key="customer_file",
        )

    with right:
        attendee_file = st.file_uploader(
            "Upload Event Attendees (CSV/XLS/XLSX)",
            type=["csv", "xls", "xlsx"],
            key="attendee_file",
        )

    if not customer_file or not attendee_file:
        st.info("Upload both files to start account mapping.")
        return

    try:
        customers_df = load_uploaded_file(customer_file)
        attendees_df = load_uploaded_file(attendee_file)
    except Exception as exc:
        st.error(f"Could not read one or both files: {exc}")
        return

    if customers_df.empty or attendees_df.empty:
        st.warning("One of the uploaded files is empty. Please upload valid data.")
        return

    customer_col = find_company_column(
        customers_df,
        preferred_names=[
            "company",
            "company name",
            "account",
            "account name",
            "customer",
            "customer name",
        ],
    )
    attendee_col = find_company_column(
        attendees_df,
        preferred_names=["company", "company name", "attendee company", "organization", "employer"],
    )

    if not customer_col or not attendee_col:
        st.error(
            "Missing required company name columns. "
            "Ensure both files include a company/account column (e.g., 'Company Name')."
        )
        return

    threshold = st.slider(
        "Fuzzy match threshold",
        min_value=60,
        max_value=100,
        value=DEFAULT_SCORE_THRESHOLD,
        help="Higher values are stricter. 100 means only exact token-level matches pass.",
    )

    matched_df, unmatched_df, export_df = run_matching(
        customers_df=customers_df,
        attendees_df=attendees_df,
        customer_col=customer_col,
        attendee_col=attendee_col,
        threshold=threshold,
    )

    total_accounts = len(export_df)
    total_matches = len(matched_df)
    match_pct = (total_matches / total_accounts * 100) if total_accounts else 0.0
    high_value_pct = (export_df["Priority Level"].eq("High").mean() * 100) if total_accounts else 0.0

    st.markdown("## Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Attendee Accounts", f"{total_accounts:,}")
    c2.metric("Total Matches", f"{total_matches:,}")
    c3.metric("Match %", f"{match_pct:.1f}%")
    c4.metric("High-Priority Accounts", f"{total_matches:,}")

    st.subheader("Matched Accounts (YES)")
    if matched_df.empty:
        st.write("No matched accounts found with current threshold.")
    else:
        st.dataframe(matched_df.style.apply(highlight_matched_rows, axis=1), use_container_width=True)

    st.subheader("Unmatched Accounts")
    st.dataframe(unmatched_df, use_container_width=True)

    st.markdown("## Insights")
    st.write(f"**% of high-value accounts:** {high_value_pct:.1f}%")

    top_matches = (
        matched_df["Matched Customer Account"]
        .dropna()
        .value_counts()
        .head(10)
        .rename_axis("Company")
        .reset_index(name="Matched Attendees")
    )

    if top_matches.empty:
        st.write("No top matched companies to display yet.")
    else:
        st.write("**Top matched companies**")
        st.table(top_matches)

    excel_bytes = build_excel_download(export_df, customers_df)
    st.download_button(
        label="Download Mapping Results (Excel)",
        data=excel_bytes,
        file_name="gtm_account_mapping_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    main()
