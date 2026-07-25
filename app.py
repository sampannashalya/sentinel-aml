from __future__ import annotations

from datetime import date, datetime

import streamlit as st

from config.settings import IBM_TRANSACTION_PATH
from investigation import InvestigationReport
from risk import RiskAssessment
from ui.components import (
    breakdown_rows,
    evidence_rows,
    format_amount,
    format_amount_compact,
    format_risk_score,
    ibm_dataset_status_label,
    parse_and_plan_query,
    timeline_rows,
)
from ui.demo_data import ALL_TYPOLOGIES, INVESTIGATION_SOURCE_LABEL, build_account_investigation_bundle, build_demo_bundle


def main() -> None:
    st.set_page_config(
        page_title="SentinelAML",
        page_icon="Shield",
        layout="wide",
    )
    st.title("SentinelAML")
    st.subheader("Agentic Anti-Money Laundering Investigation System")
    st.caption("Explainable AML pattern detection, risk scoring, and investigation support.")
    st.info(
        "This application identifies suspicious behavioral patterns for analyst review. "
        "It does not establish money laundering, fraud, criminal conduct, or legal liability."
    )

    mode = st.sidebar.selectbox("Investigation mode", ["Demo Investigation", "Account Investigation"])
    account_id = st.sidebar.text_input("Account ID", value="ACCT-DEMO-1")
    typologies = st.sidebar.multiselect("Detector / typology selection", ALL_TYPOLOGIES, default=["FAN-OUT", "VELOCITY", "CYCLE"])
    start_date = st.sidebar.date_input("Start date", value=date(2022, 9, 1))
    end_date = st.sidebar.date_input("End date", value=date(2022, 9, 18))
    query_text = st.sidebar.text_input("Natural-language query", value="Investigate fan-out activity for account 800737690")
    run_investigation = st.sidebar.button("Run Investigation", type="primary")

    st.sidebar.divider()
    st.sidebar.write(ibm_dataset_status_label(IBM_TRANSACTION_PATH.exists()))

    query_preview = parse_and_plan_query(query_text)
    with st.sidebar.expander("Query parser / planner preview", expanded=False):
        st.write(f"Intent: `{query_preview['intent']}`")
        st.write(f"Pattern: `{query_preview['aml_pattern'] or 'none'}`")
        st.write("Planned tools:")
        st.write(", ".join(query_preview["tool_names"]) or "None")
        st.caption(query_preview["planning_summary"])

    if not run_investigation:
        st.subheader("How to Start")
        st.write("Choose a mode in the sidebar and click **Run Investigation**.")
        st.write("Demo mode works without raw IBM files. Account mode uses targeted existing backend paths when the data is available.")
        return

    if mode == "Demo Investigation":
        bundle = build_demo_bundle(selected_typologies=typologies, generated_at=datetime(2022, 9, 1, 12, 0))
    else:
        bundle = build_account_investigation_bundle(
            account_id=account_id.strip() or "UNKNOWN",
            selected_typologies=typologies,
            start_date=datetime.combine(start_date, datetime.min.time()) if start_date else None,
            end_date=datetime.combine(end_date, datetime.min.time()) if end_date else None,
        )

    st.subheader(bundle.source_label)
    if bundle.is_synthetic:
        st.warning(INVESTIGATION_SOURCE_LABEL)
    if bundle.notes:
        for note in bundle.notes:
            st.info(note)

    if bundle.report is None or bundle.assessment is None:
        st.warning("No investigation report could be produced for the selected inputs.")
        return

    _render_report(bundle.assessment, bundle.report)


def _render_report(assessment: RiskAssessment, report: InvestigationReport) -> None:
    metrics = st.columns(6)
    metrics[0].metric("Risk Score", format_risk_score(assessment.risk_score))
    metrics[1].metric("Risk Level", assessment.risk_level.value if hasattr(assessment.risk_level, "value") else str(assessment.risk_level))
    metrics[2].metric("Evidence Count", str(assessment.evidence_count))
    metrics[3].metric("Typologies", str(len(assessment.typologies_detected)))
    metrics[4].metric("Suspicious Tx", str(assessment.total_suspicious_transactions))
    metrics[5].metric("Suspicious Amount", format_amount_compact(assessment.total_suspicious_amount))

    left, right = st.columns([1.25, 1])

    with left:
        st.subheader("Investigation Report")
        st.write(f"**Report ID:** {report.report_id}")
        st.write(f"**Executive Summary:** {report.executive_summary}")
        st.write("**Key Findings**")
        st.write("\n".join(f"- {finding}" for finding in report.key_findings) or "- None")
        st.write("**Recommended Actions**")
        st.write("\n".join(f"- {action}" for action in report.recommended_actions))
        st.write("**Limitations**")
        st.write("\n".join(f"- {item}" for item in report.limitations))
        st.warning(report.compliance_disclaimer)

    with right:
        st.subheader("Score Breakdown")
        st.dataframe(breakdown_rows(assessment.score_breakdown), use_container_width=True, hide_index=True)
        st.subheader("Timeline")
        st.dataframe(timeline_rows(report), use_container_width=True, hide_index=True)

    st.subheader("Evidence Summary")
    evidence_frame = evidence_rows_from_assessment(assessment)
    if evidence_frame.empty:
        st.info("No contributing evidence was available.")
    else:
        st.dataframe(evidence_frame, use_container_width=True, hide_index=True)

    if report.metadata:
        with st.expander("Report metadata", expanded=False):
            st.json(report.metadata)


def evidence_rows_from_assessment(assessment: RiskAssessment):
    frame = evidence_rows(assessment)
    if not frame.empty:
        frame["Amount"] = frame["Amount"].map(format_amount)
    return frame


if __name__ == "__main__":
    main()
