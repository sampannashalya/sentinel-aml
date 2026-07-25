from __future__ import annotations

import re
from datetime import date, timedelta

from .query_schema import (
    AMLPattern,
    NumericThreshold,
    QueryDateRange,
    QueryIntent,
    QueryRequest,
    RequestedOutput,
)


class QueryParser:
    def __init__(self, reference_date: date | None = None) -> None:
        self.reference_date = reference_date or date.today()

    def parse(self, query: str) -> QueryRequest:
        normalized = " ".join(query.strip().split())
        lowered = normalized.lower()

        customer_id = self._extract_customer_id(normalized)
        date_range = self._extract_date_range(lowered)
        amount_threshold = self._extract_amount_threshold(lowered)
        transaction_count_threshold = self._extract_transaction_count_threshold(lowered)
        transaction_type = self._extract_transaction_type(lowered)
        country = self._extract_country(normalized)
        intent = self._detect_intent(lowered)
        aml_pattern = self._detect_pattern(lowered, intent)
        requested_output = self._infer_requested_output(lowered, intent)

        return QueryRequest(
            raw_query=normalized,
            intent=intent,
            customer_id=customer_id,
            date_range=date_range,
            amount_threshold=amount_threshold,
            transaction_count_threshold=transaction_count_threshold,
            transaction_type=transaction_type,
            country=country,
            aml_pattern=aml_pattern,
            requested_output=requested_output,
        )

    def _detect_intent(self, lowered: str) -> QueryIntent:
        if "why" in lowered and ("high risk" in lowered or "suspicious" in lowered):
            return QueryIntent.risk_explanation
        if re.search(r"\b(is|investigate) customer\b", lowered):
            return QueryIntent.customer_investigation
        if "show high-risk customers" in lowered or "show high risk customers" in lowered:
            return QueryIntent.high_risk_search
        if self._has_fan_out_terms(lowered):
            return QueryIntent.fan_out_detection
        if self._has_fan_in_terms(lowered):
            return QueryIntent.fan_in_detection
        if "velocity" in lowered:
            return QueryIntent.velocity_analysis
        if "structuring" in lowered:
            return QueryIntent.structuring_detection
        if "smurfing" in lowered:
            return QueryIntent.smurfing_detection
        if "eda" in lowered:
            return QueryIntent.eda_request
        if "suspicious activity" in lowered:
            return QueryIntent.suspicious_activity_search
        if self._has_threshold_terms(lowered):
            return QueryIntent.threshold_analysis
        return QueryIntent.broad_analysis

    def _extract_customer_id(self, query: str) -> str | None:
        match = re.search(r"\b(CUST_[A-Z0-9]+)\b", query.upper())
        return match.group(1) if match else None

    def _extract_date_range(self, lowered: str) -> QueryDateRange | None:
        match = re.search(r"last\s+(\d+)\s+days?", lowered)
        if not match:
            return None

        relative_days = int(match.group(1))
        end_date = self.reference_date
        start_date = end_date - timedelta(days=relative_days)
        return QueryDateRange(start_date=start_date, end_date=end_date, relative_days=relative_days)

    def _extract_amount_threshold(self, lowered: str) -> NumericThreshold | None:
        amount_match = re.search(
            r"(?:under|below|less than|at most|no more than)\s+\$?([0-9][0-9,]*(?:\.[0-9]+)?)",
            lowered,
        )
        if amount_match:
            return NumericThreshold(operator="<", value=self._to_number(amount_match.group(1)), currency="USD")

        amount_match = re.search(
            r"(?:over|above|more than|greater than|at least)\s+\$?([0-9][0-9,]*(?:\.[0-9]+)?)",
            lowered,
        )
        if amount_match:
            return NumericThreshold(operator=">=", value=self._to_number(amount_match.group(1)), currency="USD")

        return None

    def _extract_transaction_count_threshold(self, lowered: str) -> NumericThreshold | None:
        match = re.search(r"(\d+)\s*(?:or more|\+|plus|at least)\s+transactions?", lowered)
        if match:
            return NumericThreshold(operator=">=", value=float(match.group(1)))

        match = re.search(r"(?:more than|over)\s+(\d+)\s+transactions?", lowered)
        if match:
            return NumericThreshold(operator=">", value=float(match.group(1)))

        match = re.search(r"(?:under|below|less than)\s+(\d+)\s+transactions?", lowered)
        if match:
            return NumericThreshold(operator="<", value=float(match.group(1)))

        return None

    def _extract_transaction_type(self, lowered: str) -> str | None:
        for candidate in ("card", "transfer", "atm", "cash", "wire", "deposit", "withdrawal"):
            if candidate in lowered:
                return candidate
        return None

    def _extract_country(self, query: str) -> str | None:
        match = re.search(r"\b(US|GB|DE|FR|IN|CA|AE|SG|AU|CH)\b", query)
        return match.group(1) if match else None

    def _detect_pattern(self, lowered: str, intent: QueryIntent) -> AMLPattern | None:
        if self._has_fan_out_terms(lowered) or intent == QueryIntent.fan_out_detection:
            return AMLPattern.fan_out
        if self._has_fan_in_terms(lowered) or intent == QueryIntent.fan_in_detection:
            return AMLPattern.fan_in
        if "structuring" in lowered or intent == QueryIntent.structuring_detection:
            return AMLPattern.structuring
        if "smurfing" in lowered or intent == QueryIntent.smurfing_detection:
            return AMLPattern.smurfing
        if "velocity" in lowered or intent == QueryIntent.velocity_analysis:
            return AMLPattern.velocity
        if "behavior" in lowered or "behaviour" in lowered:
            return AMLPattern.behavior_deviation
        if intent == QueryIntent.suspicious_activity_search:
            return AMLPattern.suspicious_activity
        return None

    def _infer_requested_output(self, lowered: str, intent: QueryIntent) -> RequestedOutput:
        if intent == QueryIntent.eda_request:
            return RequestedOutput.eda_summary
        if intent == QueryIntent.risk_explanation:
            return RequestedOutput.explanation
        if intent == QueryIntent.customer_investigation:
            return RequestedOutput.case_file
        if intent == QueryIntent.high_risk_search:
            return RequestedOutput.customer_list
        if intent in {
            QueryIntent.structuring_detection,
            QueryIntent.smurfing_detection,
            QueryIntent.fan_out_detection,
            QueryIntent.fan_in_detection,
            QueryIntent.velocity_analysis,
        }:
            return RequestedOutput.pattern_summary
        if self._has_threshold_terms(lowered):
            return RequestedOutput.customer_list
        if "analyse this dataset" in lowered or "analyze this dataset" in lowered:
            return RequestedOutput.investigation_summary
        if "suspicious activity" in lowered:
            return RequestedOutput.investigation_summary
        return RequestedOutput.summary

    def _has_threshold_terms(self, lowered: str) -> bool:
        return bool(
            re.search(r"\b\d+\s*(?:or more|\+|plus|at least)\s+transactions?\b", lowered)
            or re.search(r"\b(?:under|below|less than|over|above|more than)\s+\$?\d", lowered)
        )

    def _has_fan_out_terms(self, lowered: str) -> bool:
        return bool(
            re.search(r"\bfan[- ]?out\b", lowered)
            or ("sending" in lowered and "many" in lowered and ("receiver" in lowered or "different accounts" in lowered))
            or ("source account" in lowered and "many" in lowered and "receiver" in lowered)
        )

    def _has_fan_in_terms(self, lowered: str) -> bool:
        return bool(
            re.search(r"\bfan[- ]?in\b", lowered)
            or ("receiving" in lowered and "many" in lowered and ("sender" in lowered or "different accounts" in lowered))
            or ("received" in lowered and "many" in lowered and ("sender" in lowered or "different accounts" in lowered))
        )

    def _to_number(self, text: str) -> float:
        return float(text.replace(",", ""))
