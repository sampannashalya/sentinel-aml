from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(slots=True)
class PatternAttempt:
    attempt_id: int
    typology: str
    header: str
    description: str
    transactions: list[dict[str, str]]
    accounts: list[str]


@dataclass(slots=True)
class PatternAnnotationSummary:
    attempts: list[PatternAttempt]

    @property
    def unique_typologies(self) -> list[str]:
        return sorted({attempt.typology for attempt in self.attempts})

    def attempt_counts_by_typology(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for attempt in self.attempts:
            counts[attempt.typology] = counts.get(attempt.typology, 0) + 1
        return counts


class IBMPatternAnnotationParser:
    def __init__(self, pattern_path: str | Path) -> None:
        self.pattern_path = Path(pattern_path)

    def parse(self) -> PatternAnnotationSummary:
        attempts: list[PatternAttempt] = []
        current_header: str | None = None
        current_typology: str | None = None
        current_description: str = ""
        current_transactions: list[dict[str, str]] = []
        current_accounts: set[str] = set()
        attempt_id = 0

        for raw_line in self.pattern_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("BEGIN LAUNDERING ATTEMPT -"):
                current_header = line
                current_typology = self._normalize_typology(line)
                current_description = self._extract_description(line)
                current_transactions = []
                current_accounts = set()
                attempt_id += 1
                continue
            if line.startswith("END LAUNDERING ATTEMPT -"):
                if current_header and current_typology:
                    attempts.append(
                        PatternAttempt(
                            attempt_id=attempt_id,
                            typology=current_typology,
                            header=current_header,
                            description=current_description,
                            transactions=current_transactions[:],
                            accounts=sorted(current_accounts),
                        )
                    )
                current_header = None
                current_typology = None
                current_description = ""
                current_transactions = []
                current_accounts = set()
                continue
            if current_typology and self._looks_like_transaction_row(line):
                transaction = self._parse_transaction_row(line)
                current_transactions.append(transaction)
                current_accounts.update([transaction["sender_account"], transaction["receiver_account"]])

        return PatternAnnotationSummary(attempts=attempts)

    def _normalize_typology(self, header_line: str) -> str:
        match = re.match(r"BEGIN LAUNDERING ATTEMPT\s+-\s+([^:]+)", header_line)
        if not match:
            raise ValueError(f"Could not parse pattern typology from header: {header_line}")
        return match.group(1).strip().upper().replace(" ", "-")

    def _extract_description(self, header_line: str) -> str:
        if ":" not in header_line:
            return ""
        return header_line.split(":", 1)[1].strip()

    def _looks_like_transaction_row(self, line: str) -> bool:
        parts = [part.strip() for part in line.split(",")]
        return len(parts) >= 10 and parts[0][:4].isdigit()

    def _parse_transaction_row(self, line: str) -> dict[str, str]:
        parts = next(csv.reader([line], skipinitialspace=True))
        parts = [part.strip() for part in parts]
        return {
            "timestamp": parts[0],
            "from_bank": parts[1],
            "sender_account": parts[2],
            "to_bank": parts[3],
            "receiver_account": parts[4],
            "amount_received": parts[5],
            "receiving_currency": parts[6],
            "amount_paid": parts[7],
            "payment_currency": parts[8],
            "payment_format": parts[9],
            "label": parts[10] if len(parts) > 10 else "",
        }
