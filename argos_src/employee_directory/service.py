"""Site-scoped employee directory warmup and identity matching."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import threading
from typing import Any, Callable, Optional

from rapidfuzz import fuzz as rapidfuzz_fuzz
from rapidfuzz.distance import JaroWinkler as rapidfuzz_jaro_winkler

logger = logging.getLogger(__name__)

EMPLOYEE_COLUMNS = (
    "EMPLOYEE_NAME",
    "BUSINESS_TITLE",
    "TIME_IN_JOB_PROFILE",
    "EMPLOYEE_USERNAME",
    "JOB_FAMILY",
    "JOB_FAMILY_GROUP",
    "JOB_LEVEL",
    "C_LEVEL",
    "MANAGER_NAME",
    "COST_CENTER",
    "SENIOR_LEADERSHIP_TEAM",
    "BUSINESS_FUNCTION",
)
EMPLOYEE_DIRECTORY_SQL = """
    SELECT
        cemp."EMPLOYEE_NAME",
        cemp."BUSINESS_TITLE",
        cemp."TIME_IN_JOB_PROFILE",
        cemp."EMPLOYEE_USERNAME",
        cemp."JOB_FAMILY",
        cemp."JOB_FAMILY_GROUP",
        cemp."JOB_LEVEL",
        cemp."C_LEVEL",
        emp."EMPLOYEE_MANAGER1_NAME" AS "MANAGER_NAME",
        cemp."COST_CENTER",
        cemp."SENIOR_LEADERSHIP_TEAM",
        cemp."BUSINESS_FUNCTION"
    FROM "EDLDB"."CHEWYBI"."CHEWYDATA_CURRENT_EMPLOYEES" cemp
    LEFT JOIN "EDLDB"."CHEWYBI"."EMPLOYEES" emp
        ON CAST(emp."EMPLOYEE_ID" AS VARCHAR) = CAST(cemp."EMPLOYEE_ID" AS VARCHAR)
    WHERE cemp."LOCATION_CODE" = %s
"""
MAX_CANDIDATES = 3
MIN_PLAUSIBLE_SCORE = 74.0
CLARIFY_SCORE = 84.0
AUTO_CONFIRM_SCORE = 98.0
CLEAR_GAP_SCORE = 5.0
MULTIPLE_MATCH_GAP = 3.0


def load_env_file() -> None:
    """Load Snowflake connection settings from a nearby `.snowflake_env` file."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / ".snowflake_env",
        script_dir / ".snowflake_env",
        script_dir.parent / ".snowflake_env",
        script_dir.parent.parent / ".snowflake_env",
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        for raw_line in candidate.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
        return
    raise FileNotFoundError("Could not find a .snowflake_env file.")


def require_env(name: str) -> str:
    """Return a required environment variable or raise a readable error."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def connect_snowflake_from_env() -> Any:
    """Connect to Snowflake using the standard Argos environment variables."""
    import snowflake.connector

    return snowflake.connector.connect(
        account=require_env("SNOWFLAKE_ACCOUNT"),
        user=require_env("SNOWFLAKE_USER"),
        password=os.environ.get("SNOWFLAKE_PASSWORD") or None,
        authenticator=os.environ.get("SNOWFLAKE_AUTHENTICATOR") or None,
        role=os.environ.get("SNOWFLAKE_ROLE") or None,
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE") or None,
        database=require_env("SNOWFLAKE_DATABASE"),
        schema=os.environ.get("SNOWFLAKE_SCHEMA") or None,
    )


def load_directory_records_by_site(
    site_codes: list[str] | tuple[str, ...],
    *,
    env_loader: Callable[[], None] = load_env_file,
    connector_factory: Callable[[], Any] = connect_snowflake_from_env,
) -> dict[str, list["EmployeeRecord"]]:
    """Load multiple site directories through a single Snowflake connection."""
    env_loader()
    connection = connector_factory()
    try:
        records_by_site: dict[str, list[EmployeeRecord]] = {}
        with connection.cursor() as cursor:
            for raw_site_code in site_codes:
                site_code = str(raw_site_code or "").strip()
                cursor.execute(EMPLOYEE_DIRECTORY_SQL, (site_code,))
                rows = cursor.fetchall()
                records_by_site[site_code] = [
                    EmployeeRecord.from_row(tuple(row)) for row in rows
                ]
        return records_by_site
    finally:
        connection.close()


def _normalize_name(value: str) -> str:
    normalized = "".join(
        character.lower() if character.isalnum() else " "
        for character in str(value or "")
    )
    return " ".join(normalized.split())


def _token_sort_key(value: str) -> str:
    normalized = _normalize_name(value)
    if not normalized:
        return ""
    return " ".join(sorted(normalized.split()))


def _query_name_parts(value: str) -> tuple[str, str]:
    parts = [part for part in _normalize_name(value).split() if part]
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[-1]


def _build_query_identity(
    *,
    shared_first_name: str,
    shared_last_name: str,
    shared_name: str,
) -> tuple[str, str, str]:
    first_name = _normalize_name(shared_first_name)
    last_name = _normalize_name(shared_last_name)
    full_name = _normalize_name(shared_name)

    if not first_name and not last_name and full_name:
        first_name, last_name = _query_name_parts(full_name)
    elif full_name and (not first_name or not last_name):
        fallback_first_name, fallback_last_name = _query_name_parts(full_name)
        first_name = first_name or fallback_first_name
        last_name = last_name or fallback_last_name

    if not full_name:
        full_name = " ".join(part for part in (first_name, last_name) if part)

    return full_name, first_name, last_name


def _score_ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    scores = [
        rapidfuzz_fuzz.WRatio(left, right),
        rapidfuzz_fuzz.ratio(left, right),
        100.0 * rapidfuzz_jaro_winkler.normalized_similarity(left, right),
    ]
    return float(max(scores))


@dataclass(frozen=True)
class EmployeeRecord:
    """One normalized employee directory row cached in memory."""

    official_name: str
    employee_name: str
    username: str
    business_title: str
    job_family: str
    job_family_group: str
    job_level: str
    c_level: str
    manager_name: str
    cost_center: str
    senior_leadership_team: str
    business_function: str
    tenure: str
    normalized_name: str
    token_sorted_name: str

    @classmethod
    def from_row(cls, row: tuple[Any, ...]) -> "EmployeeRecord":
        def _value(index: int) -> str:
            if index >= len(row):
                return ""
            return str(row[index] or "").strip()

        official_name = _value(0)
        normalized_name = _normalize_name(official_name)
        return cls(
            official_name=official_name,
            employee_name=official_name,
            username=_value(3),
            business_title=_value(1),
            job_family=_value(4),
            job_family_group=_value(5),
            job_level=_value(6),
            c_level=_value(7),
            manager_name=_value(8),
            cost_center=_value(9),
            senior_leadership_team=_value(10),
            business_function=_value(11),
            tenure=_value(2),
            normalized_name=normalized_name,
            token_sorted_name=_token_sort_key(official_name),
        )


@dataclass(frozen=True)
class _ScoredCandidate:
    record: EmployeeRecord
    score: float


class EmployeeDirectoryService:
    """Load a site-scoped employee directory in the background and resolve names."""

    def __init__(
        self,
        *,
        site_code: str,
        env_loader: Callable[[], None] = load_env_file,
        connector_factory: Callable[[], Any] = connect_snowflake_from_env,
    ) -> None:
        self.site_code = str(site_code or "").strip()
        self._env_loader = env_loader
        self._connector_factory = connector_factory
        self._records: list[EmployeeRecord] = []
        self._normalized_name_index: dict[str, list[EmployeeRecord]] = {}
        self._token_sorted_name_index: dict[str, list[EmployeeRecord]] = {}
        self._lock = threading.RLock()
        self._ready = threading.Event()
        self._warmup_finished = threading.Event()
        self._stop_event = threading.Event()
        self._warmup_thread: Optional[threading.Thread] = None
        self._last_error = ""

    def start_background(self) -> None:
        """Warm the site directory in a background daemon thread."""
        thread = self._warmup_thread
        if thread is not None and thread.is_alive():
            return
        self._warmup_finished.clear()
        self._warmup_thread = threading.Thread(
            target=self._background_warmup,
            name="employee-directory-warmup",
            daemon=True,
        )
        self._warmup_thread.start()

    def _background_warmup(self) -> None:
        try:
            if self._stop_event.is_set():
                return
            self.load_directory()
        except Exception as exc:  # pragma: no cover - covered through state assertions.
            with self._lock:
                self._last_error = str(exc)
            logger.exception(
                "Failed to warm employee directory for site=%s",
                self.site_code,
            )
        finally:
            self._warmup_finished.set()

    def load_directory(self) -> list[EmployeeRecord]:
        """Load the configured site directory synchronously."""
        records = self._load_records_from_snowflake()
        self.set_loaded_records(records)
        logger.info(
            "Loaded %s employee directory rows for site=%s",
            len(records),
            self.site_code,
        )
        return records

    def set_loaded_records(self, records: list[EmployeeRecord]) -> list[EmployeeRecord]:
        """Hydrate the in-memory directory cache with preloaded employee records."""
        normalized_name_index: dict[str, list[EmployeeRecord]] = defaultdict(list)
        token_sorted_name_index: dict[str, list[EmployeeRecord]] = defaultdict(list)
        for record in records:
            normalized_name_index[record.normalized_name].append(record)
            token_sorted_name_index[record.token_sorted_name].append(record)
        with self._lock:
            self._records = records
            self._normalized_name_index = dict(normalized_name_index)
            self._token_sorted_name_index = dict(token_sorted_name_index)
            self._last_error = ""
            self._ready.set()
            self._warmup_finished.set()
        return records

    def mark_load_failed(self, error: str) -> None:
        """Mark the directory as unavailable after a load failure."""
        with self._lock:
            self._last_error = str(error or "")
            self._ready.clear()
            self._warmup_finished.set()

    def wait_for_warmup(self, timeout: float | None = None) -> bool:
        """Block until the background warmup finishes."""
        return self._warmup_finished.wait(timeout=timeout)

    def is_ready(self) -> bool:
        """Return whether the directory has been loaded successfully."""
        return self._ready.is_set()

    def get_verified_profile(
        self,
        *,
        username: str = "",
        official_name: str = "",
    ) -> dict[str, Any] | None:
        """Return one cached employee profile by username or unique exact name."""
        cleaned_username = str(username or "").strip().casefold()
        cleaned_name = _normalize_name(official_name)
        token_sorted_name = _token_sort_key(official_name)
        if not cleaned_username and not cleaned_name:
            return None

        with self._lock:
            if not self._ready.is_set():
                return None
            records = list(self._records)
            normalized_name_index = {
                key: list(value)
                for key, value in self._normalized_name_index.items()
            }
            token_sorted_name_index = {
                key: list(value)
                for key, value in self._token_sorted_name_index.items()
            }

        if cleaned_username:
            for record in records:
                if record.username.strip().casefold() == cleaned_username:
                    return self._profile_payload(record)

        exact_matches = normalized_name_index.get(cleaned_name, [])
        if not exact_matches and token_sorted_name:
            exact_matches = token_sorted_name_index.get(token_sorted_name, [])
        if len(exact_matches) == 1:
            return self._profile_payload(exact_matches[0])
        return None

    def shutdown(self) -> None:
        """Stop waiting on background work and join the warmup thread briefly."""
        self._stop_event.set()
        thread = self._warmup_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    def resolve_identity(
        self,
        shared_first_name: str = "",
        shared_last_name: str = "",
        shared_name: str = "",
    ) -> dict[str, Any]:
        """Resolve one shared identity against the cached site directory."""
        cleaned_name, first_name, last_name = _build_query_identity(
            shared_first_name=str(shared_first_name or "").strip(),
            shared_last_name=str(shared_last_name or "").strip(),
            shared_name=str(shared_name or "").strip(),
        )
        if not cleaned_name:
            return self._build_result(
                success=False,
                status="invalid_input",
                message=(
                    "I still need the person's first and last name before I can look "
                    "them up."
                ),
                candidates=[],
            )
        if not first_name or not last_name:
            return self._build_result(
                success=False,
                status="invalid_input",
                message="Please ask the person to share their first and last name separately.",
                candidates=[],
            )

        with self._lock:
            ready = self._ready.is_set()
            records = list(self._records)
            normalized_name_index = {
                key: list(value)
                for key, value in self._normalized_name_index.items()
            }
            token_sorted_name_index = {
                key: list(value)
                for key, value in self._token_sorted_name_index.items()
            }
            last_error = self._last_error

        if not ready:
            message = (
                "The employee directory is still warming up. Please try again in a moment."
                if not self._warmup_finished.is_set()
                else "I couldn't load the employee directory for this location yet. Please try again later."
            )
            result = self._build_result(
                success=False,
                status="directory_unavailable",
                message=message,
                candidates=[],
            )
            if last_error:
                result["error"] = last_error
            return result

        scored = self._score_candidates(
            cleaned_name,
            records=records,
            normalized_name_index=normalized_name_index,
            token_sorted_name_index=token_sorted_name_index,
        )
        if not scored:
            return self._build_result(
                success=True,
                status="no_match",
                message=(
                    f"I couldn't find a good employee match for that name at {self.site_code}."
                ),
                candidates=[],
            )

        candidates = scored[:MAX_CANDIDATES]
        status = self._classify_candidates(
            candidates,
        )
        if status == "single_match":
            message = f"I found one strong employee match for that name at {self.site_code}."
        elif status == "multiple_matches":
            message = "I found a few strong employee matches. Please confirm which one is right."
        elif status == "needs_clarification":
            message = "I found a possible employee match, but I need a bit more detail to be sure."
        else:
            return self._build_result(
                success=True,
                status="no_match",
                message=(
                    f"I couldn't find a good employee match for that name at {self.site_code}."
                ),
                candidates=[],
            )
        return self._build_result(
            success=True,
            status=status,
            message=message,
            candidates=[self._candidate_payload(item) for item in candidates],
        )

    def _load_records_from_snowflake(self) -> list[EmployeeRecord]:
        self._env_loader()
        connection = self._connector_factory()
        try:
            with connection.cursor() as cursor:
                cursor.execute(EMPLOYEE_DIRECTORY_SQL, (self.site_code,))
                rows = cursor.fetchall()
        finally:
            connection.close()
        return [EmployeeRecord.from_row(tuple(row)) for row in rows]

    def _score_candidates(
        self,
        shared_name: str,
        *,
        records: list[EmployeeRecord],
        normalized_name_index: dict[str, list[EmployeeRecord]],
        token_sorted_name_index: dict[str, list[EmployeeRecord]],
    ) -> list[_ScoredCandidate]:
        normalized_input = _normalize_name(shared_name)
        if not normalized_input:
            return []
        token_sorted_input = _token_sort_key(shared_name)
        exact_name_matches = normalized_name_index.get(normalized_input, [])
        if exact_name_matches:
            return self._sort_candidates(
                [
                    _ScoredCandidate(record=record, score=100.0)
                    for record in exact_name_matches
                ]
            )

        exact_token_matches = token_sorted_name_index.get(token_sorted_input, [])
        if exact_token_matches:
            return self._sort_candidates(
                [
                    _ScoredCandidate(record=record, score=99.0)
                    for record in exact_token_matches
                ]
            )

        scored: list[_ScoredCandidate] = []
        for record in records:
            score = self._score_record(
                normalized_input=normalized_input,
                token_sorted_input=token_sorted_input,
                record=record,
            )
            if score >= MIN_PLAUSIBLE_SCORE:
                scored.append(_ScoredCandidate(record=record, score=score))
        return self._sort_candidates(scored)

    @staticmethod
    def _sort_candidates(
        candidates: list[_ScoredCandidate],
    ) -> list[_ScoredCandidate]:
        candidates.sort(
            key=lambda item: (
                -item.score,
                item.record.official_name.casefold(),
                item.record.business_title.casefold(),
            )
        )
        return candidates

    @staticmethod
    def _score_record(
        *,
        normalized_input: str,
        token_sorted_input: str,
        record: EmployeeRecord,
    ) -> float:
        full_name_score = _score_ratio(normalized_input, record.normalized_name)
        token_score = _score_ratio(token_sorted_input, record.token_sorted_name)

        return max(
            full_name_score,
            0.85 * token_score + 0.15 * full_name_score,
        )

    @staticmethod
    def _classify_candidates(
        candidates: list[_ScoredCandidate],
    ) -> str:
        top = candidates[0]
        runner_up = candidates[1] if len(candidates) > 1 else None
        duplicate_name = any(
            candidate.record.normalized_name == top.record.normalized_name
            for candidate in candidates[1:]
        )
        if (
            top.score >= AUTO_CONFIRM_SCORE
            and not duplicate_name
            and (
                runner_up is None
                or (top.score - runner_up.score) >= CLEAR_GAP_SCORE
            )
        ):
            return "single_match"
        if duplicate_name:
            return "multiple_matches"
        if runner_up is not None and (top.score - runner_up.score) <= MULTIPLE_MATCH_GAP:
            return "multiple_matches"
        if top.score >= CLARIFY_SCORE:
            return "needs_clarification"
        return "no_match"

    def _build_result(
        self,
        *,
        success: bool,
        status: str,
        message: str,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "success": success,
            "status": status,
            "message": message,
            "site_code": self.site_code,
            "candidate_count": len(candidates),
            "data": {
                "site_code": self.site_code,
                "candidate_count": len(candidates),
                "candidates": candidates,
            },
        }

    @staticmethod
    def _candidate_payload(candidate: _ScoredCandidate) -> dict[str, Any]:
        record = candidate.record
        payload = {
            "official_name": record.official_name,
            "employee_name": record.employee_name,
            "username": record.username,
            "business_title": record.business_title,
            "tenure": record.tenure,
        }
        payload["match_score"] = round(candidate.score, 1)
        return payload

    @staticmethod
    def _profile_payload(record: EmployeeRecord) -> dict[str, Any]:
        return {
            "official_name": record.official_name,
            "employee_name": record.employee_name,
            "username": record.username,
            "business_title": record.business_title,
            "job_family": record.job_family,
            "job_family_group": record.job_family_group,
            "job_level": record.job_level,
            "c_level": record.c_level,
            "manager_name": record.manager_name,
            "cost_center": record.cost_center,
            "senior_leadership_team": record.senior_leadership_team,
            "business_function": record.business_function,
            "tenure": record.tenure,
        }
