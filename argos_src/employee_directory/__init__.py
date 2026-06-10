"""Snowflake-backed employee directory helpers for Argos registration."""

from .service import EmployeeDirectoryService, EmployeeRecord

__all__ = [
    "EmployeeDirectoryService",
    "EmployeeRecord",
]
