"""Snowflake-backed employee directory helpers for Argos registration."""

from .service import EmployeeDirectoryService, EmployeeRecord, employee_email_from_username

__all__ = [
    "EmployeeDirectoryService",
    "EmployeeRecord",
    "employee_email_from_username",
]
