"""Resolve Slack users into Argos person ids where possible."""

from __future__ import annotations

from typing import Any

from argos_src.memory.slack.models import SlackUserProfile


class SlackIdentityResolver:
    def __init__(self, identity_store: Any | None = None) -> None:
        self.identity_store = identity_store

    def resolve_user(self, profile: SlackUserProfile) -> SlackUserProfile:
        person_id = self._person_id_for_profile(profile)
        if not person_id:
            return profile
        return SlackUserProfile(
            slack_user_id=profile.slack_user_id,
            username=profile.username,
            display_name=profile.display_name,
            real_name=profile.real_name,
            email=profile.email,
            person_id=person_id,
        )

    def _person_id_for_profile(self, profile: SlackUserProfile) -> str:
        store = self.identity_store
        if store is None:
            return ""
        slack_user_id = profile.slack_user_id.casefold()
        email = profile.email.casefold()
        username = profile.username.casefold() or (email.split("@", 1)[0] if email else "")
        for candidate in (
            profile.display_name,
            profile.real_name,
            profile.username,
            email.split("@", 1)[0] if email else "",
        ):
            rendered = str(candidate or "").strip()
            if not rendered:
                continue
            try:
                resolved = store.resolve_person_id(rendered)
            except Exception:
                resolved = None
            if resolved:
                return str(resolved)
        try:
            people = store.list_people()
        except Exception:
            return ""
        for person in people:
            metadata = dict(person.get("metadata") or {})
            slack_ids = {
                str(metadata.get("slack_user_id") or "").casefold(),
                str(metadata.get("slack_id") or "").casefold(),
            }
            if slack_user_id and slack_user_id in slack_ids:
                return str(person.get("person_id") or "")
            emails = {
                str(metadata.get("slack_email") or "").casefold(),
                str(metadata.get("email") or "").casefold(),
                str(metadata.get("work_email") or "").casefold(),
            }
            if email and email in emails:
                return str(person.get("person_id") or "")
            usernames = {
                str(metadata.get("slack_username") or "").casefold(),
                str(metadata.get("username") or "").casefold(),
                str(metadata.get("employee_username") or "").casefold(),
            }
            if username and username in usernames:
                return str(person.get("person_id") or "")
        return ""
