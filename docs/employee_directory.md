# Employee Directory Validation

Read this with:

- `argos_src/employee_directory/service.py`
- `argos_src/tools/unitree_go2/vision/resolve_employee_identity.py`

This document explains how employee name validation works during registration.

## Mental Model

The model should collect the person's first and last name separately, then call
`resolve_employee_identity`.

The service loads a site-scoped employee directory from Snowflake and compares the
provided name against the cached employees for that site.

## Inputs

The tool now expects:

- `shared_first_name`
- `shared_last_name`
- optional `shared_name`

`shared_name` is only there for logging and backward compatibility. The main path
is first name plus last name.

## What Gets Loaded

For each employee row, the service loads:

- `EMPLOYEE_NAME`
- `BUSINESS_TITLE`
- `TIME_IN_JOB_PROFILE`

The directory is filtered by `LOCATION_CODE`, so matching is site-specific.
Name matching uses `EMPLOYEE_NAME` as the source of truth. The tool still asks
the model for first and last name separately, then the service joins those inputs
into a spoken full-name query. It compares that query against `EMPLOYEE_NAME`
instead of loading or trusting Snowflake first-name and last-name component
columns, because those may be masked.

## How Matching Works

The matcher normalizes names by lowercasing and removing punctuation differences.

It then scores candidates in this order:

1. exact normalized full-name match
2. exact token-order-insensitive full-name match
3. fuzzy full-name match with RapidFuzz

Full-name similarity against `EMPLOYEE_NAME` is the match signal. Imperfect fuzzy
matches are returned for clarification rather than auto-confirmed.

The matcher is intentionally conservative:

- strong clean match -> auto accept
- plausible but imperfect match -> ask for clarification
- weak or risky match -> no match

## Return Statuses

The service returns one of these statuses:

- `invalid_input`: first and last name were not provided clearly
- `directory_unavailable`: the employee directory is not ready or failed to load
- `single_match`: one strong match
- `multiple_matches`: several strong or duplicate matches
- `needs_clarification`: one plausible match, but not safe enough to auto confirm
- `no_match`: nothing good enough to use

Successful results include up to 3 candidates, each with:

- `official_name`
- `employee_name`
- `username`
- `business_title`
- `tenure`
- `match_score`

Internal Snowflake/org fields stay cached in the directory service. They are not
sent in the lookup response; enrollment rehydrates them locally from the verified
`username`.

## Why It Is Designed This Way

This flow is meant to handle realtime transcription errors like:

- `Sakshee Patil` -> `Sakshi Patil`
- `Sakshee Patil` -> `Sakshi Patel`

while still rejecting unsafe cases like:

- `Alex Patil` matching `Sakshee Patil`

In practice, the model should ask the person to say first and last name separately
whenever the split is unclear.
