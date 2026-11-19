# Priorly Profile & Privacy

Adds a Profile & Privacy navigation view directly below Pipeline settings.

- Profile fields (email, display name, current learning topic) and memory preference are persisted per account.
- Password changes verify the current password, require the existing 12-character account password policy, and revoke all sessions.
- Sign out everywhere revokes all sessions for the account and clears the current browser cookies.
- Existing SQLite databases are migrated in place with safe defaults; no `.env` changes are required.

The memory checkbox currently persists the user's preference. This codebase does not yet include a memory retrieval/storage subsystem, so the preference is not connected to an active memory engine.
