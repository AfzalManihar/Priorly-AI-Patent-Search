"""Local administrator commands for Priorly accounts.

Run from backend/: python -m app.admin reset-password USERNAME
                    python -m app.admin delete-user USERNAME
                    python -m app.admin claim-legacy USERNAME --yes
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import secrets
import sys

from .config import Settings
from .store import Store

N, R, P = 2**14, 8, 1


def _hash(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=N, r=R, p=P, dklen=64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Priorly account administration")
    sub = parser.add_subparsers(dest="command", required=True)
    reset = sub.add_parser("reset-password", help="reset an account password")
    reset.add_argument("username")
    delete = sub.add_parser("delete-user", help="delete an account and revoke its sessions")
    delete.add_argument("username")
    claim = sub.add_parser("claim-legacy", help="give searches/reports saved before login existed to one user")
    claim.add_argument("username")
    claim.add_argument("--yes", action="store_true", help="confirm the ownership change")
    args = parser.parse_args(argv)

    store = Store(Settings.from_env().db_path)
    try:
        if args.command == "reset-password":
            first = getpass.getpass("New password (minimum 12 characters): ")
            second = getpass.getpass("Confirm new password: ")
            if len(first) < 12 or len(first) > 1024:
                print("Password must be between 12 and 1024 characters.", file=sys.stderr)
                return 2
            if not secrets.compare_digest(first, second):
                print("Passwords do not match.", file=sys.stderr)
                return 2
            salt = secrets.token_bytes(16)
            if not store.set_user_password(args.username.lower(), salt, _hash(first, salt)):
                print("User not found.", file=sys.stderr)
                return 1
            print("Password reset. All existing sessions have been revoked.")
            return 0
        if args.command == "delete-user":
            if not store.delete_user(args.username.lower()):
                print("User not found.", file=sys.stderr)
                return 1
            print("User deleted; associated sessions revoked.")
            return 0
        if args.command == "claim-legacy":
            user = store.user_by_username(args.username.lower())
            if not user:
                print("User not found.", file=sys.stderr)
                return 1
            if not args.yes:
                print("Add --yes to assign every ownerless search and report to this user. Nothing was changed.")
                return 2
            moved = store.claim_legacy(user["id"])
            print(f"Assigned {moved['searches']} searches and {moved['reports']} reports to {user['username']}.")
            return 0
    finally:
        store.db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
