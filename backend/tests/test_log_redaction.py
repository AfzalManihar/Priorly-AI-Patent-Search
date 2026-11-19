import io
import logging

from uvicorn.logging import AccessFormatter

import app.main  # noqa: F401  installs the redacting log-record factory


def _logger(fmt=None):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(fmt or logging.Formatter("%(message)s"))
    log = logging.getLogger("priorly.test.redaction")
    log.handlers[:] = [handler]
    log.propagate = False
    log.setLevel(logging.INFO)
    return log, stream


def test_uvicorn_access_log_still_has_its_five_arguments():
    """Regression: the redacting factory once set record.args to (), and uvicorn's AccessFormatter then failed with
    'not enough values to unpack (expected 5, got 0)' on every request."""
    fmt = AccessFormatter('%(client_addr)s - "%(request_line)s" %(status_code)s', use_colors=False)
    log, stream = _logger(fmt)
    log.info('%s - "%s %s HTTP/%s" %d', "127.0.0.1:5000", "GET", "/api/usage", "1.1", 200)
    assert stream.getvalue().strip() == '127.0.0.1:5000 - "GET /api/usage HTTP/1.1" 200 OK'


def test_secrets_are_redacted_in_message_and_arguments():
    log, stream = _logger()
    log.info("key=%s and gsk_abcdefghij1234567890 in the text", "gsk_zzzzzzzzzz9999999999")
    out = stream.getvalue()
    assert "gsk_" not in out and "[redacted]" in out


def test_percent_formatting_with_numbers_is_unchanged():
    log, stream = _logger()
    log.info("rerank: %d of %d patents in %.1f s", 5, 20, 45.2)
    assert stream.getvalue().strip() == "rerank: 5 of 20 patents in 45.2 s"


def test_auth_secrets_are_redacted():
    from app.guardrails import redact

    value = 'password="plain-password" priorly_session=opaque-session csrf_token=csrf-secret invite_code=invite-secret'
    cleaned = redact(value)
    for secret in ("plain-password", "opaque-session", "csrf-secret", "invite-secret"):
        assert secret not in cleaned
    assert cleaned.count("[redacted]") >= 4
