"""Self-signed TLS client certificate for the Mumble bot.

Mumble identifies clients by their TLS cert fingerprint. Without a cert,
per-user *local* settings on remote Mumble clients (volume, mute, comment)
and *server-side* registrations don't survive a reconnect — Mumble shows the
"无法永久保存,因为 <用户> 没有证书" warning. Generating a stable self-signed cert
once and reusing it makes the bot a "real" Mumble user.

The cert/key live under .memory/identity/ as PEM files, keyed by user name.
"""
from __future__ import annotations

import datetime
import os
import re
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from .memory import DEFAULT_MEMORY_ROOT

DEFAULT_IDENTITY_DIR = DEFAULT_MEMORY_ROOT / "identity"


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name) or "user"


def ensure_cert(name: str, base_dir: Path | None = None) -> tuple[Path, Path]:
    """Return (cert_pem_path, key_pem_path), creating them if absent.

    RSA-2048 + SHA-256, valid 10 years. Mumble only cares about the cert
    fingerprint (stable for the cert's lifetime), not CA chain or expiry
    enforcement on the client side.
    """
    base = (base_dir or DEFAULT_IDENTITY_DIR).resolve()
    base.mkdir(parents=True, exist_ok=True)
    slug = _safe_filename(name)
    cert_path = base / f"{slug}.crt"
    key_path = base / f"{slug}.key"
    if cert_path.exists() and key_path.exists():
        return cert_path, key_path

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, name),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "LivePartner"),
    ])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True, key_encipherment=True,
                content_commitment=False, data_encipherment=False,
                key_agreement=False, key_cert_sign=False, crl_sign=False,
                encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .sign(private_key=key, algorithm=hashes.SHA256())
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    try:
        os.chmod(key_path, 0o600)
    except OSError:
        pass
    return cert_path, key_path


def cert_fingerprint_sha256(cert_path: Path) -> str:
    """Hex SHA-256 fingerprint (colon-separated, lowercase) of a PEM cert.

    Useful for letting the user verify which identity is being presented to
    the server (matches what `mumble.exe` shows in 配置→网络→证书).
    """
    pem = cert_path.read_bytes()
    c = x509.load_pem_x509_certificate(pem)
    fp = c.fingerprint(hashes.SHA256()).hex()
    return ":".join(fp[i:i + 2] for i in range(0, len(fp), 2))
