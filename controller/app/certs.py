"""Self-signed TLS certificate generation for the admission webhook server.

Generates a CA and a server certificate signed by that CA.
After generation, patches the MutatingWebhookConfiguration with the base64
CA bundle so the K8s API server trusts the webhook.
"""
from __future__ import annotations

import base64
import datetime
import ipaddress
import logging
import os
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

logger = logging.getLogger(__name__)

_ONE_YEAR = datetime.timedelta(days=365)


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def generate_certs(
    cert_dir: str,
    service_name: str,
    namespace: str,
) -> tuple[Path, Path, bytes]:
    """Generate CA and server cert/key. Returns (cert_path, key_path, ca_der_bytes)."""
    out = Path(cert_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── CA key + cert ─────────────────────────────────────────────────────────
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_name = x509.Name([
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Aegivis Controller CA"),
        x509.NameAttribute(NameOID.COMMON_NAME, "aegivis-ca"),
    ])
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(_utcnow())
        .not_valid_after(_utcnow() + _ONE_YEAR)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(ca_key, hashes.SHA256())
    )
    ca_der = ca_cert.public_bytes(serialization.Encoding.DER)

    # ── Server key + cert ─────────────────────────────────────────────────────
    # SANs: service DNS names + localhost for local dev
    dns_names = [
        x509.DNSName(service_name),
        x509.DNSName(f"{service_name}.{namespace}"),
        x509.DNSName(f"{service_name}.{namespace}.svc"),
        x509.DNSName(f"{service_name}.{namespace}.svc.cluster.local"),
        x509.DNSName("localhost"),
    ]
    san = x509.SubjectAlternativeName(
        dns_names + [x509.IPAddress(ipaddress.IPv4Address("127.0.0.1"))]
    )

    srv_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    srv_name = x509.Name([
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Aegivis"),
        x509.NameAttribute(
            NameOID.COMMON_NAME,
            f"{service_name}.{namespace}.svc.cluster.local",
        ),
    ])
    srv_cert = (
        x509.CertificateBuilder()
        .subject_name(srv_name)
        .issuer_name(ca_name)
        .public_key(srv_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(_utcnow())
        .not_valid_after(_utcnow() + _ONE_YEAR)
        .add_extension(san, critical=False)
        .sign(ca_key, hashes.SHA256())
    )

    cert_path = out / "tls.crt"
    key_path  = out / "tls.key"

    cert_path.write_bytes(srv_cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        srv_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    os.chmod(key_path, 0o600)

    logger.info(
        "Generated TLS certs: cert=%s key=%s (SANs: %s)",
        cert_path, key_path,
        [str(n) for n in dns_names],
    )
    return cert_path, key_path, ca_der


def patch_webhook_ca_bundle(
    webhook_config_name: str,
    ca_der: bytes,
) -> None:
    """Patch the MutatingWebhookConfiguration with our CA bundle."""
    try:
        from kubernetes import client as k8s_client, config as k8s_config

        try:
            k8s_config.load_incluster_config()
        except Exception:
            k8s_config.load_kube_config()

        ca_b64 = base64.b64encode(ca_der).decode()
        admission_api = k8s_client.AdmissionregistrationV1Api()

        # Get current webhook count so we can patch all entries
        cfg = admission_api.read_mutating_webhook_configuration(webhook_config_name)
        patches = [
            {
                "op": "replace",
                "path": f"/webhooks/{i}/clientConfig/caBundle",
                "value": ca_b64,
            }
            for i in range(len(cfg.webhooks or []))
        ]
        if patches:
            admission_api.patch_mutating_webhook_configuration(
                webhook_config_name,
                patches,
            )
            logger.info(
                "Patched MutatingWebhookConfiguration '%s' with CA bundle (%d webhooks)",
                webhook_config_name, len(patches),
            )
        else:
            logger.warning(
                "MutatingWebhookConfiguration '%s' has no webhooks to patch",
                webhook_config_name,
            )

    except Exception as exc:
        # Non-fatal: webhook will be called with caBundle from values.yaml if pre-set,
        # or will fail TLS verification. Log prominently so operators notice.
        logger.warning(
            "Could not patch MutatingWebhookConfiguration '%s': %s — "
            "admission webhook TLS may fail until CA bundle is set manually.",
            webhook_config_name, exc,
        )
