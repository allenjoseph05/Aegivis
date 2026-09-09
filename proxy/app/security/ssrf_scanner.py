"""
Server-Side Request Forgery (SSRF) scanner for tool call arguments.

Detects URL values in tool arguments that target private/internal networks,
cloud metadata endpoints, or other SSRF-sensitive addresses.

Detection layers (applied in order):
  1. URL extraction    — find all URL-like strings in the argument dict.
  2. Normalisation     — decode URL-encoding, resolve non-standard IP formats
                         (decimal, octal, hex) that bypass naive IP checks.
  3. Private-range check — use Python's stdlib ``ipaddress`` module to check
                           whether the resolved host falls in any reserved range.
  4. Cloud-metadata check — hard-block well-known cloud metadata endpoints
                            regardless of allowlist (169.254.169.254 etc.).
  5. Allowlist check   — configured via ``AEGIVIS_SECURITY_SSRF_ALLOWED_DOMAINS``.
                         If a non-empty allowlist is set, any host not on it
                         is blocked.  Empty allowlist = audit-only mode.
  6. DNS resolution    — optionally resolve the hostname to its IP address(es)
                         and re-apply the private-range check.  This catches
                         DNS-rebinding attacks where an attacker uses a
                         public-looking hostname that resolves to 10.x.x.x.

All IP comparisons use ``ipaddress.ip_network`` containment — no string
manipulation or regex IP matching, which is the canonical evasion surface.
"""
from __future__ import annotations

import ipaddress
import logging
import re
import socket
import urllib.parse
from dataclasses import dataclass, field
from functools import lru_cache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private / reserved IP ranges  (RFC 1918, RFC 6890, and cloud-specific)
# ---------------------------------------------------------------------------

_PRIVATE_NETWORKS: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = (
    # IPv4
    ipaddress.ip_network("0.0.0.0/8"),          # "This" network
    ipaddress.ip_network("10.0.0.0/8"),          # RFC 1918 private
    ipaddress.ip_network("100.64.0.0/10"),       # Shared address space (CGNAT)
    ipaddress.ip_network("127.0.0.0/8"),         # Loopback
    ipaddress.ip_network("169.254.0.0/16"),      # Link-local / cloud metadata
    ipaddress.ip_network("172.16.0.0/12"),       # RFC 1918 private
    ipaddress.ip_network("192.0.0.0/24"),        # IETF Protocol Assignments
    ipaddress.ip_network("192.0.2.0/24"),        # TEST-NET-1 (documentation)
    ipaddress.ip_network("192.168.0.0/16"),      # RFC 1918 private
    ipaddress.ip_network("198.18.0.0/15"),       # Benchmarking (RFC 2544)
    ipaddress.ip_network("198.51.100.0/24"),     # TEST-NET-2 (documentation)
    ipaddress.ip_network("203.0.113.0/24"),      # TEST-NET-3 (documentation)
    ipaddress.ip_network("240.0.0.0/4"),         # Reserved
    ipaddress.ip_network("255.255.255.255/32"),  # Broadcast
    # IPv6
    ipaddress.ip_network("::1/128"),             # Loopback
    ipaddress.ip_network("fc00::/7"),            # Unique local
    ipaddress.ip_network("fe80::/10"),           # Link-local
    ipaddress.ip_network("::ffff:0:0/96"),       # IPv4-mapped
    ipaddress.ip_network("64:ff9b::/96"),        # IPv4/IPv6 translation
)

# Cloud instance metadata endpoints — always blocked regardless of allowlist.
# These endpoints serve IAM credentials and must never be reachable from agents.
_CLOUD_METADATA_HOSTS: frozenset[str] = frozenset({
    "169.254.169.254",           # AWS / GCP / Azure / DigitalOcean IMDS
    "metadata.google.internal",  # GCP metadata
    "169.254.170.2",             # ECS task metadata (AWS Fargate)
    "fd00:ec2::254",             # AWS IPv6 IMDS
    "metadata.internal",         # generic internal metadata alias
})

# Regex to find URL-like strings within any text value.
# Scope: URL DISCOVERY only — this is a preprocessing step to find candidates,
# not the threat detection itself.  Detection is done by IP analysis below.
_URL_RE = re.compile(
    r'(?:https?|ftp|ftps|sftp|file|gopher|dict|ldap|ldaps|tftp)://'
    r'[^\s\'"<>()\[\]{},;\\]+',
    re.IGNORECASE,
)

# Protocol-relative URLs  (//host/path)
_PROTO_REL_RE = re.compile(r'(?<!\w)//([^/\s\'"<>()]+)')

# Non-standard IP representations used to bypass naive string-based checks.
_HEX_IP_RE    = re.compile(r'^0x[0-9a-f]+$', re.IGNORECASE)
_OCTAL_IP_RE  = re.compile(r'^0[0-7]+$')
_DECIMAL_IP_RE = re.compile(r'^\d+$')   # single 32-bit integer (e.g. 2130706433)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SsrfMatch:
    url:        str    # sanitised URL for logging (no credentials embedded)
    host:       str    # host component
    reason:     str    # human-readable explanation
    confidence: float  # 0.0-1.0


@dataclass
class SsrfScanResult:
    detected:     bool
    matches:      list[SsrfMatch] = field(default_factory=list)
    urls_scanned: int = 0

    @property
    def ssrf_category(self) -> str:
        """
        Highest-priority SSRF category across all matches.

        Returns one of:
          'cloud_metadata'      — targets IAM credential endpoint (highest risk)
          'not_in_allowlist'    — host not on the configured allowlist
          'private_ip'          — RFC 1918 / reserved direct IP address
          'resolves_to_private' — public hostname resolves to private IP
          'none'                — no matches (detected is False)

        Used by policy rules to distinguish always-block (cloud metadata) from
        review-and-alert (enterprise internal API calls on private subnets).
        """
        if not self.matches:
            return "none"
        _priority = {
            "cloud_metadata_endpoint": 4,
            "not_in_allowlist":        3,
            "private_ip":              2,
            "resolves_to_private":     1,
        }
        best_cat, best_p = "none", 0
        for m in self.matches:
            # reason format: "category:detail" or just "category"
            cat = m.reason.split(":")[0] if ":" in m.reason else m.reason
            p = _priority.get(cat, 0)
            if p > best_p:
                best_cat, best_p = cat, p
        # Normalise the cloud_metadata prefix to a clean key
        if best_cat == "cloud_metadata_endpoint":
            return "cloud_metadata"
        return best_cat


# ---------------------------------------------------------------------------
# IP address parsing (handles non-standard formats)
# ---------------------------------------------------------------------------

def _try_parse_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """
    Parse *host* as an IP address, handling non-standard representations:
      - Standard dotted-decimal:  "192.168.1.1"
      - Hexadecimal integer:      "0xc0a80101"
      - Octal integer:            "0300.0250.0001.0001"  (not valid Python oct)
      - Decimal integer:          "3232235777"
      - IPv6 in square brackets:  "[::1]"

    Returns None if *host* is not a recognisable IP address (i.e., it is a
    hostname and needs DNS resolution).
    """
    # Strip IPv6 brackets
    stripped = host.strip("[]")

    # Standard parse (handles dotted-decimal, IPv6)
    try:
        return ipaddress.ip_address(stripped)
    except ValueError:
        pass

    # Hex integer (0xC0A80101 = 192.168.1.1)
    if _HEX_IP_RE.match(stripped):
        try:
            return ipaddress.ip_address(int(stripped, 16))
        except ValueError:
            pass

    # Decimal integer (3232235777 = 192.168.1.1)
    if _DECIMAL_IP_RE.match(stripped) and len(stripped) >= 7:
        try:
            val = int(stripped)
            if 0 <= val <= 0xFFFFFFFF:
                return ipaddress.ip_address(val)
        except ValueError:
            pass

    # Octal integer (0177 = 127 in octal)
    if _OCTAL_IP_RE.match(stripped):
        try:
            return ipaddress.ip_address(int(stripped, 8))
        except ValueError:
            pass

    return None


def _is_private(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True when *addr* falls within any private/reserved network."""
    for net in _PRIVATE_NETWORKS:
        try:
            if addr in net:
                return True
        except TypeError:
            continue
    return False


# ---------------------------------------------------------------------------
# DNS resolution  (cached, synchronous — safe to call from a thread)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=512)
def _resolve_host(hostname: str) -> tuple[str, ...]:
    """
    Resolve *hostname* to IP address strings.
    Results are cached in an LRU cache (process lifetime, no TTL).
    Returns an empty tuple on resolution failure.

    Note: TTL-aware caching would be ideal to defend against DNS-rebinding
    within a cache window.  LRU is used here for simplicity; Phase 3.4 can
    upgrade to a TTL-aware cache.
    """
    try:
        results = socket.getaddrinfo(
            hostname, None,
            socket.AF_UNSPEC, socket.SOCK_STREAM,
        )
        return tuple({r[4][0] for r in results})
    except (socket.gaierror, OSError):
        return ()


# ---------------------------------------------------------------------------
# URL analysis
# ---------------------------------------------------------------------------

def _sanitise_url(url: str) -> str:
    """Remove embedded credentials from URL for safe logging."""
    try:
        parsed = urllib.parse.urlparse(url)
        # Rebuild without username/password
        safe = parsed._replace(netloc=parsed.hostname or "")
        return urllib.parse.urlunparse(safe)
    except Exception:
        return url[:120]


def _extract_host_port(url: str) -> tuple[str, int | None] | None:
    """
    Parse *url* and return (host, port) or None on parse failure.
    Applies URL-decoding to catch encoded bypass attempts.
    """
    # Decode URL encoding (catches http://127%2e0%2e0%2e1/)
    decoded = urllib.parse.unquote(url)
    try:
        parsed = urllib.parse.urlparse(decoded)
        host = parsed.hostname
        port = parsed.port
        if not host:
            return None
        return host.lower().strip("."), port
    except Exception:
        return None


def _check_url(
    url: str,
    *,
    allowed_domains: frozenset[str],
    enable_dns: bool,
) -> SsrfMatch | None:
    """
    Analyse a single URL.  Returns SsrfMatch when the URL is suspicious,
    None when it is safe (or unparseable).
    """
    host_port = _extract_host_port(url)
    if not host_port:
        return None
    host, _port = host_port
    safe_url = _sanitise_url(url)

    # 1. Cloud metadata endpoint — always blocked, highest priority
    if host in _CLOUD_METADATA_HOSTS:
        return SsrfMatch(
            url=safe_url, host=host,
            reason=f"cloud_metadata_endpoint:{host}",
            confidence=1.0,
        )

    # 2. Allowlist check — if configured, any host not in the list is suspicious
    if allowed_domains:
        host_allowed = any(
            host == d or host.endswith("." + d)
            for d in allowed_domains
        )
        if not host_allowed:
            return SsrfMatch(
                url=safe_url, host=host,
                reason=f"not_in_allowlist:{host}",
                confidence=0.70,
            )

    # 3. Direct IP check — handle non-standard formats
    addr = _try_parse_ip(host)
    if addr is not None:
        if _is_private(addr):
            return SsrfMatch(
                url=safe_url, host=host,
                reason=f"private_ip:{addr}",
                confidence=0.95,
            )
        # Public IP — not inherently suspicious (unless blocklisted)
        return None

    # 4. DNS resolution — resolve hostname and check resulting IPs
    if enable_dns:
        resolved_ips = _resolve_host(host)
        for ip_str in resolved_ips:
            addr = _try_parse_ip(ip_str)
            if addr is not None and _is_private(addr):
                return SsrfMatch(
                    url=safe_url, host=host,
                    reason=f"resolves_to_private:{ip_str}",
                    confidence=0.88,
                )

    return None


# ---------------------------------------------------------------------------
# URL extraction from tool args
# ---------------------------------------------------------------------------

def _extract_urls(args: object, _depth: int = 0) -> list[str]:
    """Recursively extract URL strings from tool argument structures."""
    if _depth > 8:
        return []
    urls: list[str] = []
    if isinstance(args, str):
        urls.extend(_URL_RE.findall(args))
        for m in _PROTO_REL_RE.finditer(args):
            urls.append("http://" + m.group(1))
    elif isinstance(args, dict):
        for val in args.values():
            urls.extend(_extract_urls(val, _depth + 1))
    elif isinstance(args, list):
        for item in args:
            urls.extend(_extract_urls(item, _depth + 1))
    return urls


def _parse_allowed_domains(raw: str) -> frozenset[str]:
    """Parse comma-separated domain list from settings string."""
    if not raw or not raw.strip():
        return frozenset()
    return frozenset(d.strip().lower() for d in raw.split(",") if d.strip())


# ---------------------------------------------------------------------------
# Public scanner
# ---------------------------------------------------------------------------

def scan(tool_args: dict | str) -> SsrfScanResult:
    """
    Scan tool call arguments for SSRF-targetted URLs.

    Args:
        tool_args: Tool argument dict (or raw string if JSON parsing failed).

    Returns:
        SsrfScanResult.  ``detected`` is True when any URL targets a
        private/reserved/metadata address.
    """
    from ..config import settings

    allowed_domains = _parse_allowed_domains(settings.security_ssrf_allowed_domains)
    enable_dns      = settings.security_ssrf_enable_dns_resolution

    urls = _extract_urls(tool_args)
    if not urls:
        return SsrfScanResult(detected=False, matches=[], urls_scanned=0)

    # Deduplicate
    seen: set[str] = set()
    unique_urls = [u for u in urls if not (u in seen or seen.add(u))]  # type: ignore[func-returns-value]

    matches: list[SsrfMatch] = []
    for url in unique_urls:
        match = _check_url(url, allowed_domains=allowed_domains, enable_dns=enable_dns)
        if match:
            matches.append(match)
            logger.warning(
                "[SECURITY] SSRF target detected: url=%r host=%r reason=%s confidence=%.2f",
                match.url, match.host, match.reason, match.confidence,
            )

    return SsrfScanResult(
        detected=len(matches) > 0,
        matches=matches,
        urls_scanned=len(unique_urls),
    )
