"""SQLAlchemy 2.0 ORM models for Aegivis."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Index,
    Integer,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class AuditEventORM(Base):
    """
    Append-only audit event table.
    DB-level append-only enforcement via trigger (see migration 0001).
    Application-level: never issue UPDATE or DELETE on this table.
    """
    __tablename__ = "audit_events"
    __table_args__ = (
        Index("idx_session", "session_id", "sequence_number"),
        Index("idx_org_time", "org_id", "timestamp_ns"),
        Index("idx_event_type", "org_id", "event_type", "timestamp_ns"),
        # Note: TimescaleDB hypertable partitioning is applied by migration 0008
        # via `SELECT create_hypertable(...)`. SQLAlchemy does not support the
        # postgresql_partition_by key — it would be silently ignored.
    )

    event_id = Column(Text, primary_key=True)
    schema_version = Column(Text, nullable=False, default="1.0")
    org_id = Column(Text, nullable=False)
    session_id = Column(Text, nullable=False)
    agent_id = Column(Text, nullable=False)
    provider = Column(Text, nullable=False)
    model = Column(Text, nullable=False)
    interception_layer = Column(Text, nullable=False, default="proxy")
    run_id = Column(Text, nullable=False)
    parent_run_id = Column(Text, nullable=True)
    event_type = Column(Text, nullable=False)
    payload = Column(JSONB, nullable=False)
    payload_hash = Column(Text, nullable=True)          # SHA-256 of pre-masking payload
    pii_detected = Column(ARRAY(Text), default=[])
    security = Column(JSONB, nullable=True)             # Enforcement scan result (injection_score, etc.)
    timestamp_ns = Column(BigInteger, nullable=False)
    sequence_number = Column(Integer, nullable=False)
    previous_hash = Column(Text, nullable=False)
    current_hash = Column(Text, nullable=False, unique=True)
    received_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "schema_version": self.schema_version,
            "org_id": self.org_id,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "provider": self.provider,
            "model": self.model,
            "interception_layer": self.interception_layer,
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "event_type": self.event_type,
            "payload": self.payload,
            "payload_hash": self.payload_hash,
            "pii_detected": self.pii_detected or [],
            "security": self.security,
            "timestamp_ns": self.timestamp_ns,
            "sequence_number": self.sequence_number,
            "previous_hash": self.previous_hash,
            "current_hash": self.current_hash,
            "received_at": self.received_at.isoformat() if self.received_at else None,
        }
