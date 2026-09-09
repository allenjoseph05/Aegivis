import { render, screen, fireEvent } from "@testing-library/react";
import { EventTimeline } from "./EventTimeline";
import type { AuditEvent } from "../types/events";

function makeEvent(i: number): AuditEvent {
  return {
    event_id: `evt-${i}`,
    schema_version: "1",
    session_id: "sess-1",
    org_id: "default-org",
    agent_id: "agent-1",
    provider: "openai",
    model: "gpt-4o",
    interception_layer: "proxy",
    run_id: "run-1",
    sequence_number: i,
    event_type: "LLM_CALL_START",
    payload: {},
    pii_detected: [],
    timestamp_ns: Date.now() * 1_000_000 + i,
    previous_hash: i === 0 ? "0".repeat(64) : `hash-${i - 1}`,
    current_hash: `hash-${i}`,
  } as unknown as AuditEvent;
}

describe("EventTimeline", () => {
  it("shows loading skeletons when loading=true", () => {
    const { container } = render(<EventTimeline events={[]} loading={true} />);
    const skeletons = container.querySelectorAll(".animate-pulse");
    expect(skeletons.length).toBeGreaterThan(0);
  });

  it("shows empty state when no events", () => {
    render(<EventTimeline events={[]} />);
    expect(screen.getByText(/no events found/i)).toBeInTheDocument();
  });

  it("renders all events when count <= 100", () => {
    const events = Array.from({ length: 10 }, (_, i) => makeEvent(i));
    render(<EventTimeline events={events} />);
    expect(screen.queryByText(/more event/i)).not.toBeInTheDocument();
  });

  it("shows 'Show N more events' button when count > 100", () => {
    const events = Array.from({ length: 130 }, (_, i) => makeEvent(i));
    render(<EventTimeline events={events} />);
    expect(screen.getByText(/show 30 more events/i)).toBeInTheDocument();
  });

  it("reveals all events when 'Show more' is clicked", () => {
    const events = Array.from({ length: 110 }, (_, i) => makeEvent(i));
    render(<EventTimeline events={events} />);
    const btn = screen.getByText(/show 10 more event/i);
    fireEvent.click(btn);
    expect(screen.queryByText(/show.*more event/i)).not.toBeInTheDocument();
  });
});
