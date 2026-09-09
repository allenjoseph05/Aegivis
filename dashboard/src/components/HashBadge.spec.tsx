import { render, screen, fireEvent } from "@testing-library/react";
import { HashBadge } from "./HashBadge";

describe("HashBadge", () => {
  it("shows loading state", () => {
    render(<HashBadge valid={null} loading={true} />);
    expect(screen.getByText("Verifying…")).toBeInTheDocument();
  });

  it("shows verify button when valid is null and not loading", () => {
    const onClick = vi.fn();
    render(<HashBadge valid={null} onClick={onClick} />);
    const btn = screen.getByRole("button", { name: /verify chain/i });
    expect(btn).toBeInTheDocument();
    fireEvent.click(btn);
    expect(onClick).toHaveBeenCalledOnce();
  });

  it("shows chain intact when valid is true", () => {
    render(<HashBadge valid={true} totalEvents={42} />);
    expect(screen.getByText(/chain intact/i)).toBeInTheDocument();
    expect(screen.getByText("(42)")).toBeInTheDocument();
  });

  it("shows TAMPERED when valid is false", () => {
    render(<HashBadge valid={false} failedAt={7} />);
    expect(screen.getByText(/tampered/i)).toBeInTheDocument();
    expect(screen.getByText("(seq #7)")).toBeInTheDocument();
  });

  it("shows TAMPERED without sequence number when failedAt is null", () => {
    render(<HashBadge valid={false} failedAt={null} />);
    expect(screen.getByText(/tampered/i)).toBeInTheDocument();
    expect(screen.queryByText(/seq #/)).not.toBeInTheDocument();
  });
});
