import { ShieldCheck, ShieldAlert, ShieldQuestion } from "lucide-react";
import clsx from "clsx";

interface HashBadgeProps {
  valid: boolean | null;
  loading?: boolean;
  totalEvents?: number;
  failedAt?: number | null;
  onClick?: () => void;
}

export function HashBadge({
  valid,
  loading = false,
  totalEvents,
  failedAt,
  onClick,
}: HashBadgeProps) {
  if (loading) {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs bg-gray-100 text-gray-500 animate-pulse">
        <ShieldQuestion size={14} />
        Verifying…
      </span>
    );
  }

  if (valid === null) {
    return (
      <button
        onClick={onClick}
        className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs bg-gray-100 text-gray-500 hover:bg-gray-200 transition-colors cursor-pointer"
      >
        <ShieldQuestion size={14} />
        Verify chain
      </button>
    );
  }

  if (valid) {
    return (
      <span
        className={clsx(
          "inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium",
          "bg-green-100 text-green-700"
        )}
        title={totalEvents ? `${totalEvents} events verified` : undefined}
      >
        <ShieldCheck size={14} />
        Chain intact
        {totalEvents !== undefined && (
          <span className="opacity-70">({totalEvents})</span>
        )}
      </span>
    );
  }

  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium",
        "bg-red-100 text-red-700"
      )}
      title={failedAt !== null ? `Chain broken at sequence #${failedAt}` : "Integrity check failed"}
    >
      <ShieldAlert size={14} />
      TAMPERED
      {failedAt !== null && failedAt !== undefined && (
        <span className="opacity-70">(seq #{failedAt})</span>
      )}
    </span>
  );
}
