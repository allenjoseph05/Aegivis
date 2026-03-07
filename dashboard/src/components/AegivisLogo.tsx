import { useId } from "react";

interface AegivisLogoProps {
  size?: number;
  className?: string;
}

export function AegivisLogo({ size = 28, className }: AegivisLogoProps) {
  const uid = useId().replace(/:/g, "");
  const topId = `al-top-${uid}`;
  const leftId = `al-left-${uid}`;
  const rightId = `al-right-${uid}`;

  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 100 116"
      width={size}
      height={Math.round(size * 116 / 100)}
      className={className}
      aria-label="Aegivis"
    >
      <defs>
        <linearGradient id={topId} x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#00C4FF" />
          <stop offset="100%" stopColor="#29B0FF" />
        </linearGradient>
        <linearGradient id={leftId} x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#1E8FFF" />
          <stop offset="100%" stopColor="#0060D0" />
        </linearGradient>
        <linearGradient id={rightId} x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#0052CC" />
          <stop offset="100%" stopColor="#003AB0" />
        </linearGradient>
      </defs>
      {/* Top face */}
      <polygon points="50,4 94,28 50,52 6,28" fill={`url(#${topId})`} />
      {/* Left face */}
      <polygon points="6,28 50,52 50,100 6,76" fill={`url(#${leftId})`} />
      {/* Right face */}
      <polygon points="50,52 94,28 94,76 50,100" fill={`url(#${rightId})`} />
      {/* Center void hex */}
      <polygon points="50,36 64,44 64,60 50,68 36,60 36,44" fill="white" opacity="0.9" />
    </svg>
  );
}
