/**
 * useWebSocket — connects to the Aegivis real-time alert stream.
 *
 * Reconnects with exponential backoff (1s → 2s → 4s → … → 30s cap)
 * after disconnect. Resets backoff delay on successful connection.
 * Keeps the last 50 messages in state.
 */
import { useCallback, useEffect, useRef, useState } from "react";

const WS_URL =
  import.meta.env.VITE_WS_URL ||
  `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/ws/alerts`;

const MAX_MESSAGES = 50;
const RECONNECT_BASE_MS  = 1_000;   // initial delay
const RECONNECT_MAX_MS   = 30_000;  // cap

export interface AlertMessage {
  id: string;
  type: "violation" | "anomaly" | string;
  session_id?: string;
  agent_id?: string;
  rule_id?: string;
  rule_name?: string;
  severity?: string;
  description?: string;
  reason?: string;
  action?: string;
  receivedAt: Date;
}

export interface UseWebSocketReturn {
  messages: AlertMessage[];
  connected: boolean;
  clear: () => void;
}

export function useWebSocket(): UseWebSocketReturn {
  const [messages, setMessages] = useState<AlertMessage[]>([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const retryDelay = useRef(RECONNECT_BASE_MS);
  const unmounted = useRef(false);

  const connect = useCallback(() => {
    if (unmounted.current) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      if (!unmounted.current) {
        setConnected(true);
        retryDelay.current = RECONNECT_BASE_MS;  // reset backoff on success
      }
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data as string);
        const msg: AlertMessage = {
          id: crypto.randomUUID(),
          receivedAt: new Date(),
          ...data,
        };
        setMessages((prev) => [msg, ...prev].slice(0, MAX_MESSAGES));
      } catch {
        // ignore unparseable frames
      }
    };

    ws.onclose = () => {
      if (unmounted.current) return;
      setConnected(false);
      // Exponential backoff: 1s → 2s → 4s → … → 30s
      reconnectTimer.current = setTimeout(connect, retryDelay.current);
      retryDelay.current = Math.min(retryDelay.current * 2, RECONNECT_MAX_MS);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, []);

  useEffect(() => {
    unmounted.current = false;
    connect();
    return () => {
      unmounted.current = true;
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  const clear = useCallback(() => setMessages([]), []);

  return { messages, connected, clear };
}
