/**
 * useWebSocket — connects to the AgentBlackBox real-time alert stream.
 *
 * Automatically reconnects every 3 seconds on disconnect.
 * Keeps the last 50 messages in state.
 */
import { useCallback, useEffect, useRef, useState } from "react";

const WS_URL =
  import.meta.env.VITE_WS_URL ||
  `ws://${window.location.hostname}:8000/ws/alerts`;

const MAX_MESSAGES = 50;
const RECONNECT_DELAY_MS = 3000;

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
  const unmounted = useRef(false);

  const connect = useCallback(() => {
    if (unmounted.current) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      if (!unmounted.current) setConnected(true);
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
      // Schedule reconnect
      reconnectTimer.current = setTimeout(connect, RECONNECT_DELAY_MS);
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
