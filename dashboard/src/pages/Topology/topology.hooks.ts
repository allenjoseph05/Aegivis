/**
 * React Query hooks for the Agent Topology page.
 */
import { useQuery } from "@tanstack/react-query";
import { getTopology } from "../../api/client";
import type { TopologyFilters, TopologyGraph } from "./topology.types";

/** Auto-refreshes every 30 seconds. */
export function useTopology(filters: TopologyFilters) {
  return useQuery<TopologyGraph>({
    queryKey: ["topology", filters],
    queryFn: () => getTopology(filters),
    refetchInterval: 30_000,
    staleTime: 10_000,
  });
}
