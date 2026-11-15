import { useQuery } from '@tanstack/react-query';
import { loadApiData } from '../services/api/client';
import { mockData } from '../services/mock/data';
import type { DashboardData, DemoScenario } from '../types';
export const isMock = import.meta.env.VITE_DATA_MODE !== 'api';
export function useDashboard(scenario: DemoScenario) {
  return useQuery<DashboardData>({
    queryKey: ['dashboard', isMock, scenario],
    queryFn: async ({ signal }) => {
      if (!isMock) return loadApiData(signal);
      await new Promise<void>((resolve, reject) => {
        const timer = setTimeout(resolve, scenario === 'loading' ? 2500 : 220);
        signal.addEventListener(
          'abort',
          () => {
            clearTimeout(timer);
            reject(new DOMException('Aborted', 'AbortError'));
          },
          { once: true },
        );
      });
      if (scenario === 'error')
        throw new Error('Demo connection interrupted. Choose Normal in Settings or retry.');
      return scenario === 'empty'
        ? { ...mockData, cameras: [], events: [], totalEvents: 0 }
        : mockData;
    },
    retry: false,
    staleTime: 30000,
    refetchOnWindowFocus: false,
  });
}
