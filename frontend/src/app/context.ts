import { createContext, useContext } from 'react';
import type { Camera, DashboardData, DemoScenario, TimeRange, ViolationEvent } from '../types';
export interface Workspace {
  data: DashboardData;
  events: ViolationEvent[];
  filtered: ViolationEvent[];
  cameraId: string;
  setCameraId: (id: string) => void;
  range: TimeRange;
  setRange: (range: TimeRange) => void;
  selectedCamera: Camera | undefined;
  openEvent: (id: string) => void;
  scenario: DemoScenario;
  setScenario: (value: DemoScenario) => void;
  refresh: () => void;
}
export const WorkspaceContext = createContext<Workspace | null>(null);
export function useWorkspace() {
  const context = useContext(WorkspaceContext);
  if (!context) throw new Error('Workspace provider missing');
  return context;
}
