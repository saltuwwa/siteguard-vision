export type CameraStatus = 'online' | 'offline' | 'unknown';
export type PPEStatus = 'COMPLIANT' | 'VIOLATION' | 'UNKNOWN';
export type EventReviewStatus = 'pending' | 'accepted' | 'rejected';
export type EventStatus = 'confirmed' | 'cleared' | 'ended';
export interface Detection {
  id: string;
  trackId: number;
  box: [number, number, number, number];
  ppe: PPEStatus;
  confidence: number | null;
}
export interface PersonTrack {
  id: number;
  cameraId: string;
  sessionId: string;
  ppe: PPEStatus;
}
export interface Camera {
  id: string;
  name: string;
  location: string;
  site: string;
  status: CameraStatus;
  sessionId: string;
  lastFrame: string | null;
  fps: number | null;
  source: 'demo' | 'unavailable';
  detections: Detection[];
  videoUrl?: string;
  previewUrl?: string;
}
export interface ViolationEvent {
  id: string;
  cameraId: string;
  sessionId: string;
  trackId: number;
  createdAt: string;
  start: number;
  end: number | null;
  confirmedAt: number;
  status: EventStatus;
  review: EventReviewStatus;
  confidence: number | null;
  observations: { timestamp: number; confidence: number }[];
  observationCount: number;
  modelVersion: string;
  evidenceUrl: string | null;
  evidenceReference: string | null;
  endReason: string | null;
  demo: boolean;
}
export interface MonitoringMetrics {
  monitoredCameras: number;
  activeStreams: number | null;
  visibleWorkers: number | null;
  confirmedViolations: number;
  violationRate: number | null;
  unreviewedEvents: number;
}
export interface DashboardData {
  cameras: Camera[];
  events: ViolationEvent[];
  asOf: string;
  demo: boolean;
  totalEvents: number;
}
export type DemoScenario = 'normal' | 'empty' | 'error' | 'loading';
export type TimeRange = '1h' | '4h' | 'shift';
