import type { Camera, DashboardData, ViolationEvent } from '../../types';
export const DEMO_AS_OF = '2026-11-15T11:24:08Z';
export const preview = '/demo-construction.jpg';
const cameras: Camera[] = [
  {
    id: 'CAM-01',
    name: 'Building Site A',
    location: 'North structure · Level 02',
    site: 'Riverside construction',
    status: 'online',
    sessionId: 'demo-session-01',
    lastFrame: DEMO_AS_OF,
    fps: 24,
    source: 'demo',
    previewUrl: preview,
    videoUrl: import.meta.env.VITE_DEMO_VIDEO_URL || undefined,
    detections: [
      { id: 'd142', trackId: 142, box: [0.12, 0.06, 0.3, 0.5], ppe: 'COMPLIANT', confidence: 0.98 },
      { id: 'd143', trackId: 143, box: [0.1, 0.36, 0.59, 0.63], ppe: 'UNKNOWN', confidence: null },
    ],
  },
  {
    id: 'CAM-02',
    name: 'Loading Zone B',
    location: 'East access · Ground floor',
    site: 'Riverside construction',
    status: 'online',
    sessionId: 'demo-session-02',
    lastFrame: DEMO_AS_OF,
    fps: 22,
    source: 'demo',
    previewUrl: preview,
    detections: [
      { id: 'd201', trackId: 201, box: [0.12, 0.06, 0.3, 0.5], ppe: 'COMPLIANT', confidence: 0.96 },
      { id: 'd202', trackId: 202, box: [0.1, 0.36, 0.59, 0.63], ppe: 'UNKNOWN', confidence: null },
    ],
  },
  {
    id: 'CAM-03',
    name: 'South Perimeter',
    location: 'South gate · Exterior',
    site: 'Riverside construction',
    status: 'offline',
    sessionId: 'demo-session-03',
    lastFrame: '2026-11-15T10:58:32Z',
    fps: null,
    source: 'demo',
    detections: [],
  },
  {
    id: 'CAM-04',
    name: 'Material Storage',
    location: 'West compound · Storage',
    site: 'Riverside construction',
    status: 'unknown',
    sessionId: 'demo-session-04',
    lastFrame: null,
    fps: null,
    source: 'demo',
    detections: [],
  },
];
const minutes = [1, 7, 21, 34, 49, 68, 86, 101, 119, 142, 164, 183, 205, 226, 257, 288, 326, 371];
const events: ViolationEvent[] = minutes.map((minute, i) => {
  const start = 1220 + i * 43.3;
  return {
    id: `SG-${String(1048 - i).padStart(5, '0')}`,
    cameraId: i % 3 === 0 ? 'CAM-02' : 'CAM-01',
    sessionId: i % 3 === 0 ? 'demo-session-02' : 'demo-session-01',
    trackId: 144 + i,
    createdAt: new Date(new Date(DEMO_AS_OF).valueOf() - minute * 60000).toISOString(),
    start,
    end: i === 0 ? null : start + 5.2 + (i % 5),
    confirmedAt: start + 1.5,
    status: i === 0 ? 'confirmed' : i % 4 === 0 ? 'cleared' : 'ended',
    review: i < 5 ? 'pending' : i % 6 === 0 ? 'rejected' : 'accepted',
    confidence: 0.94 + (i % 4) * 0.01,
    observations: [
      { timestamp: start, confidence: 0.94 },
      { timestamp: start + 0.5, confidence: 0.96 },
      { timestamp: start + 1.5, confidence: 0.97 },
    ],
    observationCount: 3,
    modelVersion: 'demo-ppe-v1 · simulated',
    evidenceUrl: null,
    evidenceReference: null,
    endReason: i === 0 ? null : 'Track left field of view',
    demo: true,
  };
});
export const mockData: DashboardData = {
  cameras,
  events,
  asOf: DEMO_AS_OF,
  demo: true,
  totalEvents: events.length,
};
