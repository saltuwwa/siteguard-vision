import type { Camera, MonitoringMetrics, TimeRange, ViolationEvent } from '../types';
export const TIME_ZONE = 'Asia/Qyzylorda';
export function clock(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.valueOf())
    ? 'Unavailable'
    : new Intl.DateTimeFormat('en-GB', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        timeZone: TIME_ZONE,
      }).format(date);
}
export function dateLabel(value: string) {
  return new Intl.DateTimeFormat('en-GB', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    timeZone: TIME_ZONE,
  }).format(new Date(value));
}
export function duration(start: number, end: number | null) {
  return end === null ? 'Open' : `${Math.max(0, end - start).toFixed(1)} s`;
}
export function percent(value: number | null) {
  return value === null ? '—' : `${(value * 100).toFixed(1)}%`;
}
export function filterEvents(
  events: ViolationEvent[],
  filters: {
    camera?: string;
    review?: string;
    status?: string;
    range?: TimeRange;
    asOf?: string;
    search?: string;
  },
) {
  const hours = filters.range === '1h' ? 1 : filters.range === '4h' ? 4 : 12;
  const end = filters.asOf ? new Date(filters.asOf).valueOf() : Infinity;
  return events.filter(
    (e) =>
      (!filters.camera || filters.camera === 'all' || e.cameraId === filters.camera) &&
      (!filters.review || filters.review === 'all' || e.review === filters.review) &&
      (!filters.status || filters.status === 'all' || e.status === filters.status) &&
      (!filters.asOf ||
        (new Date(e.createdAt).valueOf() >= end - hours * 3600000 &&
          new Date(e.createdAt).valueOf() <= end)) &&
      (!filters.search ||
        `${e.id} ${e.trackId} ${e.cameraId}`.toLowerCase().includes(filters.search.toLowerCase())),
  );
}
export function metrics(
  cameras: Camera[],
  events: ViolationEvent[],
  demo: boolean,
): MonitoringMetrics {
  const detections = cameras.flatMap((c) => c.detections);
  const known = detections.filter((d) => d.ppe !== 'UNKNOWN');
  return {
    monitoredCameras: cameras.length,
    activeStreams: demo ? cameras.filter((c) => c.status === 'online').length : null,
    visibleWorkers: demo ? detections.length : null,
    confirmedViolations: events.length,
    violationRate:
      demo && known.length
        ? known.filter((d) => d.ppe === 'VIOLATION').length / known.length
        : null,
    unreviewedEvents: events.filter((e) => e.review === 'pending').length,
  };
}
export function containRect(
  width: number,
  height: number,
  sourceWidth: number,
  sourceHeight: number,
) {
  if ([width, height, sourceWidth, sourceHeight].some((v) => !Number.isFinite(v) || v <= 0))
    return { x: 0, y: 0, width: 0, height: 0 };
  const scale = Math.min(width / sourceWidth, height / sourceHeight);
  const w = sourceWidth * scale,
    h = sourceHeight * scale;
  return { x: (width - w) / 2, y: (height - h) / 2, width: w, height: h };
}
export function projectBox(
  box: [number, number, number, number],
  rect: ReturnType<typeof containRect>,
) {
  const clamp = (n: number) => Math.min(1, Math.max(0, n));
  const [x, y, w, h] = box;
  return {
    left: rect.x + clamp(x) * rect.width,
    top: rect.y + clamp(y) * rect.height,
    width: Math.min(clamp(w), 1 - clamp(x)) * rect.width,
    height: Math.min(clamp(h), 1 - clamp(y)) * rect.height,
  };
}
