import type {
  Camera,
  DashboardData,
  EventReviewStatus,
  EventStatus,
  ViolationEvent,
} from '../../types';
export interface ApiCamera {
  cameraid: string;
  streamsessionid: string;
  lastseen: string;
  metrics: Record<string, unknown>;
}
export interface ApiEvent {
  eventid: string;
  cameraid: string;
  streamsessionid: string;
  trackid: number;
  createdat: string;
  starttimestamp: number;
  endtimestamp: number | null;
  confirmedtimestamp: number;
  status: EventStatus;
  reviewstatus: EventReviewStatus;
  modelversion: string;
  confidencesummary: Record<string, number>;
  supportingobservations: { timestamp: number; confidence: number }[];
  observationcount: number;
  evidenceimagepath: string | null;
  evidencevideopath: string | null;
  endreason: string | null;
}
interface EventPage {
  total: number;
  limit: number;
  offset: number;
  items: ApiEvent[];
}
export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}
const base = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/$/, '');
export async function request<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${base}${path}`, {
    signal,
    credentials: 'same-origin',
    headers: { Accept: 'application/json' },
  });
  if (!response.ok)
    throw new ApiError(
      response.status === 401
        ? 'Authentication required. Configure the server-side API proxy.'
        : response.status === 503
          ? 'The API is not ready. Check the backend configuration.'
          : `Request failed (${response.status}).`,
      response.status,
    );
  try {
    return (await response.json()) as T;
  } catch {
    throw new ApiError('The API returned an invalid JSON response.', response.status);
  }
}
export function mapCamera(c: ApiCamera): Camera {
  return {
    id: c.cameraid,
    name: c.cameraid,
    site: 'Configured site',
    location: 'Location not provided',
    sessionId: c.streamsessionid,
    lastFrame: c.lastseen,
    status: 'unknown',
    fps: typeof c.metrics.processed_fps === 'number' ? c.metrics.processed_fps : null,
    source: 'unavailable',
    detections: [],
  };
}
export function mapEvent(e: ApiEvent): ViolationEvent {
  return {
    id: e.eventid,
    cameraId: e.cameraid,
    sessionId: e.streamsessionid,
    trackId: e.trackid,
    createdAt: e.createdat,
    start: e.starttimestamp,
    end: e.endtimestamp,
    confirmedAt: e.confirmedtimestamp,
    status: e.status,
    review: e.reviewstatus,
    confidence: typeof e.confidencesummary.mean === 'number' ? e.confidencesummary.mean : null,
    observations: e.supportingobservations,
    observationCount: e.observationcount,
    modelVersion: e.modelversion,
    evidenceUrl: null,
    evidenceReference: e.evidenceimagepath,
    endReason: e.endreason,
    demo: false,
  };
}
export async function loadApiData(signal: AbortSignal): Promise<DashboardData> {
  const [cameras, page] = await Promise.all([
    request<ApiCamera[]>('/cameras', signal),
    request<EventPage>('/events?limit=200&offset=0', signal),
  ]);
  if (!Array.isArray(cameras) || !Array.isArray(page.items) || typeof page.total !== 'number')
    throw new ApiError('Unsupported API response shape. Check the integration mapping.', 502);
  const all = [...page.items];
  // Exhaust server pages: date/status filters are not part of the existing contract.
  for (let offset = page.items.length; offset < page.total; ) {
    const next = await request<EventPage>(`/events?limit=200&offset=${offset}`, signal);
    if (!next.items.length)
      throw new ApiError('Event pagination changed while loading. Refresh to retry.', 409);
    all.push(...next.items);
    offset += next.items.length;
  }
  const unique = [...new Map(all.map((e) => [e.eventid, e])).values()];
  return {
    cameras: cameras.map(mapCamera),
    events: unique.map(mapEvent),
    totalEvents: page.total,
    asOf: new Date().toISOString(),
    demo: false,
  };
}
export const api = {
  health: (signal?: AbortSignal) => request<{ status: string }>('/health', signal),
  ready: (signal?: AbortSignal) => request<{ ready: boolean }>('/ready', signal),
  metrics: (signal?: AbortSignal) =>
    request<{ outbox: Record<string, number>; cameras: ApiCamera[] }>('/metrics', signal),
  event: async (id: string, signal?: AbortSignal) =>
    mapEvent(await request<ApiEvent>(`/events/${encodeURIComponent(id)}`, signal)),
};
