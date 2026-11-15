import { describe, expect, it } from 'vitest';
import {
  clock,
  containRect,
  duration,
  filterEvents,
  metrics,
  percent,
  projectBox,
} from '../lib/domain';
import { mockData } from '../services/mock/data';
import { mapCamera, mapEvent, type ApiEvent } from '../services/api/client';
import { timeline } from '../features/analytics/Charts';
describe('source formatting', () => {
  it('uses the selected site timezone independently of the host', () =>
    expect(clock('2026-11-15T11:24:08Z')).toBe('16:24:08'));
  it('preserves open intervals and missing confidence', () => {
    expect(duration(10, null)).toBe('Open');
    expect(duration(10, 15.2)).toBe('5.2 s');
    expect(percent(null)).toBe('—');
  });
});
describe('normalized detection transforms', () => {
  it('accounts for horizontal letterboxing', () => {
    const rect = containRect(1000, 500, 1000, 1000);
    expect(rect).toEqual({ x: 250, y: 0, width: 500, height: 500 });
    expect(projectBox([0.1, 0.2, 0.4, 0.5], rect)).toEqual({
      left: 300,
      top: 100,
      width: 200,
      height: 250,
    });
  });
  it('accounts for vertical letterboxing', () =>
    expect(containRect(800, 800, 1600, 900)).toEqual({ x: 0, y: 175, width: 800, height: 450 }));
  it('clamps boxes to source bounds', () =>
    expect(projectBox([0.9, -0.1, 0.4, 2], containRect(100, 100, 100, 100))).toEqual({
      left: 90,
      top: 0,
      width: 9.999999999999998,
      height: 100,
    }));
  it('rejects unusable sizes', () => expect(containRect(0, 100, 0, 10).width).toBe(0));
});
describe('event filters and metrics', () => {
  it('combines camera, review and source period', () => {
    const result = filterEvents(mockData.events, {
      camera: 'CAM-01',
      review: 'pending',
      range: '1h',
      asOf: mockData.asOf,
    });
    expect(result.length).toBeGreaterThan(0);
    expect(result.every((e) => e.cameraId === 'CAM-01' && e.review === 'pending')).toBe(true);
    expect(
      result.every(
        (e) => new Date(e.createdAt).valueOf() >= new Date(mockData.asOf).valueOf() - 3600000,
      ),
    ).toBe(true);
  });
  it('does not include future events', () =>
    expect(
      filterEvents([{ ...mockData.events[0], createdAt: '2030-01-01T00:00:00Z' }], {
        range: 'shift',
        asOf: mockData.asOf,
      }),
    ).toEqual([]));
  it('does not count unknown PPE as a violation', () => {
    const cameras = [
      {
        ...mockData.cameras[0],
        detections: [
          {
            id: 'x',
            trackId: 1,
            box: [0, 0, 0.5, 0.5] as [number, number, number, number],
            ppe: 'UNKNOWN' as const,
            confidence: null,
          },
        ],
      },
    ];
    expect(metrics(cameras, [], true).violationRate).toBeNull();
    expect(metrics(cameras, [], true).confirmedViolations).toBe(0);
    expect(metrics(cameras, [], false).visibleWorkers).toBeNull();
  });
  it('buckets each selected event exactly once', () => {
    const events = filterEvents(mockData.events, { range: '4h', asOf: mockData.asOf });
    expect(
      timeline(events, mockData.asOf, '4h').reduce((sum, b) => sum + b.reviewed + b.unreviewed, 0),
    ).toBe(events.length);
  });
});
describe('backend mapping', () => {
  it('does not infer live status from a heartbeat', () => {
    const camera = mapCamera({
      cameraid: 'c1',
      streamsessionid: 's1',
      lastseen: mockData.asOf,
      metrics: {},
    });
    expect(camera.status).toBe('unknown');
    expect(camera.fps).toBeNull();
    expect(camera.source).toBe('unavailable');
  });
  it('never turns a backend filesystem evidence path into an image URL', () => {
    const source: ApiEvent = {
      eventid: '1',
      cameraid: 'c1',
      streamsessionid: 's1',
      trackid: 1,
      createdat: mockData.asOf,
      starttimestamp: 0,
      endtimestamp: 3,
      confirmedtimestamp: 1,
      status: 'ended',
      reviewstatus: 'pending',
      modelversion: 'v1',
      confidencesummary: { mean: 0.95 },
      supportingobservations: [],
      observationcount: 2,
      evidenceimagepath: 'runs/evidence/1.jpg',
      evidencevideopath: null,
      endreason: null,
    };
    expect(mapEvent(source).evidenceUrl).toBeNull();
    expect(mapEvent(source).evidenceReference).toBe('runs/evidence/1.jpg');
    expect(mapEvent(source).confidence).toBe(0.95);
  });
});
