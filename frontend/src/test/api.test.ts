import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiError, loadApiData, request } from '../services/api/client';
afterEach(() => vi.unstubAllGlobals());
describe('API request states', () => {
  it('provides an actionable authentication error', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{}', { status: 401 })));
    await expect(request('/cameras')).rejects.toThrow('Authentication required');
  });
  it('reports readiness and invalid JSON without falling back to demo data', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockResolvedValueOnce(new Response('{}', { status: 503 }))
        .mockResolvedValueOnce(new Response('<html>error</html>', { status: 200 })),
    );
    await expect(request('/ready')).rejects.toBeInstanceOf(ApiError);
    await expect(request('/cameras')).rejects.toThrow('invalid JSON');
  });
  it('passes cancellation and same-origin credentials to fetch', async () => {
    const fetch = vi.fn().mockResolvedValue(new Response('{"status":"ok"}'));
    vi.stubGlobal('fetch', fetch);
    const signal = new AbortController().signal;
    await request('/health', signal);
    expect(fetch).toHaveBeenCalledWith(
      '/api/health',
      expect.objectContaining({ signal, credentials: 'same-origin' }),
    );
  });
  it('loads actual event pages and handles empty service data', async () => {
    const fetch = vi
      .fn()
      .mockResolvedValueOnce(new Response('[]'))
      .mockResolvedValueOnce(new Response('{"total":0,"limit":200,"offset":0,"items":[]}'));
    vi.stubGlobal('fetch', fetch);
    const result = await loadApiData(new AbortController().signal);
    expect(result).toMatchObject({ demo: false, cameras: [], events: [], totalEvents: 0 });
  });
  it('rejects an unsupported response envelope', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValueOnce(new Response('{}')).mockResolvedValueOnce(new Response('[]')),
    );
    await expect(loadApiData(new AbortController().signal)).rejects.toThrow(
      'Unsupported API response shape',
    );
  });
});
