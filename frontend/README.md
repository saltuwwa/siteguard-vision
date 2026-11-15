# SiteGuard Vision frontend

Independent, desktop-first React monitoring workspace. All application changes are contained in this directory; no Python, model, dataset, or backend dependency files are changed.

## Run

Requires Node.js 22.12+ (verified with 22.16.0) or supported Node.js 24, and npm.

```powershell
cd frontend
npm ci
npm run dev
```

Open **http://127.0.0.1:5173**. Mock mode is the default and needs no backend. The deterministic snapshot is **15 November 2026, 16:24:08 UTC+05**. Refresh reloads this snapshot; it does not manufacture a live timeline. Overview, Monitoring, Cameras, Violations, Analytics and Settings are implemented.

```powershell
npm run typecheck
npm run lint
npm test
npm run test:e2e
npm run format:check
npm run build
npm run preview
```

The browser suite uses installed Chrome by default. Alternatively run `npx playwright install chromium`, set `PLAYWRIGHT_CHANNEL=chromium`, and run the suite. Screenshots are regenerated under `docs/screenshots/`. `npm run preview` serves the production build at **http://127.0.0.1:4173**.

If npm 10 reports `Cannot read properties of null (reading 'edgesOut')` when changing dependency versions, use `npx --yes npm@11.6.2 install`. The resolved lockfile is included for `npm ci`. A narrowly scoped override updates the lint parser's transitive minimatch dependency to 11.0.1; lint and TypeScript checks validate compatibility.

## Demo interactions

- Switch cameras in the monitor or the overview header. Offline/unknown sources have explicit fallback states.
- Change the overview time range; metrics, recent events and chart buckets use the same range.
- Toggle detection overlays, enter fullscreen, or load an optional recorded MP4.
- Open an event to inspect camera/session, track ID, lifecycle, source offsets, observations, model version and review state. Accept/reject changes are in memory and reset on reload.
- Violations has its own URL-backed camera, review, lifecycle, search, date-range and page filters. Share or reload the URL to retain the view.
- Settings exposes normal, empty, error and delayed-loading scenarios. Error has retry and restore actions. Returning to Settings remains possible after an error.
- Sidebar collapse, responsive navigation, profile placeholder and keyboard-accessible dialogs work. Account setup, camera connection management and production review writes are explicitly unavailable.

## Recorded video and streaming

Copy `.env.example` to `.env.local` and optionally set:

```dotenv
VITE_DEMO_VIDEO_URL=/media/my-demo.mp4
```

Place your own MP4 in `public/media/`, which is gitignored, and restart Vite. Playback, pause, seeking and fullscreen use the browser video element. Media must be browser-decodable (for example H.264 MP4). Broken video sources fall back to the illustrative preview; broken preview images show a useful source fallback. No video is bundled, and no live connectivity is claimed.

`VideoViewer` receives typed camera data and normalized `[x, y, width, height]` detections. A ResizeObserver and the intrinsic media size project boxes into the contained image area, including letterboxing and fullscreen. Demo boxes are separate HTML overlays. The still's fixed detections are **not** drawn over arbitrary MP4 footage: no synchronized metadata is available for that clip. The exported `StreamingAdapter` interface reserves attach/detach integration for a future WebRTC/HLS adapter. RTSP cannot be played directly in this UI. Never place RTSP passwords, backend tokens or private camera URLs in `VITE_*` variables; Vite values are public client configuration.

## API integration

```dotenv
VITE_DATA_MODE=api
VITE_API_BASE_URL=/api
```

API mode never substitutes mock data on failure. Restart Vite after configuration changes. The client uses same-origin credentials. Deploy a same-origin authenticated reverse proxy/BFF that strips `/api` and adds the backend's bearer token on the server. That proxy is an infrastructure prerequisite, not implemented by this frontend. The existing API has no CORS middleware and requires a token for protected routes, so a direct cross-origin browser request is not a working default integration.

The adapter was checked against `src/siteguard/api/app.py`, `src/siteguard/events/schema.py`, `src/siteguard/storage/database.py`, `src/siteguard/rules/temporal.py` and `src/siteguard/pipeline.py` present in this workspace on 15 November 2026.

- `GET /health` returns liveness; `GET /ready` covers API/database readiness only. Typed helpers are exported, but neither is presented as proof of camera connectivity.
- `GET /cameras` returns an array of `cameraid`, `streamsessionid`, `lastseen`, `metrics`. IDs map directly; the API supplies no display name, site, location, detections, viewing URL or reliable stream-online field. Connectivity remains **unknown**. `lastseen` is a worker heartbeat, not a verified camera frame timestamp. The Cameras page labels it “Last frame / heartbeat”. FPS is shown only if a numeric `metrics.processed_fps` exists; current heartbeat payloads may omit it.
- `GET /events?limit=200&offset=N` returns `{total, limit, offset, items}` and supports `camera_id`. It has no review/status/date/search filter. The adapter traverses pages, deduplicates by `eventid`, and applies view filters locally. Pagination is not an atomic database snapshot; concurrent ingestion can change results. For large deployments, coordinate a server-filtered, cursor-based contract before scaling history beyond practical client memory.
- `GET /events/{event_id}` is used by the API-mode detail drawer. Compact backend names map explicitly to camelCase UI fields. `confidencesummary.mean` is the only mean confidence displayed; missing means remain unavailable.
- `createdat` drives event wall-clock display/filtering and chart buckets. Source timestamps remain seconds within the stream session. They are never interpreted as Unix wall-clock timestamps.
- `status` (`confirmed`, `cleared`, `ended`) is separate from `reviewstatus` (`pending`, `accepted`, `rejected`). All persisted events passed temporal confirmation; a cleared or rejected event retains that event lifecycle history.
- Filesystem evidence references are retained as metadata, **never converted to public image URLs**. No evidence-serving route is implemented, so the drawer shows an unavailable state. Future signed evidence URLs must be mapped explicitly.
- `GET /metrics` provides outbox counts and cameras, not the dashboard's proposed KPI contract. A typed helper preserves this response; camera heartbeat availability is not treated as uptime, active browser streams, GPU usage or model accuracy.
- There is no review-write route, camera configuration route, frame-metadata route, streaming route, evidence route or audit-history route. Those capabilities remain unavailable in API mode. User identity remains a marked placeholder.

## Metric definitions

Definitions are available through focusable help icons in the KPI strip.

- **Monitored cameras:** camera records returned by the selected view, including offline/unknown states.
- **Active streams:** simulated online sources in demo mode. Unavailable in API mode until source connectivity is defined.
- **Visible workers:** distinct camera/session tracks in the current sample. Not global unique people. Unavailable without track metadata in API mode.
- **Confirmed violations:** persisted temporally confirmed events created in the selected period, including ended/cleared events; not raw detection frames.
- **Violation rate:** current `VIOLATION` tracks divided by known-PPE tracks. Unknown is excluded. No known denominator means unavailable, not 0%.
- **Unreviewed events:** selected events with `reviewstatus=pending`.

The PPE ring's colored segments show the current sample's compliant, violation and unknown tracks; its numeric compliance uses **known tracks only**. Availability is a current snapshot, not historical uptime. All chart axes count events and show time in UTC+05. “Current shift” means the 12 hours ending at the displayed snapshot/fetch time; no shift calendar is supplied by the API.

## Structure

`src/app` owns navigation and workspace state. `components/ui` contains small reusable primitives and the accessible Radix dialog integration; `components/video` owns media and coordinate transforms. `features` contains route-level views, `services/mock` centralizes reproducible data, `services/api` isolates the existing backend mapping, `hooks` owns TanStack Query integration, and `types` is the shared domain layer. Secondary routes are lazy loaded; charts and React are split into reusable production chunks. No inference or raw stream processing occurs in React.

## Media attribution

The bundled 310 KB illustrative still is [Sim Kimhort's construction photograph on Unsplash](https://unsplash.com/photos/t5j-XCZDSfw), used under the [Unsplash license](https://unsplash.com/license). It is an interface preview, not a captured camera frame or safety-event evidence. Both demo cameras intentionally reuse it; their identities and detection metadata are simulated. No large media, node_modules, build artifacts, camera credentials or tokens should be committed.

See [verification](docs/verification.md) and [screenshots](docs/screenshots/overview-1440x900.png).
