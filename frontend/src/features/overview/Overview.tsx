import {
  ArrowUpRight,
  Camera,
  CheckCheck,
  CircleHelp,
  ScanLine,
  ShieldAlert,
  Video,
  Users,
} from 'lucide-react';
import { Link } from 'react-router-dom';
import { useWorkspace } from '../../app/context';
import { Badge, Empty, Panel } from '../../components/ui';
import { VideoViewer } from '../../components/video/VideoViewer';
import { metrics, percent } from '../../lib/domain';
import { EventTable } from '../violations/EventTable';
import { TrendChart } from '../analytics/Charts';
export function Kpis() {
  const { data, filtered, cameraId } = useWorkspace();
  const cameras = data.cameras.filter((c) => cameraId === 'all' || c.id === cameraId),
    m = metrics(cameras, filtered, data.demo);
  const values = [
    {
      label: 'Monitored cameras',
      value: m.monitoredCameras,
      icon: Camera,
      definition: 'Cameras returned by the service, including offline and unknown states.',
    },
    {
      label: 'Active streams',
      value: m.activeStreams,
      icon: Video,
      definition:
        'Simulated online sources. Actual browser stream availability is not supplied by the current API.',
    },
    {
      label: 'Visible workers',
      value: m.visibleWorkers,
      icon: Users,
      definition:
        'Distinct camera/session track IDs in the current sample, not unique people across cameras or time.',
    },
    {
      label: 'Confirmed violations',
      value: m.confirmedViolations,
      icon: ShieldAlert,
      definition:
        'Temporally confirmed events in the selected camera and period, including ended and cleared events.',
    },
    {
      label: 'Violation rate',
      value: percent(m.violationRate),
      icon: ScanLine,
      definition:
        'Current tracked workers with VIOLATION PPE / workers with known PPE. Unknown tracks excluded; not an event-based rate.',
    },
    {
      label: 'Unreviewed events',
      value: m.unreviewedEvents,
      icon: CheckCheck,
      definition: 'Events with human review status pending in the selected period.',
    },
  ];
  return (
    <div className="kpi-grid">
      {values.map(({ label, value, icon: Icon, definition }) => (
        <div className="kpi" key={label}>
          <div className="kpi-label">
            <Icon size={15} />
            <span>{label}</span>
            <button
              className="definition"
              aria-label={`${label}: ${definition}`}
              title={definition}
            >
              <CircleHelp size={12} />
            </button>
          </div>
          <strong>{value ?? '—'}</strong>
          <small>
            {data.demo ? 'Simulated data' : value === null ? 'Not supplied' : 'Selected period'}
          </small>
        </div>
      ))}
    </div>
  );
}
export function PpeSummary() {
  const { selectedCamera, data } = useWorkspace();
  const detections = selectedCamera?.detections || [];
  const counts = {
    COMPLIANT: detections.filter((d) => d.ppe === 'COMPLIANT').length,
    VIOLATION: detections.filter((d) => d.ppe === 'VIOLATION').length,
    UNKNOWN: detections.filter((d) => d.ppe === 'UNKNOWN').length,
  };
  const known = counts.COMPLIANT + counts.VIOLATION,
    rate = known ? Math.round((counts.COMPLIANT / known) * 100) : null;
  return (
    <Panel
      title="PPE in current view"
      action={<Badge>{data.demo ? 'Demo sample' : 'Unavailable'}</Badge>}
    >
      <div className="ppe-body">
        <div
          className="compliance-ring"
          style={{
            background: `conic-gradient(var(--green) 0 ${detections.length ? (counts.COMPLIANT / detections.length) * 100 : 0}%, var(--red) 0 ${detections.length ? (known / detections.length) * 100 : 0}%, #526070 0 100%)`,
          }}
        >
          <div>
            <strong>{rate === null ? '—' : `${rate}%`}</strong>
            <span>Known PPE compliant</span>
          </div>
        </div>
        <div className="ppe-legend">
          {[
            ['COMPLIANT', 'Helmet', 'green'],
            ['VIOLATION', 'No helmet', 'red'],
            ['UNKNOWN', 'Unknown', 'neutral'],
          ].map(([key, label, tone]) => (
            <div key={key}>
              <span>
                <i className={`legend-dot ${tone}-dot`} />
                {label}
              </span>
              <strong>{data.demo ? counts[key as keyof typeof counts] : '—'}</strong>
            </div>
          ))}
        </div>
      </div>
      <p className="panel-footnote">Unknown PPE excluded from compliance. Current tracks only.</p>
    </Panel>
  );
}
export function CameraList() {
  const { data, setCameraId } = useWorkspace();
  return (
    <Panel
      title="Camera status"
      action={
        <Link className="text-link" to="/cameras">
          Manage <ArrowUpRight size={13} />
        </Link>
      }
    >
      {!data.cameras.length ? (
        <Empty title="No cameras configured" />
      ) : (
        <div className="camera-list">
          {data.cameras.map((c) => (
            <Link to="/monitoring" onClick={() => setCameraId(c.id)} key={c.id}>
              <Camera size={16} />
              <span>
                <strong>{c.name}</strong>
                <small>{c.id}</small>
              </span>
              <span className={`status-text ${c.status}`}>
                <i />
                {c.status}
              </span>
            </Link>
          ))}
        </div>
      )}
      <p className="panel-footnote">
        {data.demo
          ? 'Simulated connectivity · 4 configured cameras'
          : 'Camera connectivity is not reported by the API.'}
      </p>
    </Panel>
  );
}
export default function Overview() {
  const { data, selectedCamera, setCameraId, filtered, openEvent, range } = useWorkspace();
  return (
    <>
      <Kpis />
      <div className="overview-grid">
        <div className="monitor-column">
          {selectedCamera ? (
            <VideoViewer
              key={selectedCamera.id}
              camera={selectedCamera}
              cameras={data.cameras}
              onCameraChange={setCameraId}
              timestamp={data.asOf}
              demo={data.demo}
            />
          ) : (
            <Panel title="Camera preview">
              <Empty
                title="No cameras configured"
                detail="Camera metadata will appear here when a worker registers with the backend."
              />
            </Panel>
          )}
          <Panel
            title="Recent violations"
            action={
              <Link className="text-link" to="/violations">
                View all <ArrowUpRight size={14} />
              </Link>
            }
          >
            <div className="event-subheading">
              <span>Temporally confirmed helmet events</span>
              <Badge tone="amber">
                {filtered.filter((e) => e.review === 'pending').length} unreviewed
              </Badge>
            </div>
            <EventTable
              events={filtered.slice(0, 4)}
              cameras={data.cameras}
              onSelect={openEvent}
              compact
            />
          </Panel>
        </div>
        <aside className="insights-column">
          <PpeSummary />
          <Panel
            title="Violations over time"
            action={
              <span className="muted text-xs">
                {range === 'shift' ? 'Current shift' : `Last ${range}`}
              </span>
            }
          >
            <TrendChart events={filtered} asOf={data.asOf} range={range} />
          </Panel>
          <CameraList />
        </aside>
      </div>
    </>
  );
}
