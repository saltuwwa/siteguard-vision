import { useWorkspace } from '../../app/context';
import { Badge, Empty, Panel } from '../../components/ui';
import { CameraChart, TrendChart } from './Charts';
import { Kpis } from '../overview/Overview';
export default function Analytics() {
  const { data, filtered, range, cameraId } = useWorkspace();
  const cameras = data.cameras.filter((c) => cameraId === 'all' || c.id === cameraId);
  return (
    <>
      <Kpis />
      <div className="analytics-grid">
        <Panel title="Confirmed events over time" action={<Badge>Events</Badge>}>
          <TrendChart events={filtered} asOf={data.asOf} range={range} />
        </Panel>
        <Panel title="Alerts by camera">
          <CameraChart events={filtered} />
        </Panel>
        <Panel title="Event lifecycle">
          <div className="breakdown">
            {(['confirmed', 'ended', 'cleared'] as const).map((status) => (
              <div key={status}>
                <span>
                  {status === 'confirmed'
                    ? 'Confirmed / open'
                    : status === 'ended'
                      ? 'Ended'
                      : 'Cleared'}
                </span>
                <strong>{filtered.filter((e) => e.status === status).length}</strong>
                <progress
                  aria-label={`${status} events`}
                  value={filtered.filter((e) => e.status === status).length}
                  max={filtered.length || 1}
                />
              </div>
            ))}
          </div>
          <p className="panel-footnote">
            All lifecycle states originated from a temporally confirmed event.
          </p>
        </Panel>
        <Panel
          title="Camera availability"
          action={<Badge>{data.demo ? 'Demo snapshot' : 'Unknown'}</Badge>}
        >
          {cameras.length ? (
            <div className="breakdown">
              {(['online', 'offline', 'unknown'] as const).map((status) => (
                <div key={status}>
                  <span>{status}</span>
                  <strong>{cameras.filter((c) => c.status === status).length}</strong>
                  <progress
                    aria-label={`${status} cameras`}
                    value={cameras.filter((c) => c.status === status).length}
                    max={cameras.length || 1}
                  />
                </div>
              ))}
            </div>
          ) : (
            <Empty title="No camera metadata" />
          )}
          <p className="panel-footnote">
            Current source status, not historical uptime. The API does not provide availability
            history.
          </p>
        </Panel>
        <Panel title="Human review">
          <div className="breakdown">
            {(['pending', 'accepted', 'rejected'] as const).map((review) => (
              <div key={review}>
                <span>{review === 'pending' ? 'Unreviewed' : review}</span>
                <strong>{filtered.filter((e) => e.review === review).length}</strong>
                <progress
                  aria-label={`${review} events`}
                  value={filtered.filter((e) => e.review === review).length}
                  max={filtered.length || 1}
                />
              </div>
            ))}
          </div>
        </Panel>
        <Panel title="How to read these metrics">
          <div className="explanation">
            <h3>Events, observations and people</h3>
            <p>
              One event contains repeated observations of a track. Observation counts do not
              represent unique people.
            </p>
            <h3>Unknown remains unknown</h3>
            <p>
              Occlusion and uncertain PPE classification do not contribute to confirmed violations
              or the known-PPE denominator.
            </p>
            <h3>Selected period</h3>
            <p>
              Events are grouped by their creation time. Source offsets in event details are seconds
              within the recording, not wall-clock times.
            </p>
          </div>
        </Panel>
      </div>
    </>
  );
}
