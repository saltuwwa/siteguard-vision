import { ChevronRight, ImageOff, TriangleAlert } from 'lucide-react';
import { Badge, Empty } from '../../components/ui';
import { clock, duration, percent } from '../../lib/domain';
import type { Camera, ViolationEvent } from '../../types';
export function EventTable({
  events,
  cameras,
  onSelect,
  compact = false,
}: {
  events: ViolationEvent[];
  cameras: Camera[];
  onSelect: (id: string) => void;
  compact?: boolean;
}) {
  if (!events.length) return <Empty />;
  return (
    <div className="table-scroll">
      <table className={compact ? 'event-table compact' : 'event-table'}>
        <caption className="sr-only">
          Safety violations, newest first. Open an event for details.
        </caption>
        <thead>
          <tr>
            <th>Event / time</th>
            <th>Camera</th>
            <th>Violation</th>
            {!compact && <th>Lifecycle</th>}
            <th>Review</th>
            <th>
              <span className="sr-only">Open event</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {events.map((e) => (
            <tr key={e.id}>
              <td>
                <button className="event-open" onClick={() => onSelect(e.id)}>
                  <span className="evidence-thumb">
                    <ImageOff size={16} />
                  </span>
                  <span>
                    <strong>{clock(e.createdAt)}</strong>
                    <small>{e.id}</small>
                  </span>
                </button>
              </td>
              <td>
                <strong>{e.cameraId}</strong>
                <small>{cameras.find((c) => c.id === e.cameraId)?.name || 'Unmapped camera'}</small>
              </td>
              <td>
                <span className="violation-name">
                  <TriangleAlert size={15} /> No helmet
                </span>
                <small>
                  {duration(e.start, e.end)} · {percent(e.confidence)} confidence
                </small>
              </td>
              {!compact && (
                <td>
                  <Badge tone={e.status === 'confirmed' ? 'red' : 'neutral'}>{e.status}</Badge>
                </td>
              )}
              <td>
                <Badge
                  tone={
                    e.review === 'pending' ? 'amber' : e.review === 'accepted' ? 'green' : 'neutral'
                  }
                >
                  {e.review === 'pending'
                    ? 'Unreviewed'
                    : e.review === 'accepted'
                      ? 'Accepted'
                      : 'Rejected'}
                </Badge>
              </td>
              <td>
                <button
                  className="row-action"
                  aria-label={`Open event ${e.id}`}
                  onClick={() => onSelect(e.id)}
                >
                  <ChevronRight size={17} />
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
