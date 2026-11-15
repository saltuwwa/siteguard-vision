import * as Dialog from '@radix-ui/react-dialog';
import { useQuery } from '@tanstack/react-query';
import { ImageOff, ShieldCheck, X } from 'lucide-react';
import { api } from '../../services/api/client';
import { Badge, Button } from '../../components/ui';
import { clock, dateLabel, duration, percent } from '../../lib/domain';
import type { ViolationEvent } from '../../types';
export function EventDrawer({
  id,
  events,
  demo,
  onClose,
  onReview,
}: {
  id: string | null;
  events: ViolationEvent[];
  demo: boolean;
  onClose: () => void;
  onReview: (id: string, review: 'accepted' | 'rejected') => void;
}) {
  const detail = useQuery({
    queryKey: ['event', id],
    queryFn: ({ signal }) => api.event(id!, signal),
    enabled: !demo && Boolean(id),
    retry: false,
  });
  const event = demo ? events.find((e) => e.id === id) : detail.data;
  return (
    <Dialog.Root
      open={Boolean(id)}
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <Dialog.Portal>
        <Dialog.Overlay className="dialog-overlay" />
        <Dialog.Content className="event-drawer">
          <div className="drawer-heading">
            <div>
              <span className="eyebrow">Safety event</span>
              <Dialog.Title>{id}</Dialog.Title>
            </div>
            <Dialog.Close asChild>
              <Button className="icon-button" aria-label="Close event details">
                <X size={20} />
              </Button>
            </Dialog.Close>
          </div>
          <Dialog.Description className="drawer-description">
            Detection evidence, source timing and review history.
          </Dialog.Description>
          {!event ? (
            <div className="empty" role="status">
              {detail.isError ? 'Event details could not be loaded.' : 'Loading event details…'}
              {detail.isError && <Button onClick={() => void detail.refetch()}>Retry</Button>}
            </div>
          ) : (
            <>
              <div className="event-summary">
                <TriangleLabel />
                <Badge tone={event.review === 'pending' ? 'amber' : 'green'}>{event.review}</Badge>
                {demo && <Badge tone="amber">DEMO</Badge>}
              </div>
              <div className="evidence-preview">
                {event.evidenceUrl ? (
                  <img src={event.evidenceUrl} alt={`Evidence for ${event.id}`} />
                ) : (
                  <>
                    <ImageOff size={28} />
                    <strong>No evidence snapshot available</strong>
                    <p>
                      {demo
                        ? 'This simulated event has no real evidence.'
                        : 'The API provides a storage reference, not a public image URL.'}
                    </p>
                  </>
                )}
              </div>
              <dl className="detail-grid">
                <dt>Camera</dt>
                <dd>{event.cameraId}</dd>
                <dt>Session</dt>
                <dd>{event.sessionId}</dd>
                <dt>Person track</dt>
                <dd>
                  #{event.trackId} <small>Within this session only</small>
                </dd>
                <dt>Recorded at</dt>
                <dd>
                  {dateLabel(event.createdAt)} · {clock(event.createdAt)} UTC+05
                </dd>
                <dt>Source start</dt>
                <dd>{event.start.toFixed(2)} s</dd>
                <dt>Source end</dt>
                <dd>{event.end === null ? 'Still open' : `${event.end.toFixed(2)} s`}</dd>
                <dt>Duration</dt>
                <dd>{duration(event.start, event.end)}</dd>
                <dt>Lifecycle</dt>
                <dd>{event.status}</dd>
                <dt>Mean confidence</dt>
                <dd>{percent(event.confidence)}</dd>
                <dt>Model version</dt>
                <dd>{event.modelVersion}</dd>
                <dt>End reason</dt>
                <dd>{event.endReason || '—'}</dd>
              </dl>
              <h3>Supporting observations</h3>
              <p className="muted">
                {event.observationCount} observations recorded; {event.observations.length}{' '}
                supplied.
              </p>
              <table className="observation-table">
                <thead>
                  <tr>
                    <th>Source time</th>
                    <th>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {event.observations.map((o, i) => (
                    <tr key={i}>
                      <td>{o.timestamp.toFixed(2)} s</td>
                      <td>{percent(o.confidence)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="review-section">
                <h3>
                  <ShieldCheck size={17} /> Human review
                </h3>
                <p>
                  {demo
                    ? 'Review changes apply to this demo session only.'
                    : 'Review updates require a backend write endpoint.'}
                </p>
                <div className="flex gap-2">
                  <Button
                    disabled={!demo || event.review === 'accepted'}
                    onClick={() => onReview(event.id, 'accepted')}
                  >
                    Accept event
                  </Button>
                  <Button
                    disabled={!demo || event.review === 'rejected'}
                    onClick={() => onReview(event.id, 'rejected')}
                  >
                    Reject event
                  </Button>
                </div>
                <small>
                  Audit:{' '}
                  {demo
                    ? `Simulated operator · current review: ${event.review}`
                    : 'No review audit trail supplied by API'}
                </small>
              </div>
            </>
          )}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
function TriangleLabel() {
  return <strong className="text-red">No helmet detected</strong>;
}
