import { ChevronLeft, ChevronRight, Search } from 'lucide-react';
import { useSearchParams } from 'react-router-dom';
import { useWorkspace } from '../../app/context';
import { Button, Panel, Select } from '../../components/ui';
import { filterEvents } from '../../lib/domain';
import { EventTable } from './EventTable';
import type { TimeRange } from '../../types';
export default function Violations() {
  const { data, events, openEvent } = useWorkspace();
  const [params, setParams] = useSearchParams();
  const camera = params.get('camera') || 'all',
    review = params.get('review') || 'all',
    status = params.get('status') || 'all',
    search = params.get('search') || '';
  const range: TimeRange =
    params.get('range') === '1h' ? '1h' : params.get('range') === 'shift' ? 'shift' : '4h';
  const filtered = filterEvents(events, { camera, review, status, search, range, asOf: data.asOf });
  const pages = Math.max(1, Math.ceil(filtered.length / 8));
  const page = Math.max(1, Math.min(pages, Number(params.get('page')) || 1));
  function update(key: string, value: string) {
    const next = new URLSearchParams(params);
    if (value === 'all' || !value) next.delete(key);
    else next.set(key, value);
    if (key !== 'page') next.delete('page');
    setParams(next);
  }
  return (
    <Panel
      title="Violation events"
      action={<span className="muted">{filtered.length} events</span>}
    >
      <div className="table-filters">
        <label className="search-field">
          <Search size={15} />
          <input
            aria-label="Search events"
            placeholder="Event ID, track or camera"
            value={search}
            onChange={(e) => update('search', e.target.value)}
          />
        </label>
        <Select
          label="Filter events by camera"
          value={camera}
          onChange={(e) => update('camera', e.target.value)}
        >
          <option value="all">All cameras</option>
          {data.cameras.map((c) => (
            <option key={c.id}>{c.id}</option>
          ))}
        </Select>
        <Select
          label="Filter by review"
          value={review}
          onChange={(e) => update('review', e.target.value)}
        >
          <option value="all">All reviews</option>
          <option value="pending">Unreviewed</option>
          <option value="accepted">Accepted</option>
          <option value="rejected">Rejected</option>
        </Select>
        <Select
          label="Filter by lifecycle"
          value={status}
          onChange={(e) => update('status', e.target.value)}
        >
          <option value="all">All lifecycle states</option>
          <option value="confirmed">Confirmed / open</option>
          <option value="ended">Ended</option>
          <option value="cleared">Cleared</option>
        </Select>
        <Select
          label="Event date range"
          value={range}
          onChange={(e) => update('range', e.target.value)}
        >
          <option value="1h">Last hour</option>
          <option value="4h">Last 4 hours</option>
          <option value="shift">Current shift · 12 h</option>
        </Select>
        <Button onClick={() => setParams({})}>Reset</Button>
      </div>
      <EventTable
        events={filtered.slice((page - 1) * 8, page * 8)}
        cameras={data.cameras}
        onSelect={openEvent}
      />
      <div className="pagination">
        <span>
          {filtered.length ? `${(page - 1) * 8 + 1}–${Math.min(page * 8, filtered.length)}` : '0'}{' '}
          of {filtered.length} events
        </span>
        <div>
          <Button
            aria-label="Previous page"
            disabled={page === 1}
            onClick={() => update('page', String(page - 1))}
          >
            <ChevronLeft size={16} />
          </Button>
          <span>
            Page {page} of {pages}
          </span>
          <Button
            aria-label="Next page"
            disabled={page >= pages}
            onClick={() => update('page', String(page + 1))}
          >
            <ChevronRight size={16} />
          </Button>
        </div>
      </div>
      <p className="panel-footnote">
        Confirmation is automated; review is a separate human decision. Unknown PPE observations are
        not violation events.
      </p>
    </Panel>
  );
}
