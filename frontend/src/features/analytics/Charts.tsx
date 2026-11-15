import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { clock } from '../../lib/domain';
import { Empty } from '../../components/ui';
import type { TimeRange, ViolationEvent } from '../../types';
export function timeline(events: ViolationEvent[], asOf: string, range: TimeRange) {
  const end = new Date(asOf).valueOf(),
    hours = range === '1h' ? 1 : range === '4h' ? 4 : 12;
  const start = end - hours * 3600000,
    step = (hours * 3600000) / 12;
  return Array.from({ length: 12 }, (_, i) => {
    const bucket = events.filter((e) => {
      const t = new Date(e.createdAt).valueOf();
      return t >= start + i * step && (i === 11 ? t <= end : t < start + (i + 1) * step);
    });
    return {
      time: clock(new Date(start + i * step).toISOString()).slice(0, 5),
      unreviewed: bucket.filter((e) => e.review === 'pending').length,
      reviewed: bucket.filter((e) => e.review !== 'pending').length,
    };
  });
}
const tooltipStyle = {
  background: '#1b2330',
  border: '1px solid #344054',
  borderRadius: 5,
  color: '#e7ecf5',
  fontSize: 12,
};
export function TrendChart({
  events,
  asOf,
  range,
}: {
  events: ViolationEvent[];
  asOf: string;
  range: TimeRange;
}) {
  const data = timeline(events, asOf, range);
  if (!events.length) return <Empty title="No events in selected period" />;
  return (
    <>
      <div
        className="chart"
        role="img"
        aria-label={`Violations over time, ${events.length} events. Stacked by reviewed and unreviewed.`}
      >
        <ResponsiveContainer width="100%" height="100%" minWidth={0}>
          <BarChart
            data={data}
            margin={{ top: 10, right: 8, left: -27, bottom: 0 }}
            barCategoryGap="24%"
          >
            <CartesianGrid vertical={false} stroke="#26303d" />
            <XAxis
              dataKey="time"
              tick={{ fill: '#8e9aab', fontSize: 10 }}
              axisLine={false}
              tickLine={false}
              interval={2}
            />
            <YAxis
              allowDecimals={false}
              tick={{ fill: '#8e9aab', fontSize: 10 }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip contentStyle={tooltipStyle} cursor={{ fill: '#ffffff06' }} />
            <Bar
              dataKey="reviewed"
              name="Reviewed events"
              stackId="events"
              fill="#8493a7"
              radius={[0, 0, 0, 0]}
            />
            <Bar
              dataKey="unreviewed"
              name="Unreviewed events"
              stackId="events"
              fill="#e8b85a"
              radius={[2, 2, 0, 0]}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="chart-legend">
        <span>
          <i className="legend-dot neutral-dot" />
          Reviewed
        </span>
        <span>
          <i className="legend-dot amber-dot" />
          Unreviewed
        </span>
        <span className="ml-auto">Events · UTC+05</span>
      </div>
    </>
  );
}
export function CameraChart({ events }: { events: ViolationEvent[] }) {
  const data = [...new Set(events.map((e) => e.cameraId))].map((id) => ({
    camera: id,
    events: events.filter((e) => e.cameraId === id).length,
  }));
  return !data.length ? (
    <Empty />
  ) : (
    <div className="chart tall" role="img" aria-label="Confirmed event counts by camera">
      <ResponsiveContainer width="100%" height="100%" minWidth={0}>
        <BarChart data={data} margin={{ left: -20, right: 20, top: 20 }}>
          <CartesianGrid vertical={false} stroke="#26303d" />
          <XAxis dataKey="camera" tick={{ fill: '#8e9aab', fontSize: 12 }} />
          <YAxis allowDecimals={false} tick={{ fill: '#8e9aab', fontSize: 11 }} />
          <Tooltip contentStyle={tooltipStyle} />
          <Bar
            dataKey="events"
            name="Events"
            fill="#8493a7"
            maxBarSize={70}
            radius={[3, 3, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
