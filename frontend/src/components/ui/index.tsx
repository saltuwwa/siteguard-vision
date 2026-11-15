import type { ButtonHTMLAttributes, ReactNode, SelectHTMLAttributes } from 'react';
import { AlertTriangle, RefreshCw, SearchX } from 'lucide-react';
export function Button({
  children,
  className = '',
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button className={`button ${className}`} {...props}>
      {children}
    </button>
  );
}
export function Select({
  label,
  children,
  ...props
}: SelectHTMLAttributes<HTMLSelectElement> & { label: string }) {
  return (
    <label className="select-label">
      <span className="sr-only">{label}</span>
      <select {...props}>{children}</select>
    </label>
  );
}
export function Badge({
  children,
  tone = 'neutral',
}: {
  children: ReactNode;
  tone?: 'green' | 'amber' | 'red' | 'neutral';
}) {
  return <span className={`badge ${tone}`}>{children}</span>;
}
export function Panel({
  title,
  action,
  children,
  className = '',
}: {
  title: ReactNode;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel ${className}`}>
      <div className="panel-heading">
        <h2>{title}</h2>
        {action}
      </div>
      {children}
    </section>
  );
}
export function Empty({
  title = 'No events in this view',
  detail = 'Try another camera or a wider time range.',
}: {
  title?: string;
  detail?: string;
}) {
  return (
    <div className="empty">
      <SearchX size={25} />
      <strong>{title}</strong>
      <p>{detail}</p>
    </div>
  );
}
export function DataState({
  loading,
  error,
  retry,
}: {
  loading: boolean;
  error?: Error | null;
  retry: () => void;
}) {
  return (
    <div className="data-state" role={error ? 'alert' : 'status'}>
      {loading ? <RefreshCw size={26} /> : <AlertTriangle size={26} />}
      <h2>{loading ? 'Loading monitoring workspace' : 'Unable to load monitoring data'}</h2>
      <p>{loading ? 'Retrieving cameras and safety events…' : error?.message}</p>
      {!loading && (
        <Button onClick={retry}>
          <RefreshCw size={15} /> Retry
        </Button>
      )}
    </div>
  );
}
