import { lazy, Suspense, useEffect, useMemo, useState } from 'react';
import {
  Activity,
  ArrowLeftToLine,
  ArrowRightFromLine,
  BarChart3,
  Building2,
  Camera,
  ChevronDown,
  HardHat,
  LayoutDashboard,
  Radio,
  RefreshCw,
  Settings as SettingsIcon,
  ShieldAlert,
  Video,
} from 'lucide-react';
import { Link, NavLink, Route, Routes, useLocation } from 'react-router-dom';
import { WorkspaceContext } from './context';
import { isMock, useDashboard } from '../hooks/useDashboard';
import { Badge, Button, DataState, Select } from '../components/ui';
import { EventDrawer } from '../features/violations/EventDrawer';
import Overview from '../features/overview/Overview';
import { clock, dateLabel, filterEvents } from '../lib/domain';
import type { DashboardData, DemoScenario, EventReviewStatus, TimeRange } from '../types';
const Monitoring = lazy(() => import('../features/monitoring/Monitoring'));
const Cameras = lazy(() => import('../features/cameras/Cameras'));
const Violations = lazy(() => import('../features/violations/Violations'));
const Analytics = lazy(() => import('../features/analytics/Analytics'));
const Settings = lazy(() => import('../features/settings/Settings'));
const navigation = [
  { path: '/', label: 'Overview', icon: LayoutDashboard },
  { path: '/monitoring', label: 'Live Monitoring', icon: Video },
  { path: '/cameras', label: 'Cameras', icon: Camera },
  { path: '/violations', label: 'Violations', icon: ShieldAlert },
  { path: '/analytics', label: 'Analytics', icon: BarChart3 },
];
const titles: Record<string, string> = {
  '/': 'Safety Overview',
  '/monitoring': 'Live Monitoring',
  '/cameras': 'Camera Management',
  '/violations': 'Violations',
  '/analytics': 'Safety Analytics',
  '/settings': 'Settings',
};
const emptyData: DashboardData = {
  cameras: [],
  events: [],
  asOf: '2026-11-15T11:24:08Z',
  demo: isMock,
  totalEvents: 0,
};
export default function App() {
  const [scenario, setScenario] = useState<DemoScenario>('normal'),
    [collapsed, setCollapsed] = useState(false),
    [cameraId, setCameraId] = useState('all'),
    [range, setRange] = useState<TimeRange>('4h'),
    [eventId, setEventId] = useState<string | null>(null),
    [reviews, setReviews] = useState<Record<string, EventReviewStatus>>({}),
    [profile, setProfile] = useState(false);
  const query = useDashboard(scenario),
    location = useLocation(),
    data = query.data || emptyData;
  const title = titles[location.pathname] || 'Page not found';
  useEffect(() => {
    document.title = `${title} · SiteGuard Vision`;
  }, [title]);
  const events = useMemo(
    () => data.events.map((e) => (reviews[e.id] ? { ...e, review: reviews[e.id] } : e)),
    [data.events, reviews],
  );
  const filtered = useMemo(
    () => filterEvents(events, { camera: cameraId, range, asOf: data.asOf }),
    [events, cameraId, range, data.asOf],
  );
  const selectedCamera = data.cameras.find((c) => c.id === cameraId) || data.cameras[0];
  const failed = query.isError || query.isPending;
  return (
    <div className={`app-shell ${collapsed ? 'sidebar-collapsed' : ''}`}>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <aside className="sidebar">
        <Link className="brand" to="/" aria-label="SiteGuard Vision overview">
          <div className="brand-icon">
            <HardHat size={32} strokeWidth={1.8} />
          </div>
          <div>
            <strong>
              SiteGuard<span>Vision</span>
            </strong>
            <small>Construction safety platform</small>
          </div>
        </Link>
        <div className="workspace-label">Workspace</div>
        <nav aria-label="Main navigation">
          {navigation.map(({ path, label, icon: Icon }) => (
            <NavLink key={path} to={path} end={path === '/'} title={collapsed ? label : undefined}>
              <Icon size={19} />
              <span>{label}</span>
              {path === '/violations' &&
                events.filter((e) => e.review === 'pending').length > 0 && (
                  <small>{events.filter((e) => e.review === 'pending').length}</small>
                )}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <NavLink to="/settings" title={collapsed ? 'Settings' : undefined}>
            <SettingsIcon size={19} />
            <span>Settings</span>
          </NavLink>
          <div className="sidebar-status">
            <div>
              <i className={`legend-dot ${isMock ? 'amber' : 'neutral'}-dot`} />
              <strong>{isMock ? 'Demo workspace' : 'API workspace'}</strong>
            </div>
            <p>{isMock ? 'Simulated data · no live feeds' : 'Server metadata integration'}</p>
            <small>
              SiteGuard Vision <span>v0.1.0</span>
            </small>
          </div>
          <Button
            className="collapse-button"
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            onClick={() => setCollapsed(!collapsed)}
          >
            {collapsed ? <ArrowRightFromLine size={16} /> : <ArrowLeftToLine size={16} />}
            <span>Collapse sidebar</span>
          </Button>
        </div>
      </aside>
      <div className="workspace">
        <header className="topbar">
          <div className="breadcrumb">
            <Building2 size={15} />
            <span>Riverside construction</span>
            <span className="breadcrumb-divider">/</span>
            <strong>{title}</strong>
          </div>
          <div className="topbar-right">
            <span className="header-date">{dateLabel(data.asOf)}</span>
            <Badge tone={isMock ? 'amber' : 'neutral'}>
              <span className="status-dot" />
              {isMock ? 'Demo mode' : 'API mode'}
            </Badge>
            <div className="profile-container">
              <Button
                className="profile-button"
                aria-label="Operator profile"
                aria-expanded={profile}
                onClick={() => setProfile(!profile)}
              >
                AK
                <ChevronDown size={12} />
              </Button>
              {profile && (
                <div className="profile-popover">
                  <strong>Alex Kim</strong>
                  <small>Demo operator placeholder</small>
                  <Link to="/settings" onClick={() => setProfile(false)}>
                    Workspace settings
                  </Link>
                </div>
              )}
            </div>
          </div>
        </header>
        <main id="main">
          <div className="page-heading">
            <div>
              <div className="heading-title">
                <h1>{title}</h1>
                {location.pathname === '/' && <span className="site-tag">Site A</span>}
              </div>
              <p>
                {location.pathname === '/'
                  ? 'A clear view of site safety. Every camera, every event.'
                  : location.pathname === '/monitoring'
                    ? 'Camera previews and track-level PPE observations.'
                    : location.pathname === '/violations'
                      ? 'Investigate confirmed events and manage the review queue.'
                      : location.pathname === '/analytics'
                        ? 'Understand safety events within a defined monitoring period.'
                        : location.pathname === '/cameras'
                          ? 'Monitor source health across your construction site.'
                          : 'Manage workspace preferences and integration readiness.'}
              </p>
            </div>
            <div className="updated">
              <Activity size={13} />
              <span>
                {isMock ? 'Demo snapshot' : 'Fetched'} {clock(data.asOf)}
              </span>
              <Button
                className="icon-button"
                aria-label="Refresh dashboard"
                disabled={query.isFetching}
                onClick={() => void query.refetch()}
              >
                <RefreshCw size={14} />
              </Button>
            </div>
          </div>
          <div className="filterbar">
            <div className="site-selector">
              <Building2 size={15} />
              <span>{isMock ? 'Riverside construction' : 'Configured site'}</span>
            </div>
            <span className="filter-divider" />
            <span className="monitoring-status">
              <Radio size={14} />
              {isMock ? 'Simulated monitoring' : 'Metadata workspace'}
            </span>
            {location.pathname !== '/violations' &&
              location.pathname !== '/settings' &&
              location.pathname !== '/cameras' && (
                <div className="filterbar-right">
                  <Select
                    label="Camera filter"
                    value={cameraId}
                    onChange={(e) => setCameraId(e.target.value)}
                  >
                    <option value="all">All cameras</option>
                    {data.cameras.map((c) => (
                      <option key={c.id}>{c.id}</option>
                    ))}
                  </Select>
                  {location.pathname !== '/monitoring' && (
                    <Select
                      label="Time range"
                      value={range}
                      onChange={(e) => setRange(e.target.value as TimeRange)}
                    >
                      <option value="1h">Last hour</option>
                      <option value="4h">Last 4 hours</option>
                      <option value="shift">Current shift · 12 h</option>
                    </Select>
                  )}
                </div>
              )}
          </div>
          {isMock && (
            <div className="demo-notice">
              <span>
                <span className="demo-marker">DEMO</span>Simulated events and camera states. Preview
                images are illustrative.
              </span>
              <Link to="/settings">
                Configure workspace <ChevronDown size={12} />
              </Link>
            </div>
          )}
          <WorkspaceContext.Provider
            value={{
              data,
              events,
              filtered,
              cameraId,
              setCameraId,
              range,
              setRange,
              selectedCamera,
              openEvent: setEventId,
              scenario,
              setScenario,
              refresh: () => void query.refetch(),
            }}
          >
            {failed && location.pathname !== '/settings' ? (
              <>
                <DataState
                  loading={query.isPending}
                  error={query.error}
                  retry={() => void query.refetch()}
                />
                {query.isError && isMock && (
                  <div className="recovery">
                    <Button onClick={() => setScenario('normal')}>Restore normal demo</Button>
                    <Link to="/settings">Demo settings</Link>
                  </div>
                )}
              </>
            ) : (
              <Suspense fallback={<DataState loading retry={() => undefined} />}>
                <Routes>
                  <Route path="/" element={<Overview />} />
                  <Route path="/monitoring" element={<Monitoring />} />
                  <Route path="/cameras" element={<Cameras />} />
                  <Route path="/violations" element={<Violations />} />
                  <Route path="/analytics" element={<Analytics />} />
                  <Route path="/settings" element={<Settings />} />
                  <Route
                    path="*"
                    element={
                      <div className="data-state">
                        <h2>Page not found</h2>
                        <Link className="button" to="/">
                          Return to overview
                        </Link>
                      </div>
                    }
                  />
                </Routes>
              </Suspense>
            )}
            <EventDrawer
              id={eventId}
              events={events}
              demo={data.demo}
              onClose={() => setEventId(null)}
              onReview={(id, review) => setReviews((r) => ({ ...r, [id]: review }))}
            />
          </WorkspaceContext.Provider>
          <footer className="workspace-footer">
            <span>
              <HardHat size={13} />
              SiteGuard Vision
            </span>
            <span>
              {isMock ? 'Demonstration environment' : 'Read-only monitoring workspace'}
              <span className="footer-separator">·</span>Times shown in UTC+05
            </span>
          </footer>
        </main>
      </div>
    </div>
  );
}
