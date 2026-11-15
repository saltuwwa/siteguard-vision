import { useState } from 'react';
import { useWorkspace } from '../../app/context';
import { Badge, Button, Panel, Select } from '../../components/ui';
import type { DemoScenario } from '../../types';
export default function Settings() {
  const { data, scenario, setScenario, refresh } = useWorkspace();
  const [notice, setNotice] = useState('');
  return (
    <div className="settings-grid">
      <Panel title="Workspace configuration">
        <div className="settings-content">
          <div className="setting-row">
            <div>
              <h3>Data source</h3>
              <p>
                {data.demo
                  ? 'Deterministic simulated cameras and events'
                  : 'FastAPI through your configured server proxy'}
              </p>
            </div>
            <Badge tone={data.demo ? 'amber' : 'green'}>
              {data.demo ? 'Demo mode' : 'API mode'}
            </Badge>
          </div>
          <div className="setting-row">
            <div>
              <h3>Site</h3>
              <p>Riverside construction · UTC+05</p>
            </div>
            <Badge>Single-site workspace</Badge>
          </div>
          <div className="setting-row">
            <div>
              <h3>Review persistence</h3>
              <p>
                {data.demo
                  ? 'In-memory demo session; resets when the page reloads.'
                  : 'Read-only. No review mutation route exists in the backend.'}
              </p>
            </div>
          </div>
          <div className="setting-row">
            <div>
              <h3>Camera connection management</h3>
              <p>Configure sources in the backend. Browser streaming is not yet provided.</p>
            </div>
            <Button disabled>Backend required</Button>
          </div>
          <div className="setting-row">
            <div>
              <h3>User & permissions</h3>
              <p>Operator placeholder. Authentication and account management are not configured.</p>
            </div>
            <Badge>Preview</Badge>
          </div>
        </div>
      </Panel>
      <Panel title="Demonstration states">
        <div className="settings-content">
          <p className="muted">
            Preview how the workspace handles missing data and service interruptions.
          </p>
          <label className="field-label">
            Scenario
            <Select
              label="Demo scenario"
              disabled={!data.demo}
              value={scenario}
              onChange={(e) => {
                setScenario(e.target.value as DemoScenario);
                setNotice('Scenario applied. Open Overview to inspect the result.');
              }}
            >
              <option value="normal">Normal · populated</option>
              <option value="empty">Empty · no cameras or events</option>
              <option value="error">Error · connection interrupted</option>
              <option value="loading">Loading · 2.5 second delay</option>
            </Select>
          </label>
          <Button
            onClick={() => {
              refresh();
              setNotice('Refreshing workspace data.');
            }}
          >
            Refresh data
          </Button>
          {notice && (
            <p role="status" className="inline-note">
              {notice}
            </p>
          )}
          <p className="page-note">
            Mode and API base URL are configured at build time using the documented environment
            variables.
          </p>
        </div>
      </Panel>
    </div>
  );
}
