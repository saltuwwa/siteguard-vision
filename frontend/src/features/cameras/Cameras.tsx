import { Camera as CameraIcon, VideoOff } from 'lucide-react';
import { useWorkspace } from '../../app/context';
import { Badge, Empty, Panel } from '../../components/ui';
import { clock } from '../../lib/domain';
import { Link } from 'react-router-dom';
export default function Cameras() {
  const { data, setCameraId } = useWorkspace();
  return (
    <>
      <div className="page-intro">
        <p>Source availability and processing configuration across your site.</p>
        <Badge tone={data.demo ? 'amber' : 'neutral'}>
          {data.demo ? 'Simulated camera states' : 'Metadata only'}
        </Badge>
      </div>
      <div className="camera-cards">
        {data.cameras.map((c) => (
          <Panel
            key={c.id}
            title={
              <>
                <CameraIcon size={17} />
                {c.id}
              </>
            }
            action={
              <Badge
                tone={c.status === 'online' ? 'green' : c.status === 'offline' ? 'red' : 'neutral'}
              >
                {c.status}
              </Badge>
            }
          >
            <div className="camera-card-image">
              {c.previewUrl ? (
                <img src={c.previewUrl} alt="Illustrative site preview" />
              ) : (
                <VideoOff size={30} />
              )}
              <span>{data.demo ? 'DEMO · illustrative still' : 'No viewing endpoint'}</span>
            </div>
            <div className="camera-card-body">
              <h3>{c.name}</h3>
              <p>{c.location}</p>
              <dl>
                <dt>Site</dt>
                <dd>{c.site}</dd>
                <dt>Last frame / heartbeat</dt>
                <dd>{c.lastFrame ? `${clock(c.lastFrame)} UTC+05` : 'Not received'}</dd>
                <dt>Processing</dt>
                <dd>
                  {c.fps === null
                    ? 'Unavailable'
                    : `${c.fps} FPS${data.demo ? ' · simulated' : ''}`}
                </dd>
                <dt>Monitoring rule</dt>
                <dd>
                  {data.demo ? 'Helmet · temporal confirmation' : 'Configuration not supplied'}
                </dd>
              </dl>
              <Link className="button" to="/monitoring" onClick={() => setCameraId(c.id)}>
                Open monitoring
              </Link>
            </div>
          </Panel>
        ))}
      </div>
      {!data.cameras.length && (
        <Panel title="Cameras">
          <Empty
            title="No cameras configured"
            detail="Start a backend worker to register camera metadata. Connection management is not available in this interface."
          />
        </Panel>
      )}
      <p className="page-note">
        Camera connection setup and RTSP credentials are managed on the server. This interface does
        not connect directly to RTSP.
      </p>
    </>
  );
}
