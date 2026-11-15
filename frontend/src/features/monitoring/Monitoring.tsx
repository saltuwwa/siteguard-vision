import { useWorkspace } from '../../app/context';
import { Badge, Empty, Panel } from '../../components/ui';
import { VideoViewer } from '../../components/video/VideoViewer';
import { CameraList, PpeSummary } from '../overview/Overview';
export default function Monitoring() {
  const { data, selectedCamera, setCameraId } = useWorkspace();
  return (
    <div className="monitoring-grid">
      <div>
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
          <Empty title="No cameras available" />
        )}
        <Panel
          title="Tracked workers"
          action={<Badge>{data.demo ? 'Simulated sample' : 'No metadata endpoint'}</Badge>}
        >
          <div className="track-list">
            {selectedCamera?.detections.length ? (
              selectedCamera.detections.map((d) => (
                <div key={d.id}>
                  <strong>Track #{d.trackId}</strong>
                  <span>
                    {d.ppe === 'COMPLIANT'
                      ? 'Helmet detected'
                      : d.ppe === 'VIOLATION'
                        ? 'No helmet'
                        : 'PPE unknown'}
                  </span>
                  <Badge
                    tone={
                      d.ppe === 'COMPLIANT' ? 'green' : d.ppe === 'VIOLATION' ? 'red' : 'neutral'
                    }
                  >
                    {d.confidence === null
                      ? 'Not classified'
                      : `${Math.round(d.confidence * 100)}% confidence`}
                  </Badge>
                </div>
              ))
            ) : (
              <Empty
                title="No track metadata"
                detail="Frame-level detections are not supplied by the current API."
              />
            )}
          </div>
          <p className="panel-footnote">
            Track identities are scoped to the camera and session. They do not identify a person
            across cameras.
          </p>
        </Panel>
      </div>
      <aside className="insights-column">
        <PpeSummary />
        <CameraList />
      </aside>
    </div>
  );
}
