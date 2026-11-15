import { useEffect, useRef, useState } from 'react';
import {
  Camera as CameraIcon,
  Expand,
  Eye,
  EyeOff,
  Image as ImageIcon,
  Pause,
  Play,
  VideoOff,
} from 'lucide-react';
import { Badge, Button, Select } from '../ui';
import { clock, containRect, projectBox } from '../../lib/domain';
import type { Camera, Detection } from '../../types';
export interface StreamingAdapter {
  attach: (video: HTMLVideoElement) => Promise<void>;
  detach: () => void;
}
interface Props {
  camera: Camera;
  cameras: Camera[];
  onCameraChange: (id: string) => void;
  timestamp: string;
  demo: boolean;
}
export function DetectionLayer({
  detections,
  width,
  height,
  sourceWidth,
  sourceHeight,
}: {
  detections: Detection[];
  width: number;
  height: number;
  sourceWidth: number;
  sourceHeight: number;
}) {
  const rect = containRect(width, height, sourceWidth, sourceHeight);
  return (
    <div className="detection-layer" aria-label={`${detections.length} simulated tracked workers`}>
      {detections.map((d) => (
        <div
          key={d.id}
          className={`detection ${d.ppe.toLowerCase()}`}
          style={projectBox(d.box, rect)}
        >
          <span>
            #{d.trackId} ·{' '}
            {d.ppe === 'COMPLIANT' ? 'Helmet' : d.ppe === 'VIOLATION' ? 'No helmet' : 'Unknown'}
            {d.confidence !== null ? ` ${(d.confidence * 100).toFixed(0)}%` : ''}
          </span>
        </div>
      ))}
    </div>
  );
}
export function VideoViewer({ camera, cameras, onCameraChange, timestamp, demo }: Props) {
  const root = useRef<HTMLDivElement>(null),
    stage = useRef<HTMLDivElement>(null),
    video = useRef<HTMLVideoElement>(null);
  const [overlays, setOverlays] = useState(true),
    [playing, setPlaying] = useState(false),
    [failed, setFailed] = useState(false),
    [imageFailed, setImageFailed] = useState(false),
    [message, setMessage] = useState('');
  const [size, setSize] = useState({ width: 0, height: 0 }),
    [sourceSize, setSourceSize] = useState({ width: 1440, height: 810 }),
    [position, setPosition] = useState(0),
    [length, setLength] = useState(0);
  useEffect(() => {
    if (!stage.current) return;
    const observer = new ResizeObserver(([entry]) =>
      setSize({ width: entry.contentRect.width, height: entry.contentRect.height }),
    );
    observer.observe(stage.current);
    return () => observer.disconnect();
  }, []);
  const playable = Boolean(camera.videoUrl) && !failed;
  const available = camera.status !== 'offline' && (playable || Boolean(camera.previewUrl));
  async function togglePlayback() {
    if (!video.current) return;
    if (video.current.paused) {
      try {
        await video.current.play();
      } catch {
        setMessage('Playback could not start. Check the media format.');
      }
    } else video.current.pause();
  }
  async function fullscreen() {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await root.current?.requestFullscreen();
    } catch {
      setMessage('Fullscreen is unavailable in this browser.');
    }
  }
  return (
    <div className="panel video-viewer" ref={root}>
      <div className="viewer-heading">
        <div className="camera-title">
          <CameraIcon size={18} />
          <Select
            label="Monitoring camera"
            value={camera.id}
            onChange={(e) => onCameraChange(e.target.value)}
          >
            {cameras.map((c) => (
              <option key={c.id} value={c.id}>
                {c.id} — {c.name}
              </option>
            ))}
          </Select>
        </div>
        <Badge tone={demo ? 'amber' : 'neutral'}>{demo ? 'DEMO' : 'No browser stream'}</Badge>
      </div>
      <div className="video-stage" ref={stage}>
        {available ? (
          playable ? (
            <video
              ref={video}
              src={camera.videoUrl}
              playsInline
              preload="metadata"
              onError={() => {
                setFailed(true);
                setPlaying(false);
              }}
              onLoadedMetadata={(e) => {
                setSourceSize({
                  width: e.currentTarget.videoWidth,
                  height: e.currentTarget.videoHeight,
                });
                setLength(e.currentTarget.duration);
              }}
              onTimeUpdate={(e) => setPosition(e.currentTarget.currentTime)}
              onPlay={() => setPlaying(true)}
              onPause={() => setPlaying(false)}
              onEnded={() => setPlaying(false)}
            />
          ) : !imageFailed ? (
            <img
              src={camera.previewUrl}
              alt="Illustrative construction site; not a live feed or event evidence"
              onError={() => setImageFailed(true)}
              onLoad={(e) =>
                setSourceSize({
                  width: e.currentTarget.naturalWidth,
                  height: e.currentTarget.naturalHeight,
                })
              }
            />
          ) : null
        ) : null}
        {(!available || imageFailed) && (
          <div className="preview-fallback">
            <VideoOff size={34} />
            <strong>
              {camera.status === 'offline' ? 'Camera unavailable' : 'No browser-playable source'}
            </strong>
            <p>
              {demo
                ? 'Simulated camera state. Select another camera.'
                : 'This camera provides metadata only. A streaming adapter is required.'}
            </p>
          </div>
        )}
        {available && !imageFailed && overlays && !playable && (
          <DetectionLayer
            detections={camera.detections}
            {...size}
            sourceWidth={sourceSize.width}
            sourceHeight={sourceSize.height}
          />
        )}
        <div className="source-stamp">
          <span>{camera.id}</span>
          <span>{clock(timestamp)} UTC+05</span>
        </div>
        {available && !playable && !imageFailed && (
          <span className="still-label">
            <ImageIcon size={12} /> Illustrative still · simulated overlays
          </span>
        )}
        {playable && overlays && (
          <span className="still-label">No synchronized detection metadata</span>
        )}
      </div>
      <div className="viewer-controls">
        <Button
          className="icon-button"
          aria-label={playing ? 'Pause video' : 'Play video'}
          disabled={!playable}
          onClick={() => void togglePlayback()}
        >
          {playing ? <Pause size={17} /> : <Play size={17} />}
        </Button>
        <span className="source-label">
          {playable ? 'Recorded demo clip' : available ? 'Preview image' : 'Source unavailable'}
        </span>
        {playable && (
          <input
            aria-label="Video position"
            type="range"
            min={0}
            max={length || 0}
            step={0.1}
            value={position}
            onChange={(e) => {
              if (video.current) video.current.currentTime = Number(e.target.value);
            }}
          />
        )}
        <div className="control-spacer" />
        <Button
          className="overlay-button"
          aria-pressed={overlays}
          onClick={() => setOverlays(!overlays)}
        >
          {overlays ? <Eye size={16} /> : <EyeOff size={16} />}
          <span>Overlays {overlays ? 'on' : 'off'}</span>
        </Button>
        <Button
          className="icon-button"
          aria-label="Toggle fullscreen"
          onClick={() => void fullscreen()}
        >
          <Expand size={17} />
        </Button>
      </div>
      {message && (
        <p className="inline-note" role="status">
          {message}
        </p>
      )}
      <div className="viewer-caption">
        <span>{camera.location}</span>
        <span>{demo ? 'Sample scene · no live connection' : 'Metadata only'}</span>
      </div>
    </div>
  );
}
