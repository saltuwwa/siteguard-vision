import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it, vi } from 'vitest';
import { VideoViewer } from '../components/video/VideoViewer';
import { mockData } from '../services/mock/data';
import { EventDrawer } from '../features/violations/EventDrawer';
import { EventTable } from '../features/violations/EventTable';
describe('monitoring controls', () => {
  it('switches cameras and toggles overlays without claiming live playback', async () => {
    const onChange = vi.fn();
    render(
      <VideoViewer
        camera={mockData.cameras[0]}
        cameras={mockData.cameras}
        onCameraChange={onChange}
        timestamp={mockData.asOf}
        demo
      />,
    );
    expect(screen.getByRole('button', { name: 'Play video' })).toBeDisabled();
    const toggle = screen.getByRole('button', { name: 'Overlays on' });
    await userEvent.click(toggle);
    expect(toggle).toHaveAttribute('aria-pressed', 'false');
    fireEvent.change(screen.getByLabelText('Monitoring camera'), { target: { value: 'CAM-02' } });
    expect(onChange).toHaveBeenCalledWith('CAM-02');
    expect(screen.getByText('DEMO')).toBeInTheDocument();
  });
});
describe('event review', () => {
  it('opens the selected event from the table', async () => {
    const select = vi.fn();
    render(
      <EventTable events={[mockData.events[0]]} cameras={mockData.cameras} onSelect={select} />,
    );
    await userEvent.click(
      screen.getByRole('button', { name: `Open event ${mockData.events[0].id}` }),
    );
    expect(select).toHaveBeenCalledWith(mockData.events[0].id);
  });
  it('supports review actions and keyboard dismissal in an accessible dialog', async () => {
    const onClose = vi.fn(),
      onReview = vi.fn();
    render(
      <QueryClientProvider client={new QueryClient()}>
        <EventDrawer
          id={mockData.events[0].id}
          events={mockData.events}
          demo
          onClose={onClose}
          onReview={onReview}
        />
      </QueryClientProvider>,
    );
    expect(screen.getByRole('dialog')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Accept event' }));
    expect(onReview).toHaveBeenCalledWith(mockData.events[0].id, 'accepted');
    await userEvent.keyboard('{Escape}');
    expect(onClose).toHaveBeenCalled();
  });
});
describe('URL filters', () => {
  it('filters event rows by review and resets the query', async () => {
    const { default: Violations } = await import('../features/violations/Violations');
    const { WorkspaceContext } = await import('../app/context');
    render(
      <MemoryRouter initialEntries={['/violations?review=rejected&range=shift']}>
        <WorkspaceContext.Provider
          value={{
            data: mockData,
            events: mockData.events,
            filtered: mockData.events,
            cameraId: 'all',
            setCameraId: vi.fn(),
            range: 'shift',
            setRange: vi.fn(),
            selectedCamera: mockData.cameras[0],
            openEvent: vi.fn(),
            scenario: 'normal',
            setScenario: vi.fn(),
            refresh: vi.fn(),
          }}
        >
          <Violations />
        </WorkspaceContext.Provider>
      </MemoryRouter>,
    );
    expect(screen.getByLabelText('Filter by review')).toHaveValue('rejected');
    expect(screen.getAllByRole('row').length).toBe(3);
    await userEvent.click(screen.getByRole('button', { name: 'Reset' }));
    expect(screen.getByLabelText('Filter by review')).toHaveValue('all');
  });
});
