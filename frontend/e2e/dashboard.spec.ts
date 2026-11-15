import { expect, test } from '@playwright/test';
import { mkdir } from 'node:fs/promises';
test('overview is usable at desktop, tablet and mobile widths', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text());
  });
  await mkdir('docs/screenshots', { recursive: true });
  for (const [width, height] of [
    [1440, 900],
    [1920, 1080],
    [1024, 768],
    [390, 844],
  ]) {
    await page.setViewportSize({ width, height });
    await page.goto('/');
    await expect(page.getByRole('heading', { name: 'PPE in current view' })).toBeVisible();
    await expect(page.locator('.video-stage img')).toBeVisible();
    await page.locator('.video-stage img').evaluate(async (image: HTMLImageElement) => {
      if (!image.complete)
        await new Promise((resolve) => image.addEventListener('load', resolve, { once: true }));
    });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(
      true,
    );
    const viewport = await page.locator('.video-stage').boundingBox();
    expect(viewport?.width).toBeGreaterThan(300);
    await page.screenshot({
      path: `docs/screenshots/overview-${width}x${height}.png`,
      fullPage: true,
      animations: 'disabled',
    });
  }
  expect(errors).toEqual([]);
});
test('camera, overlay, time range, drawer and review interactions', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/');
  await expect(page.getByLabel('Monitoring camera')).toBeVisible();
  await page.getByLabel('Monitoring camera').selectOption('CAM-02');
  await expect(page.getByLabel('Camera filter')).toHaveValue('CAM-02');
  await page.getByRole('button', { name: 'Overlays on' }).click();
  await expect(page.locator('.detection-layer')).toHaveCount(0);
  await page.getByRole('button', { name: 'Overlays off' }).click();
  await expect(page.locator('.detection-layer')).toHaveCount(1);
  await page.getByLabel('Time range').selectOption('1h');
  await expect(page.getByRole('heading', { name: 'Violations over time' })).toBeVisible();
  await page.getByRole('button', { name: 'Open event SG-01048' }).click();
  await expect(page.getByRole('dialog')).toBeVisible();
  await page.screenshot({ path: 'docs/screenshots/event-details.png', fullPage: true });
  await page.getByRole('button', { name: 'Accept event' }).click();
  await expect(page.getByRole('button', { name: 'Accept event' })).toBeDisabled();
  await page.keyboard.press('Escape');
  await expect(page.getByRole('dialog')).toHaveCount(0);
  await page.getByLabel('Monitoring camera').selectOption('CAM-03');
  await expect(page.getByText('Camera unavailable', { exact: true })).toBeVisible();
});
test('secondary routes, URL filters and pagination', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/violations?range=shift');
  await expect(page.getByRole('heading', { name: 'Violation events' })).toBeVisible();
  await page.getByRole('button', { name: 'Next page' }).click();
  await expect(page).toHaveURL(/page=2/);
  await page.getByLabel('Filter by review').selectOption('pending');
  await expect(page).toHaveURL(/review=pending/);
  await expect(page.getByRole('button', { name: 'Next page' })).toBeDisabled();
  await page.reload();
  await expect(page.getByLabel('Filter by review')).toHaveValue('pending');
  await page.getByRole('button', { name: 'Reset', exact: true }).click();
  await expect(page.getByLabel('Filter by review')).toHaveValue('all');
  await page.getByLabel('Search events').fill('does-not-exist');
  await expect(page.getByText('No events in this view')).toBeVisible();
  for (const [route, title] of [
    ['/cameras', 'Camera Management'],
    ['/analytics', 'Safety Analytics'],
    ['/monitoring', 'Live Monitoring'],
    ['/settings', 'Settings'],
  ]) {
    await page.goto(route);
    await expect(page.getByRole('heading', { name: title, exact: true })).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(
      true,
    );
  }
  await page.goto('/cameras');
  await page.screenshot({ path: 'docs/screenshots/cameras.png', fullPage: true });
  await page.goto('/analytics');
  await page.screenshot({ path: 'docs/screenshots/analytics.png', fullPage: true });
});
test('empty, error and loading scenarios recover', async ({ page }) => {
  await page.goto('/settings');
  await page.getByLabel('Demo scenario').selectOption('empty');
  await page.getByRole('link', { name: 'Overview', exact: true }).click();
  await expect(page.getByText('No cameras configured').first()).toBeVisible();
  await page.goto('/settings');
  await page.getByLabel('Demo scenario').selectOption('error');
  await page.getByRole('link', { name: 'Overview', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Unable to load monitoring data' })).toBeVisible();
  await page.getByRole('button', { name: 'Restore normal demo' }).click();
  await expect(page.getByLabel('Monitoring camera')).toBeVisible();
  await page.getByRole('link', { name: 'Settings', exact: true }).click();
  await page.getByLabel('Demo scenario').selectOption('loading');
  await page.getByRole('link', { name: 'Overview', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Loading monitoring workspace' })).toBeVisible();
  await expect(page.getByLabel('Monitoring camera')).toBeVisible();
});
