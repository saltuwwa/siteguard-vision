"""Evaluate a genuine checkpoint and preserve native-class error evidence.

The same explicit protocol is used for smoke/baseline comparison. Predictions
never become labels; source annotations are read only.
"""
from __future__ import annotations
import argparse
import collections
import json
import os
from pathlib import Path
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from siteguard.evaluation.objects import frame_diagnostics
from siteguard.training.datasets import class_names, sha256, write_json
from siteguard.inference.runtime import require_ultralytics

def dataset_statistics(data_path: Path, imgsz: int) -> dict:
    data = yaml.safe_load(data_path.read_text())
    names = class_names(data)
    root = Path(data['path'])
    report = {'imgsz': imgsz, 'size_basis': 'short side in pixels after isotropic resize to imgsz', 'splits': {}}
    for split in ('train', 'val'):
        objects, images = (collections.Counter(), collections.Counter())
        sizes = {name: [] for name in names.values()}
        total = 0
        for image in sorted((root / data[split]).glob('*.jpg')):
            with Image.open(image) as im:
                width, height = im.size
            scale = imgsz / max(width, height)
            rows = [list(map(float, line.split())) for line in (root / 'labels' / split / f'{image.stem}.txt').read_text().splitlines() if line.strip()]
            total += 1
            images.update({names[int(row[0])] for row in rows})
            for cls, _, _, w, h in rows:
                name = names[int(cls)]
                objects[name] += 1
                sizes[name].append(min(w * width, h * height) * scale)
        report['splits'][split] = {'images': total, 'classes': {name: {'objects': objects[name], 'images': images[name], 'short_side_quantiles': [float(v) for v in np.percentile(values, [10, 50, 90])] if values else [], 'short_side_bins': {'under8': sum((v < 8 for v in values)), '8to16': sum((8 <= v < 16 for v in values)), '16to32': sum((16 <= v < 32 for v in values)), '32plus': sum((v >= 32 for v in values))}} for name, values in sizes.items()}}
    return report

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default='0')
    parser.add_argument('--batch', type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Analysis output must be a new directory')
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ.setdefault('YOLO_CONFIG_DIR', str(Path('runs/ultralytics-config').resolve()))
    version = require_ultralytics()
    from ultralytics import YOLO
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    protocol = {'imgsz': args.imgsz, 'batch': args.batch, 'device': int(args.device) if args.device.isdigit() else args.device, 'workers': 0, 'conf': 0.001, 'iou': 0.7, 'max_det': 300, 'half': False, 'seed': 42, 'deterministic': True, 'split': 'val'}
    model = YOLO(str(args.model))
    metrics = model.val(data=str(args.data.resolve()), **protocol, plots=True, save_txt=True, save_conf=True, project=str(args.output.parent.resolve()), name=args.output.name, exist_ok=False)
    output = Path(metrics.save_dir)
    classes = model.names
    records = [{'class_id': int(cls), 'class': classes[int(cls)], 'precision': float(p), 'recall': float(r), 'ap50': float(a50), 'ap50_95': float(ap)} for cls, p, r, a50, ap in zip(metrics.box.ap_class_index, metrics.box.p, metrics.box.r, metrics.box.ap50, metrics.box.ap, strict=True)]
    curves = metrics.curves_results[0]
    recall, precisions = (np.asarray(curves[0]), np.asarray(curves[1]))
    for index, cls in enumerate(metrics.box.ap_class_index):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(recall, precisions[index])
        ax.set(xlabel='Recall', ylabel='Precision', title=f'{classes[int(cls)]} — development validation', xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / f'PR-{classes[int(cls)]}.png', dpi=140)
        plt.close(fig)
    diagnostic = frame_diagnostics(args.data, output / 'labels', output / 'errors', threshold=0.25, match_iou=0.5)
    matrix = np.asarray(diagnostic['confusion_rows_gt_columns_prediction'])
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matrix, cmap='Blues')
    ax.set_xticks(range(len(diagnostic['classes'])), diagnostic['classes'], rotation=70, ha='right')
    ax.set_yticks(range(len(diagnostic['classes'])), diagnostic['classes'])
    for y, x in zip(*np.nonzero(matrix), strict=True):
        ax.text(x, y, str(matrix[y, x]), ha='center', va='center', fontsize=7)
    ax.set(xlabel='Prediction', ylabel='Ground truth', title='Fixed confidence 0.25 / IoU 0.5 — includes background')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output / 'fixed-threshold-confusion.png', dpi=150)
    plt.close(fig)
    focus = {name: int(cls) for cls, name in classes.items() if name in {'helmet', 'no_helmet'}}
    report = {'scope': 'official development validation; source independence unproven', 'protocol': protocol, 'checkpoint_sha256': sha256(args.model), 'data_yaml_sha256': sha256(args.data), 'ultralytics_version': version, 'results': {k: float(v) for k, v in metrics.results_dict.items()}, 'classwise': records, 'fixed_confusion': diagnostic['confusion_rows_gt_columns_prediction'], 'confusion_classes': diagnostic['classes'], 'focus_fixed_counts': {}, 'head_error_crops': []}
    for name, cls in focus.items():
        report['focus_fixed_counts'][name] = {'tp': int(matrix[cls, cls]), 'fn': int(matrix[cls].sum() - matrix[cls, cls]), 'fp': int(matrix[:, cls].sum() - matrix[cls, cls]), 'missed_background': int(matrix[cls, -1])}
    if len(focus) == 2:
        report['helmet_to_no_helmet'] = int(matrix[focus['helmet'], focus['no_helmet']])
        report['no_helmet_to_helmet'] = int(matrix[focus['no_helmet'], focus['helmet']])
    try:
        font = ImageFont.truetype('arial.ttf', 18)
    except OSError:
        font = ImageFont.load_default()
    crops = output / 'head-error-crops'
    crops.mkdir()
    counters = collections.Counter()
    for record in diagnostic['examples']:
        for error in record['errors']:
            g = error.get('gt')
            p = error.get('prediction')
            if not any((row is not None and int(row[0]) in focus.values() for row in (g, p))):
                continue
            kind = error['kind']
            if counters[kind] >= 16:
                continue
            with Image.open(record['image']) as image:
                box = g[1:5] if g is not None else p[1:5]
                x, y, w, h = box
                pad = max(w * image.width, h * image.height) * 0.7
                cx, cy = (x * image.width, y * image.height)
                left, top = (max(0, int(cx - pad)), max(0, int(cy - pad)))
                right, bottom = (min(image.width, int(cx + pad)), min(image.height, int(cy + pad)))
                canvas = image.crop((left, top, right, bottom)).convert('RGB')
                draw = ImageDraw.Draw(canvas)
                for row, color in ((g, 'lime'), (p, 'red')):
                    if row is None:
                        continue
                    _, bx, by, bw, bh = row[:5]
                    draw.rectangle(((bx - bw / 2) * image.width - left, (by - bh / 2) * image.height - top, (bx + bw / 2) * image.width - left, (by + bh / 2) * image.height - top), outline=color, width=2)
                canvas.thumbnail((420, 420))
                panel = Image.new('RGB', (max(420, canvas.width), canvas.height + 48), '#141414')
                panel.paste(canvas, (0, 48))
                label = f"{kind}: GT {(classes[int(g[0])] if g else 'background')} / Pred {(classes[int(p[0])] if p else 'none')}"
                ImageDraw.Draw(panel).text((4, 4), label, fill='white', font=font)
                path = crops / f'{kind}-{counters[kind]:02}.jpg'
                panel.save(path)
            report['head_error_crops'].append({'file': str(path), 'image': record['image'], **error})
            counters[kind] += 1
    write_json(output / 'metrics.json', report)
    write_json(output / 'dataset-statistics.json', dataset_statistics(args.data, args.imgsz))
    print(json.dumps({'output': str(output), 'focus': report['focus_fixed_counts'], 'helmet_to_no_helmet': report.get('helmet_to_no_helmet'), 'no_helmet_to_helmet': report.get('no_helmet_to_helmet')}, indent=2))
if __name__ == '__main__':
    main()
