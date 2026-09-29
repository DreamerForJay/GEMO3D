"""Reproject saved GEMO3D predictions; no model inference or synthetic results.

The three samples are the common matched objects nearest 10, 20, and 30 m in
the saved front-view sequence. Ground-truth poses are checked against labels.
"""
from pathlib import Path
import base64
import csv
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAIRS = Path('/home/e114/mono3d_comparison_fr/pairs')
DATA = Path('/home/e114/data/KITTIDataset_Carla/training')
methods = ['gemo3d', 'monoamnet', 'monodgp', 'monodetr']
rows = {m: list(csv.DictReader((PAIRS / f'{m}_pairs.csv').open(encoding='utf-8-sig'))) for m in methods}
common = set.intersection(*[{(r['image_id'], r['gt_id']) for r in rows[m]} for m in methods])
edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

def corners(row, prefix):
    h,w,l = [float(row[f'{prefix}_{d}']) for d in ['H','W','L']]
    x,y,z = [float(row[f'{prefix}_{d}']) for d in ['x','y','z']]
    a = float(row[f'{prefix}_ry_rad'])
    c,s = np.cos(a),np.sin(a)
    local = np.array([[l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2],
                      [0,0,0,0,-h,-h,-h,-h],[w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]])
    return (np.array([[c,0,s],[0,1,0],[-s,0,c]]) @ local).T + [x,y,z]

manifest = []
for target in [10,20,30]:
    r = min((r for r in rows['gemo3d'] if (r['image_id'],r['gt_id']) in common), key=lambda r:abs(float(r['gt_z'])-target))
    fid = r['image_id']
    fields = (DATA / 'label_2' / f'{fid}.txt').read_text().splitlines()[int(r['gt_id'])].split()
    assert np.allclose([float(r[f'gt_{d}']) for d in ['H','W','L','x','y','z','ry_rad']], [float(v) for v in fields[8:15]], atol=1e-5)
    p = next(l for l in (DATA / 'calib' / f'{fid}.txt').read_text().splitlines() if l.startswith('P2:'))
    p = np.array([float(x) for x in p.split()[1:]]).reshape(3,4)
    box = corners(r,'pred')
    q = np.c_[box,np.ones(8)] @ p.T
    assert (q[:,2] > 0).all()
    uv = q[:,:2] / q[:,2,None]
    lines = ''.join(f'<path d="M{uv[i,0]:.3f} {uv[i,1]:.3f}L{uv[j,0]:.3f} {uv[j,1]:.3f}"/>' for i,j in edges)
    raw = base64.b64encode((DATA / 'image_2' / f'{fid}.png').read_bytes()).decode()
    cx,cy = uv.mean(axis=0)
    cropw = max(220, np.ptp(uv[:,0])*1.8)
    croph = cropw * 230 / 530
    cropx,cropy = max(0,cx-cropw/2),max(0,cy-croph/2)
    gt = corners(r,'gt')
    x0,z0 = float(r['gt_x']),float(r['gt_z'])
    def footprint(pts, color, dashed=False):
        points=' '.join(f'{955+(v[0]-x0)*38:.2f},{550-(v[2]-z0)*38:.2f}' for v in pts[:4])
        return f'<polygon points="{points}" fill="none" stroke="{color}" stroke-width="3"'+(' stroke-dasharray="7 5"' if dashed else '')+'/>'
    grid=''.join(f'<path d="M{955+i*38} 435V659M754 {550+i*38}H1173" stroke="#e7ecf1"/>' for i in range(-3,3))
    svg=f'''<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="1242" height="690" viewBox="0 0 1242 690" role="img" aria-labelledby="t d">
<title id="t">GEMO3D prediction at {float(r['gt_z']):.2f} m</title><desc id="d">Saved frame {fid}. Blue: predicted 3D box. Below: enlarged image region and bird's-eye-view comparison with orange dashed ground truth.</desc>
<defs><g id="scene"><image width="1242" height="374" xlink:href="data:image/png;base64,{raw}"/><g fill="none" stroke="#087bf0" stroke-width="2.5" stroke-linejoin="round">{lines}</g></g></defs>
<rect width="1242" height="690" fill="white"/><use xlink:href="#scene"/>
<g font-family="Arial,sans-serif" fill="#354456"><text x="26" y="410" font-size="17" font-weight="600">Image detail</text><text x="740" y="410" font-size="17" font-weight="600">Bird’s-eye view</text>
<text x="1208" y="410" text-anchor="end" font-size="14">1 m grid · forward ↑</text></g>
<svg x="26" y="430" width="650" height="230" viewBox="{cropx} {cropy} {cropw} {croph}" preserveAspectRatio="xMidYMid meet"><use xlink:href="#scene"/></svg>
{grid}{footprint(gt,'#ba7b35',True)}{footprint(box,'#2468aa')}
<g font-family="Arial,sans-serif" font-size="14" fill="#536171"><path d="M749 676H779" stroke="#2468aa" stroke-width="3"/><text x="787" y="681">GEMO3D</text><path d="M947 676H977" stroke="#ba7b35" stroke-width="3" stroke-dasharray="7 5"/><text x="985" y="681">Ground truth</text></g></svg>'''
    (ROOT / f'assets/prediction-{target}m.svg').write_text(svg)
    manifest.append({'target_depth_m':target,'frame':fid,'gt_id':r['gt_id'],
                     'ground_truth_depth_m':float(r['gt_z']),'prediction_depth_m':float(r['pred_z']),
                     'bev_iou':float(r['iou_bev']), 'P2':p.tolist(),
                     'prediction':{k:v for k,v in r.items() if k.startswith('pred_')},
                     'ground_truth':{k:v for k,v in r.items() if k.startswith('gt_')}})
(ROOT/'assets/data/examples.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Validated and rendered frames:', ', '.join(m['frame'] for m in manifest))
