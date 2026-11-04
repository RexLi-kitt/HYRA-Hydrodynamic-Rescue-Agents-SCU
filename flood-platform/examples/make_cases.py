from pathlib import Path
import argparse,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hyra_flood.contracts import SCHEMA,COORDINATES,UNITS,write_json,sha

def make(folder,name):
    folder=Path(folder)/name;folder.mkdir(parents=True,exist_ok=False)
    ny,nx=12,40;dx=.1;z,x=np.indices((ny,nx));bed=np.zeros((ny,nx));h=np.full((ny,nx),.2)
    if name=='dam_x':h=np.where(x<nx//2,.4,.1)
    elif name=='dam_y':h=np.where(z<ny//2,.4,.1)
    elif name!='lake':raise ValueError('Unknown case')
    np.savez_compressed(folder/'input.npz',bed_m=bed,depth_m=h,qx_m2_s=bed,qy_m2_s=bed,manning=bed)
    c=dict(schema=SCHEMA,id=name,evidence=dict(category='analytic' if name=='lake' else 'synthetic',
        source='constant-depth lake-at-rest exact equilibrium' if name=='lake' else 'prescribed closed rectangular dam-break initial state; engineering/software comparison only',role='verification'),
        coordinates=COORDINATES,units=UNITS,grid=dict(shape=[ny,nx],dx_m=dx,origin_xy_m=[100.,200.]),
        arrays=dict(path='input.npz',sha256=sha(folder/'input.npz')),boundaries={k:dict(kind='reflective') for k in ['west','east','south','north']},
        time=dict(end_s=.2,output_interval_s=.05),numerics=dict(cfl=.4,max_dt_s=.005,native_flux='rusanov',anuga_algorithm='DE1'),
        gauges=[dict(id='south_west',x_m=100.05,y_m=200.05),dict(id='north_east',x_m=103.95,y_m=201.15)])
    write_json(folder/'case.json',c)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args()
    for name in ('lake','dam_x','dam_y'):make(a.output,name)
