"""Read-only geometry and observation-coverage audit, no model/observation fitting."""
from pathlib import Path
import hashlib, heapq, json, sqlite3, struct, sys, time
import numpy as np
from scipy.ndimage import label
OUT=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs')
R=OUT/'round78';DATA=OUT/'round73/original/Carlisle';SRC=OUT/'round77/lisflood8-rk-ledger'
sys.path.insert(0,str(OUT/'reference-geospatial-libs'))
from shapely import from_wkb
from shapely.geometry import box,mapping,Point
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,a):p.write_text(json.dumps(a,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def grid(name):
    p=DATA/name
    lines=p.read_text().splitlines();h={k.lower():float(v) for k,v in [x.split() for x in lines[:6]]}
    a=np.loadtxt(p,skiprows=6);assert a.shape==(int(h['nrows']),int(h['ncols']))
    assert np.isfinite(a).all()
    return a,h
def stats(a):
    a=np.asarray(a);return {'n':int(a.size),'min':float(a.min()),'p05':float(np.quantile(a,.05)),'median':float(np.median(a)),'p95':float(np.quantile(a,.95)),'max':float(a.max())}
def faces(z,sx,sy):
    ny,nx=z.shape;x=np.empty((ny,nx+1));y=np.empty((ny+1,nx));s3=np.sqrt(3.)
    x[:,1:-1]=np.maximum(z[:,:-1]+s3*sx[:,:-1],z[:,1:]-s3*sx[:,1:])
    y[1:-1,:]=np.maximum(z[1:,:]+s3*sy[1:,:],z[:-1,:]-s3*sy[:-1,:])
    x[:,0]=z[:,0]-s3*sx[:,0];x[:,-1]=z[:,-1]+s3*sx[:,-1]
    y[0,:]=z[0,:]+s3*sy[0,:];y[-1,:]=z[-1,:]-s3*sy[-1,:]
    return x,y
def bottleneck(mask,fx,fy):
    ny,nx=mask.shape;d=np.full(mask.shape,np.inf);prev={};heap=[]
    for j in np.flatnonzero(mask[:,0]):d[j,0]=fx[j,0];heapq.heappush(heap,(d[j,0],int(j),0))
    while heap:
        val,j,i=heapq.heappop(heap)
        if val!=d[j,i]:continue
        for jj,ii,edge in [(j,i-1,fx[j,i]),(j,i+1,fx[j,i+1]),(j-1,i,fy[j,i]),(j+1,i,fy[j+1,i])]:
            if 0<=jj<ny and 0<=ii<nx and mask[jj,ii]:
                new=max(val,float(edge))
                if new<d[jj,ii]:d[jj,ii]=new;prev[(jj,ii)]=(j,i);heapq.heappush(heap,(new,jj,ii))
    return d,prev
def gpkg_native(p):
    with sqlite3.connect(p.resolve().as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        specs=db.execute('select * from gpkg_geometry_columns').fetchall();assert len(specs)==1
        spec=dict(specs[0]);assert spec['srs_id']==27700
        quote=lambda s:'"'+s.replace('"','""')+'"'
        rows=[]
        for a in db.execute('select * from '+quote(spec['table_name'])):
            a=dict(a);blob=a.pop(spec['column_name']);assert blob[:3]==b'GP\x00'
            flags=blob[3];assert struct.unpack_from('<i' if flags&1 else '>i',blob,4)[0]==27700
            env=(flags>>1)&7;assert env<=4
            g=from_wkb(blob[8+8*(0,4,6,6,8)[env]:]);assert g.is_valid and not g.is_empty
            rows.append((a,g))
        return rows

def main():
    start=time.perf_counter();R.mkdir(exist_ok=False)
    dump(R/'protocol.json',{
      'date':'2026-10-30','inputs':'Read author2023 archive grids and independent round68 observation files, unchanged. No hydrodynamic run or scoring against model output.',
      'hypothesis':'Original raw raster and DG2 coefficients may have different channel elevations, planar face barriers and spatial support; distinguish mean shifts from face-level channel connectivity before proposing geometry changes.',
      'method':['Reproduce the published-code NE/NW/SE/SW conversion exactly, including edge repetition','All width>0 channel cells retained; 4-neighbour and8-neighbour topology separate','Runtime DG2 perimeter slope zeroing and reconstructed interface heights follow saved source','Static minimax face barrier to west boundary on channel mask is geometric only, not arrival time/discharge or full floodplain flow','Save unshifted source profiles, all channel cells, all outline/domain intersections, all263 watermarks'],
      'predefined_source_sections':'For all3 original source cells, both row and column profiles +/-20 grid cells, no inferred cross-flow orientation or discharge capacity claim.',
      'checks':['No source file hash changes','Original affine header and native27700 outline coordinates, no reprojection of outlines','All5 outlines and263 marks retained including outside-domain','Implementation checked against direct four-corner formulas and simple nonuniform planar examples'],
      'decision':'Do not replace bed or move gauges based on score. If differences change static connectivity, register a spatial-support/geometry candidate; if not, reject this as sufficient explanation. Verify SGC units and version before considering it.',
      'cost':'One light geometry/coverage audit and scientific plots, no repeat60s/800s or new long computation.',
      'limit':'This tests representational differences, not which DEM is physically correct. Unknown bank/epoch/outlet/initial-state provenance remains unresolved. No field qualitative pass added.'})
    names=['CarlislefixDEMwithbed.asc','CarlislefixDEMwithbedDG2.asc','CarlislefixDEMwithbedDG2.asc.raw','CarlislefixDEMwithbedDG2.asc1x','CarlislefixDEMwithbedDG2.asc1y','width.asc','bed.asc']
    files=[DATA/n for n in names]+[DATA/'carlisle.bci',DATA/'carlisle.bdy',DATA/'Carlisle.par',DATA/'Carlisle-sgc.par',SRC/'preprocess/generateDG2DEM.cpp',SRC/'swe/dg2/fields.cpp',SRC/'swe/dg2/modifiedvars.cpp',SRC/'sgc.cpp',SRC/'input.cpp',SRC/'pars.cpp',OUT/'round68/data/carlisle_recorded_flood_outlines.gpkg',OUT/'round68/prepared/all-survey-marks.json']
    hashes={str(p):sha(p) for p in files}
    raw,h=grid(names[0]);z,h2=grid(names[1]);r2,_=grid(names[2]);sx,_=grid(names[3]);sy,_=grid(names[4]);width,_=grid('width.asc');bed,_=grid('bed.asc')
    assert h==h2 and np.array_equal(raw,r2)
    ny,nx=raw.shape;mask=width>0;zero=np.zeros_like(raw)
    nw=raw;ne=np.c_[raw[:,1:],raw[:,-1:]];sw=np.r_[raw[1:,:],raw[-1:,:]];se=np.c_[sw[:,1:],sw[:,-1:]]
    generated=[.25*(nw+ne+sw+se),(ne+se-nw-sw)/(4*np.sqrt(3)),(ne+nw-se-sw)/(4*np.sqrt(3))]
    errors=[float(np.max(np.abs(a-b))) for a,b in zip(generated,[z,sx,sy])]
    runtime_x=sx.copy();runtime_y=sy.copy();runtime_x[:,[0,-1]]=0;runtime_y[[0,-1],:]=0
    rawfx,rawfy=faces(raw,zero,zero);fx,fy=faces(z,runtime_x,runtime_y)
    c4,n4=label(mask,np.array([[0,1,0],[1,1,1],[0,1,0]]));c8,n8=label(mask,np.ones((3,3)))
    rawbar,rawprev=bottleneck(mask,rawfx,rawfy);dgbar,dgprev=bottleneck(mask,fx,fy)
    # Formula sanity independent of observed geometry, not a hydrodynamic test.
    zz=np.array([[1.,2.],[3.,4.]]);xx,yy=faces(zz,np.zeros_like(zz),np.zeros_like(zz))
    assert np.array_equal(xx[:,1],[2.,4.]) and np.array_equal(yy[1,:],[3.,4.])
    fullmask=np.ones_like(zz,dtype=bool);b,_=bottleneck(fullmask,xx,yy)
    assert np.array_equal(b,zz)
    edgesx=mask[:,:-1]&mask[:,1:];edgesy=mask[:-1,:]&mask[1:,:]
    delta_edges=np.r_[ (fx[:,1:-1]-rawfx[:,1:-1])[edgesx], (fy[1:-1,:]-rawfy[1:-1,:])[edgesy] ]
    channel_records=[]
    for j,i in zip(*np.where(mask)):
        channel_records.append(dict(row=int(j),col=int(i),x=h['xllcorner']+(i+.5)*h['cellsize'],y=h['yllcorner']+(ny-j-.5)*h['cellsize'],raw_bed=float(raw[j,i]),dg2_mean=float(z[j,i]),dg2_slope_x=float(sx[j,i]),dg2_slope_y=float(sy[j,i]),width=float(width[j,i]),subgrid_bed=float(bed[j,i]),component4=int(c4[j,i]),component8=int(c8[j,i]),raw_static_path_threshold=float(rawbar[j,i]) if np.isfinite(rawbar[j,i]) else None,dg2_static_path_threshold=float(dgbar[j,i]) if np.isfinite(dgbar[j,i]) else None))
    dump(R/'all-channel-cells.json',channel_records)
    inputs=[];profiles={}
    for line in (DATA/'carlisle.bci').read_text().splitlines():
        t=line.split()
        if t and t[0]=='P':
            x,y=map(float,t[1:3]);i=int((x-h['xllcorner'])/h['cellsize']);j=int((h['yllcorner']+ny*h['cellsize']-y)/h['cellsize'])
            assert mask[j,i]
            inputs.append(dict(name=t[4],x=x,y=y,row=j,col=i,raw_bed=float(raw[j,i]),dg2_mean=float(z[j,i]),subgrid_bed=float(bed[j,i]),width_m=float(width[j,i]),raw_path_threshold=float(rawbar[j,i]) if np.isfinite(rawbar[j,i]) else None,dg2_path_threshold=float(dgbar[j,i]) if np.isfinite(dgbar[j,i]) else None,
                raw_faces=[float(rawfx[j,i]),float(rawfx[j,i+1]),float(rawfy[j,i]),float(rawfy[j+1,i])],dg2_faces=[float(fx[j,i]),float(fx[j,i+1]),float(fy[j,i]),float(fy[j+1,i])]))
            profiles[t[4]]={}
            for axis in ['row','column']:
                coords=[(j,ii) for ii in range(max(0,i-20),min(nx,i+21))] if axis=='row' else [(jj,i) for jj in range(max(0,j-20),min(ny,j+21))]
                profiles[t[4]][axis]=[dict(row=jj,col=ii,raw=float(raw[jj,ii]),dg2=float(z[jj,ii]),sx=float(runtime_x[jj,ii]),sy=float(runtime_y[jj,ii]),width=float(width[jj,ii]),channel_bed=float(bed[jj,ii])) for jj,ii in coords]
    dump(R/'source-grid-profiles.json',profiles)
    domain=box(h['xllcorner'],h['yllcorner'],h['xllcorner']+nx*h['cellsize'],h['yllcorner']+ny*h['cellsize'])
    outlines=gpkg_native(OUT/'round68/data/carlisle_recorded_flood_outlines.gpkg');coverage=[];features=[]
    for a,g in outlines:
        clip=g.intersection(domain);outside=g.difference(domain)
        record=dict(id=str(a['outline_co']),name=a['name'],source=a,source_area_m2=g.area,domain_intersection_m2=clip.area,outside_domain_m2=outside.area,coverage='outside' if clip.is_empty else 'partial' if outside.area>1e-6 else 'inside',observed='event_maximum_reported_inundation',outside_outline='unknown_not_dry',model_comparison='unscored_no_full_event_result')
        coverage.append(record);features.append({'type':'Feature','properties':record,'geometry':mapping(g)})
    marks=json.loads((OUT/'round68/prepared/all-survey-marks.json').read_text('utf-8'));markrows=[]
    for a in marks:
        v=a['source_attributes'];x,y=v['E'],v['N'];inside=h['xllcorner']<=x<h['xllcorner']+nx*h['cellsize'] and h['yllcorner']<=y<h['yllcorner']+ny*h['cellsize']
        markrows.append(dict(a,inside_model_domain=inside,model_cell=[int((h['yllcorner']+ny*h['cellsize']-y)/h['cellsize']),int((x-h['xllcorner'])/h['cellsize'])] if inside else None,model_comparison='unscored_no_full_event_result'))
    dump(R/'all-survey-mark-coverage.json',markrows)
    dump(R/'all-outline-coverage.json',coverage)
    dump(R/'observation-objects-bng.geojson',{'type':'FeatureCollection','crs':{'type':'name','properties':{'name':'urn:ogc:def:crs:EPSG::27700'}},'features':features})
    np.savez_compressed(R/'geometry-arrays.npz',raw=raw,dg2=z,sx=sx,sy=sy,width=width,bed=bed,mask=mask,dg2_face_x=fx,dg2_face_y=fy,raw_path_threshold=rawbar,dg2_path_threshold=dgbar)
    result={'grid':h,'all_input_hashes':hashes,'runtime_slope_rule':'x slope zero on west/east cell columns, y slope zero on north/south rows, before Zstar',
      'conversion_fingerprint':dict(formula='DG2 uses raw[r,c],raw[r,c+1],raw[r+1,c],raw[r+1,c+1], repeated last edge, as NW,NE,SW,SE',max_abs_errors_mean_sx_sy=errors,matched_within_1e_12=bool(max(errors)<1e-12),not_a_coordinate_truth_claim=True),
      'channel':dict(count=int(mask.sum()),component4_sizes=sorted([int((c4==k).sum()) for k in range(1,n4+1)],reverse=True),component8_sizes=sorted([int((c8==k).sum()) for k in range(1,n8+1)],reverse=True),west_boundary_cells=int(mask[:,0].sum()),west_connected_cells=int(np.isfinite(rawbar[mask]).sum()),dg2_mean_minus_raw_m=stats((z-raw)[mask]),raw_minus_subgrid_bed_m=stats((raw-bed)[mask]),width_m=stats(width[mask]),dg2_minus_raw_interface_barrier_m=stats(delta_edges)),
      'sources':inputs,'domain_coverage':{'outlines':coverage,'marks_total':len(markrows),'marks_inside':sum(x['inside_model_domain'] for x in markrows),'marks_outside':sum(not x['inside_model_domain'] for x in markrows)},
      'field_qualitative_passes_added':0,'production_changed':False,'time_epoch':'unresolved','sgc_source_units':'Local sgc.cpp SGC_UpdateH multiplies QVAR by dx on regular grid, matching m2/s semantics; not proof same improved algorithm as2023 paper.',
      'limits':['Static mask-limited path threshold is not dynamic inundation or actual discharge capacity','Equal conversion fingerprint does not prove correct georeferencing','No point shifting, terrain correction or observation exclusion','Independent dry-area and road/site evidence remains missing']}
    # Native data and fixed source section diagnostic; no GIS raster approximation used.
    extent=[h['xllcorner'],h['xllcorner']+nx*h['cellsize'],h['yllcorner'],h['yllcorner']+ny*h['cellsize']]
    fig,ax=plt.subplots(1,2,figsize=(13,5.3),constrained_layout=True)
    plot=np.ma.masked_where(~mask,z-raw);im=ax[0].imshow(plot,extent=extent,origin='upper',cmap='coolwarm',vmin=-5,vmax=5)
    ax[0].set_title('Channel cells: DG2 mean minus raw (m)\nDisplay clipped +/-5m; all values retained');fig.colorbar(im,ax=ax[0])
    for s in inputs:ax[0].plot(s['x'],s['y'],'ko');ax[0].annotate(s['name'],(s['x'],s['y']),xytext=(3,6),textcoords='offset points',fontsize=8)
    for s in inputs:
        j,i=s['row'],s['col'];a=profiles[s['name']]['row'];xx=[(x['col']-i)*10 for x in a]
        ax[1].plot(xx,[x['raw']-s['raw_bed'] for x in a],label=s['name']+' raw',linewidth=1.3)
        ax[1].plot(xx,[x['dg2']-s['raw_bed'] for x in a],'--',label=s['name']+' DG2',linewidth=1)
    ax[1].set_title('Fixed source rows, relative to each raw source bed\nNot discharge-normal cross sections');ax[1].set_xlabel('Grid-row offset east (m)');ax[1].set_ylabel('Elevation difference (m)');ax[1].legend(fontsize=7,ncol=2)
    for x in ax:x.grid(alpha=.2)
    fig.savefig(R/'channel-geometry.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,7),constrained_layout=True)
    colors=plt.cm.tab10(np.arange(len(outlines)))
    for idx,((a,g),co) in enumerate(zip(outlines,colors)):
        geoms=list(g.geoms) if hasattr(g,'geoms') else [g]
        for n,gg in enumerate(geoms):
            x,y=gg.exterior.xy;ax.fill(x,y,color=co,alpha=.3,label=a['name'] if n==0 else None);ax.plot(x,y,color=co,linewidth=.6)
    xx,yy=domain.exterior.xy;ax.plot(xx,yy,'k--',linewidth=1.6,label='Fixed author model domain')
    ax.scatter([x['source_attributes']['E'] for x in markrows],[x['source_attributes']['N'] for x in markrows],s=5,c='black',label='All263 survey marks')
    ax.set_aspect('equal');ax.ticklabel_format(style='plain');ax.set_xlabel('Easting EPSG27700 (m)');ax.set_ylabel('Northing EPSG27700 (m)');ax.set_title('Independent maximum flood outlines and model coverage\nWhite area has no dry-ground classification');ax.legend(fontsize=7,loc='upper left');fig.savefig(R/'observation-coverage.png',dpi=160);plt.close(fig)
    assert all(sha(Path(p))==v for p,v in hashes.items())
    result['inputs_unchanged']=True;result['wall_s']=time.perf_counter()-start
    dump(R/'geometry-coverage-audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['all_input_hashes','limits','grid','runtime_slope_rule']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
