from pathlib import Path
import urllib.request,json,struct,zlib,binascii,hashlib,sqlite3,shutil
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round91'
D=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261029\terrain-workbench\docs\round91-artifacts')
entry=next(x for x in json.loads((O/'round68/zip-central-directory.json').read_text()) if x['name']=='carlisle_vml_buildings_epsg3035.gpkg')
dest=R/entry['name'];assert not dest.exists(),'Do not repeat acquisition'
start=entry['offset'];end=start+entry['compressed']+4095
url='https://api.figshare.com/v2/file/download/24425537'
req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}','User-Agent':'HYRA-local-research/1.0'})
with urllib.request.urlopen(req,timeout=60) as response:
 assert response.status==206 and response.headers['Content-Range'].startswith(f'bytes {start}-'), 'Refuse full archive download'
 headers=dict(response.headers);data=response.read(end-start+2)
assert len(data)==end-start+1
(R/'building-entry-range.bin').write_bytes(data)
head=struct.unpack('<4s5H3I2H',data[:30]);assert head[0]==b'PK\x03\x04'
method=head[3];n,x=head[-2:];name=data[30:30+n].decode('utf8');assert name==entry['name'] and method==8
payload=data[30+n+x:30+n+x+entry['compressed']];raw=zlib.decompress(payload,-15)
assert len(raw)==entry['size'] and f'{binascii.crc32(raw)&0xffffffff:08x}'==entry['crc32']
dest.write_bytes(raw)
db=sqlite3.connect('file:'+str(dest)+'?mode=ro',uri=True)
layers=db.execute('select table_name,column_name,geometry_type_name,srs_id from gpkg_geometry_columns').fetchall();db.close()
manifest={'url':url,'range':[start,end],'headers':headers,'entry':entry,'entry_crc_verified':True,'full_zip_md5_verified':False,'sha256':hashlib.sha256(raw).hexdigest(),'layers':layers,'source_paper':'https://doi.org/10.1029/2020WR028241','paper_layer_vintage':'OS VectorMapLocal2018; not proof of2005 footprint','license_metadata':'CC BY4.0 author archive; retain source attribution and embedded dataset notices','purpose':'independent footprint context for81 elevation flags, no terrain editing or acceptance exclusion','model_changes':False,'outgoing_contact':False}
(R/'building-acquisition.json').write_text(json.dumps(manifest,indent=2));shutil.copy2(__file__,R/Path(__file__).name)
for n in [entry['name'],'building-entry-range.bin','building-acquisition.json',Path(__file__).name]:shutil.copy2(R/n,D/n)
print(json.dumps({'entry_verified':True,'bytes':len(raw),'layers':layers,'sha256':manifest['sha256']}))
