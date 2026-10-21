import json,sys,urllib.request,pathlib,concurrent.futures as cf
m=json.load(open('data/manifest.json'));out=pathlib.Path('data/images')
def go(i):
    p=out/f"{i['barcode']}.jpg"
    if p.exists() and p.stat().st_size>1000: return i['barcode'],'ok'
    for u in (i['image_url'],i['image_url_400']):
        try:
            r=urllib.request.Request(u,headers={'User-Agent':'opencv-label-guard-eval/0.1 (contest research)'})
            b=urllib.request.urlopen(r,timeout=30).read(); p.write_bytes(b); return i['barcode'],'ok' if u==i['image_url'] else 'fallback400'
        except Exception as e: err=str(e)
    return i['barcode'],'fail:'+err
with cf.ThreadPoolExecutor(6) as ex: res=list(ex.map(go,m['items']))
from collections import Counter;print(Counter(r[1][:8] for r in res));print([r for r in res if r[1].startswith('fail')])
