"""Run from any working directory: python python/run_yeonwoo.py
Full proposal denoising sweep: 3 images x 10 seeds x 4 noise conditions.
No parameter is fitted on a separate holdout: best means descriptive sweep best.
"""
from pathlib import Path
import csv, json, platform, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.metrics import structural_similarity
import my_filters as M
import img_utils as U
from degradation4gaussian import gaussian_noise, salt_pepper, gaussian_blur
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/'results'/'gaussian'
OUT.mkdir(exist_ok=True)

def write_csv(name, rows):
    with (OUT/name).open('w', newline='', encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    rng=np.random.default_rng(100)
    x=rng.uniform(0,255,(31,43)); h=rng.normal(size=(3,5))
    err=float(np.max(np.abs(M.conv2d(x,h)-ndimage.convolve(x,h,mode='nearest'))))
    assert err < 1e-10
    checks={'asymmetric_conv_max_error':err,'kernels':[]}
    sigmas=[0.5,0.8,1.0,1.2,1.5,2.0,3.0]
    filters=[(f'box{k}', M.box_kernel(k)) for k in [3,5,7]]
    filters += [(f'gauss{s}',M.gaussian_kernel(s)) for s in sigmas]
    for s in sigmas:
        g=M.gaussian_kernel(s)
        assert abs(g.sum()-1)<1e-12 and np.allclose(g,g[::-1,::-1])
        assert np.max(np.abs(M.conv2d(np.ones((8,9))*77,g)-77))<1e-10
        checks['kernels'].append({'sigma':s,'size':len(g),'sum':float(g.sum())})
    assert np.array_equal(gaussian_noise(x,20,2),gaussian_noise(x,20,2))
    assert np.array_equal(gaussian_noise(x,0),x)
    assert np.array_equal(salt_pepper(x,0),x)
    (OUT/'validation.json').write_text(json.dumps(checks,indent=2))
    raw=[]
    for name in ['camera','astronaut','coffee']:
        clean=U.read_gray(ROOT/'data/images_clean'/f'{name}.png')
        for condition in ['gauss10','gauss20','gauss30','sp5']:
            print(name,condition,flush=True)
            for seed in range(10):
                inp=salt_pepper(clean,.05,seed) if condition=='sp5' else gaussian_noise(clean,float(condition[5:]),seed)
                base=U.psnr(clean,inp)
                candidates=[(label,M.conv2d(inp,k)) for label,k in filters]
                candidates += [('median3',M.median_filter(inp,3)),('median5',M.median_filter(inp,5))]
                for label,y in candidates:
                    p=U.psnr(clean,y)
                    raw.append(dict(image=name,condition=condition,seed=seed,filter=label,input_psnr=base,psnr=p,delta_psnr=p-base,ssim=structural_similarity(clean,y,data_range=255),sharpness_ratio=U.sharpness(y)/U.sharpness(clean)))
                if name=='camera' and seed==0 and condition=='gauss20':
                    panels=[('Clean',clean),('Noisy',inp)] + [(lab,y) for lab,y in candidates if lab in ['box3','gauss0.8','gauss1.5','gauss3.0']]
                    fig,axs=plt.subplots(2,3,figsize=(12,8))
                    for ax,(lab,y) in zip(axs.flat,panels):
                        ax.imshow(y[100:260,180:340],cmap='gray',vmin=0,vmax=255);ax.set_title(f'{lab}: {U.psnr(clean,y):.2f} dB');ax.axis('off')
                    fig.tight_layout();fig.savefig(OUT/'denoise_tradeoff.png',dpi=140);plt.close(fig)
                    for lab,y in panels:
                        U.save_gray(OUT/f'camera_{lab}.png',y)
        U.save_gray(OUT/f'{name}_blur_sigma2.png',gaussian_blur(clean,2))
    write_csv('raw_metrics.csv',raw)
    summary=[]
    for name in ['camera','astronaut','coffee']:
        for c in ['gauss10','gauss20','gauss30','sp5']:
            for label in sorted(set(r['filter'] for r in raw)):
                rr=[r for r in raw if r['image']==name and r['condition']==c and r['filter']==label]
                row=dict(image=name,condition=c,filter=label,n=len(rr))
                for metric in ['input_psnr','psnr','delta_psnr','ssim','sharpness_ratio']:
                    vals=[r[metric] for r in rr];row[metric+'_mean']=float(np.mean(vals));row[metric+'_std']=float(np.std(vals,ddof=1))
                summary.append(row)
    write_csv('summary_metrics.csv',summary)
    best=[]
    for name in ['camera','astronaut','coffee']:
        for c in ['gauss10','gauss20','gauss30','sp5']:
            rr=[r for r in summary if r['image']==name and r['condition']==c]
            for family in ['linear','all']:
                chosen=max([r for r in rr if family=='all' or not r['filter'].startswith('median')],key=lambda r:r['psnr_mean'])
                best.append(dict(family=family,**chosen,passes_3db_mean=chosen['delta_psnr_mean']>=3))
    write_csv('best_settings.csv',best)
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    for ax,name in zip(axs,['camera','astronaut','coffee']):
        rr=sorted([r for r in summary if r['image']==name and r['condition']=='gauss20' and r['filter'].startswith('gauss')],key=lambda r:float(r['filter'][5:]))
        ax.errorbar([float(r['filter'][5:]) for r in rr],[r['delta_psnr_mean'] for r in rr],yerr=[r['delta_psnr_std'] for r in rr],marker='o',capsize=3)
        ax.axhline(3,color='gray',ls='--');ax.set_title(name);ax.set_xlabel('Filter sigma (pixels)');ax.set_ylabel('Delta PSNR (dB)');ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(OUT/'sigma_sweep.png',dpi=140);plt.close(fig)
    freq=[];fig,ax=plt.subplots(figsize=(8,4))
    for s in sigmas:
        g=M.gaussian_kernel(s);H=np.abs(np.fft.fft2(g,(1024,1024)))[0,:513]
        w=np.arange(513)*2*np.pi/1024;db=20*np.log10(np.maximum(H,1e-15));cross=np.where(db<=-3)[0]
        wc=None
        if len(cross):
            i=cross[0];wc=float(np.interp(-3,[db[i],db[i-1]],[w[i],w[i-1]]))
        freq.append(dict(sigma=s,size=len(g),cutoff_3db_rad=wc,gain_pi_over2_db=float(db[256]),gain_pi_db=float(db[512])))
        ax.plot(w,db,label=f'sigma={s}')
    ax.set(xlim=(0,np.pi),ylim=(-100,2),xlabel='Angular frequency (rad/pixel)',ylabel='Magnitude (dB)',title='Gaussian kernel: exact 1024-point DFT axis cut');ax.legend();ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(OUT/'gaussian_frequency.png',dpi=140);plt.close(fig);write_csv('gaussian_frequency.csv',freq)
    # Flat field: unclipped white noise variance reduction is sum(h^2), interior only.
    theory=[]
    z=np.random.default_rng(123).normal(0,20,(512,512))
    for s in sigmas:
        g=M.gaussian_kernel(s);r=len(g)//2;y=M.conv2d(z,g);inter=y[r:-r,r:-r]
        theory.append(dict(sigma=s,predicted_variance=float(400*np.sum(g*g)),measured_variance=float(np.var(inter))))
    write_csv('noise_variance_check.csv',theory)
    (OUT/'environment.json').write_text(json.dumps(dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),seeds=list(range(10)),evaluation='float64 clipped input, no PNG quantization before metrics',implementation='direct conv2d; scipy only as validation oracle'),indent=2))
    print('COMPLETE',OUT,flush=True)
if __name__=='__main__':main()
