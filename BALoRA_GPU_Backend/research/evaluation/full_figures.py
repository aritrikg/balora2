"""Scientific figures drawn only from the completed comparison and saved evidence."""
from pathlib import Path
import json,csv,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.runtime_budget import check_deadline,write_json


def export_full_figures(root,context,*,deadline=None):
    root=Path(root);out=root/'figures';out.mkdir(exist_ok=True)
    ref=Path(__file__).resolve().parents[1]/'figure_reference'
    historical=json.loads((ref/'results.json').read_text())
    selected=historical['selected_checkpoint']
    # The bundled historical curve/rank/audit belongs to one selected checkpoint.
    original_digest=historical['checkpoint'].get('sha256')
    if original_digest is None:
        original_digest=selected.get('sha256',selected.get('checkpoint_sha256'))
    expected=context['reference']['checkpoint_sha256']
    if original_digest!=expected:
        raise ValueError('Historical figure reference checkpoint differs from completed experiment')
    result=json.loads((root/'method_comparison.json').read_text())
    if not result['complete']:raise ValueError('Full comparison figures require completed evaluation')
    def read(name):
        with (root/name).open() as f:return list(csv.DictReader(f))
    methods=read('method_comparison.csv');organs=read('per_organ_comparison.csv');pairs=read('per_volume_comparison.csv')
    manifest=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','pdf.fonttype':42})
    def save(fig,name,caption):
        check_deadline(deadline,'figure export')
        for ext in ('png','svg','pdf'):fig.savefig(out/(name+'.'+ext),dpi=350,bbox_inches='tight',facecolor='white')
        plt.close(fig);manifest.append({'figure':name,'caption':caption})
    for ext in ('png','svg','pdf'):shutil.copy2(ref/('figure_1_method.'+ext),out/('figure_1_method.'+ext))
    manifest.append({'figure':'figure_1_method','caption':'Implemented Bayesian pipeline schematic; not an experimental image. Oracle positive point; two probe passes plus K final draws.'})
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    h=historical['history'];hc=json.loads((root/'training_history.json').read_text())
    for ax,key,label in zip(axs,('train_loss','val_dice'),('Mean training objective','Validation Dice')):
        ax.plot(np.array(h['epoch'])+1,h[key],'-o',ms=3,label='BALoRA')
        ax.plot([x['epoch']+1 for x in hc],[x[key] for x in hc],'-s',ms=3,label='Deterministic LoRA')
        ax.set(xlabel='Completed epoch',ylabel=label);ax.legend();ax.grid(alpha=.2)
    save(fig,'figure_2_training','Actual completed epochs for two separately trained methods. Frozen SAM2 has no training curve. Validation selects checkpoints; no test values appear.')
    keys=('sam2_dice','deterministic_lora_dice','balora_dice');labels=('Frozen SAM2','Deterministic LoRA','BALoRA no refinement');colors=('#b84c23','#648147','#17679b')
    fig,ax=plt.subplots(figsize=(10,6),layout='constrained');y=np.arange(len(organs))
    for key,label,c in zip(keys,labels,colors):ax.plot([float(r[key]) for r in organs],y,'o',label=label,color=c)
    ax.set_yticks(y,[r['organ'].replace('_',' ') for r in organs]);ax.invert_yaxis();ax.set(xlim=(0,1),xlabel='Mean Dice across six volume-organ means');ax.legend()
    save(fig,'figure_3_organs','Test Dice. Each organ averages six volume-organ means after within-pair slice averaging. No seed error bars are implied.')
    cases=sorted({r['case_id'] for r in pairs});fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    for key,label,c in zip(keys,labels,colors):
        ax.plot(cases,[np.mean([float(r[key]) for r in pairs if r['case_id']==v]) for v in cases],'o-',label=label,color=c)
    ax.set(ylabel='Mean Dice across 13 organs',xlabel='Independent test volume');ax.legend();ax.grid(alpha=.2)
    save(fig,'figure_4_volumes','Paired test-volume means. Six volumes are assumed independent; organ pairs are dependent within a volume. Lines only connect displayed records.')
    for name,metrics,unit in [('figure_5_probability',('ece','brier','nll'),'All-pixel probability score'),('figure_6_surfaces',('hd95_pixels','assd_pixels'),'Resized 2D pixels')]:
        fig,axs=plt.subplots(1,len(metrics),figsize=(11,4),layout='constrained')
        for ax,key in zip(np.atleast_1d(axs),metrics):
            ax.bar(range(3),[float(r[key]) for r in methods],color=colors);ax.set_xticks(range(3),labels,rotation=25,ha='right');ax.set(title=key.upper(),ylabel=unit)
        save(fig,name,'Completed test scores average 78 volume-organ means. Probability metrics include background and do not validate epistemic reliability; surface distances are 2D pixels, not 3D millimetres.')
    with (ref/'validation_audit_rounded.csv').open() as f:audit=list(csv.DictReader(f))
    fig,ax=plt.subplots(figsize=(9,6),layout='constrained')
    for key,label in [('none','None'),('legacy','Legacy'),('guarded','Guarded')]:ax.plot([float(r[key]) for r in audit],range(len(audit)),'o',label=label)
    ax.set_yticks(range(len(audit)),[r['organ'].replace('_',' ') for r in audit]);ax.invert_yaxis();ax.set(xlabel='Validation organ mean Dice');ax.legend()
    save(fig,'figure_7_validation_refinement','Historical validation-only refinement audit, five volumes and 371 foreground slices. Values have printed six-decimal precision; per-organ volume denominators were not saved in the copied evidence. No refinement was retained. This is separate from test evaluation.')
    counts={k:sum(int(v['counts'].get(str(k),0)) for v in historical['rank_histogram'].values()) for k in (4,8,16)}
    assert sum(counts.values())==2466*32
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained');ax.bar([str(k) for k in counts],list(counts.values()));ax.set(xlabel='Selected rank',ylabel='Initial layer-level choices')
    save(fig,'figure_8_rank_use','Historical BALoRA initial rank use: 2,466 prompts times 32 layers. Low ranks do not establish memory or speed gains; maximum-rank parameters and probe passes remain present.')
    fig,axs=plt.subplots(1,2,figsize=(12,7),layout='constrained');names=[r['organ'] for r in organs]
    for ax,base,title in zip(axs,('sam2_dice','deterministic_lora_dice'),('BALoRA minus frozen','BALoRA minus deterministic')):
        grid=np.array([[100*(float(next(r for r in pairs if r['organ']==o and r['case_id']==v)['balora_dice'])-float(next(r for r in pairs if r['organ']==o and r['case_id']==v)[base])) for v in cases] for o in names])
        im=ax.imshow(grid,vmin=-60,vmax=60,cmap='RdBu');ax.set_xticks(range(len(cases)),cases,rotation=45,ha='right');ax.set_yticks(range(len(names)),[o.replace('_',' ') for o in names]);ax.set_title(title)
    fig.colorbar(im,ax=list(axs),label='Dice difference in percentage points',shrink=.65)
    save(fig,'figure_9_pair_differences','All 78 test volume-organ differences per comparison, including negative outcomes. Six volumes, not 78 pairs, are independent sampling units; no significance is inferred.')
    write_json(root/'figure_manifest.json',{'complete':True,'figure_sets':manifest,'test_volumes':len(cases),'test_volume_organ_pairs':len(pairs),'test_predictions_per_method':result['n_predictions_per_method'],'qualitative_masks':'NOT REPORTED: historical selected-checkpoint pixel arrays unavailable','uncertainty_maps':'NOT REPORTED: historical selected-checkpoint pixel arrays unavailable'})
