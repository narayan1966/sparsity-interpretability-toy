import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
BLUE='#1f6fb5'; GRAY='#dcdcdc'; GREEN='#2e9e3e'; ORANGE='#f08a24'; RED='#c0262d'; LB='#a9c8e8'
plt.rcParams['font.family']='DejaVu Sans'
fig=plt.figure(figsize=(13,8.2),dpi=200)
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,130); ax.set_ylim(0,82); ax.axis('off')
rng=np.random.default_rng(7)

def grid(x,y,n,s,cols,outline=None,lw=1.6):
    # cols: dict (r,c)->color; default gray. (x,y)=top-left
    for r in range(n):
        for c in range(n):
            col=cols.get((r,c),GRAY)
            ax.add_patch(Rectangle((x+c*s,y-(r+1)*s),s,s,fc=col,ec='white',lw=1.2))
    if outline:
        for (r,c) in outline:
            ax.add_patch(Rectangle((x+c*s+.12,y-(r+1)*s+.12),s-.24,s-.24,fc='none',ec=ORANGE,lw=lw))
def rand_cells(n,k,exclude=()):
    allc=[(r,c) for r in range(n) for c in range(n) if (r,c) not in exclude]
    idx=rng.choice(len(allc),k,replace=False); return [allc[i] for i in idx]
def t(x,y,s,**k): ax.text(x,y,s,**k)

# ---- (a)
t(1,80,'(a)',fontsize=15,fontweight='bold',va='center'); t(6,80,'Always-sparse weights',fontsize=15,va='center')
s=2.6; grid(3,72,8,s,{(r,c):BLUE for r in range(8) for c in range(8)})
t(3+4*s,73.5,'Dense',ha='center',fontsize=12)
A8=rand_cells(8,16); grid(30,72,8,s,{p:BLUE for p in A8})
t(30+4*s,73.5,'Sparse (75%)',ha='center',fontsize=12)
t(27,48.5,'Unstructured: any connection can be active.\nWeights outside the mask are fixed at zero and never trained.',ha='center',va='top',fontsize=10.5)

# ---- (b)
t(59,80,'(b)',fontsize=15,fontweight='bold',va='center'); t(64.5,80,'One prune-grow round',fontsize=15,va='center')
n=6; s=2.4; xs=[60,78,96,114]; ytop=67
act=rand_cells(6,9); inact=[(r,c) for r in range(6) for c in range(6) if (r,c) not in act]
sub=[inact[i] for i in rng.choice(len(inact),9,replace=False)]
grown=[sub[i] for i in rng.choice(9,4,replace=False)]
pruned=[act[i] for i in rng.choice(9,4,replace=False)]
titles=['1\nActive set A','2\nSample subset S','3\nGrow','4\nPrune']
for x,tt in zip(xs,titles): t(x+3*s,ytop+6.6,tt,ha='center',va='center',fontsize=11)
grid(xs[0],ytop,n,s,{p:BLUE for p in act})
grid(xs[1],ytop,n,s,{p:BLUE for p in act},outline=sub)
c3={p:BLUE for p in act}; c3.update({p:GREEN for p in grown})
grid(xs[2],ytop,n,s,c3,outline=[p for p in sub if p not in grown] if False else None)
c4={p:BLUE for p in act if p not in pruned}; c4.update({p:GREEN for p in grown})
grid(xs[3],ytop,n,s,c4)
for (r,c) in pruned:
    cx=xs[3]+c*s; cy=ytop-(r+1)*s
    ax.add_patch(Rectangle((cx,cy),s,s,fc=GRAY,ec='white',lw=1.2))
    ax.plot([cx+.35,cx+s-.35],[cy+.35,cy+s-.35],color=RED,lw=2.2); ax.plot([cx+.35,cx+s-.35],[cy+s-.35,cy+.35],color=RED,lw=2.2)
# step 3: show sampled-but-not-grown faintly outlined
for (r,c) in sub:
    if (r,c) not in grown:
        ax.add_patch(Rectangle((xs[2]+c*s+.12,ytop-(r+1)*s+.12),s-.24,s-.24,fc='none',ec=ORANGE,lw=1.0,ls=(0,(2,2)),alpha=.7))
for i in range(3):
    ax.add_patch(FancyArrowPatch((xs[i]+6*s+.6,ytop-3*s),(xs[i+1]-.6,ytop-3*s),arrowstyle='-|>',mutation_scale=12,color='k',lw=1.2))
t(xs[1]+3*s,ytop-6*s-1.2,'random subset of the\ninactive connections',ha='center',va='top',fontsize=9.5)
t(xs[2]+3*s,ytop-6*s-1.2,'largest-gradient ones in S\nare grown (new weight\nstarts at 0)',ha='center',va='top',fontsize=9.5)
t(xs[3]+3*s,ytop-6*s-1.2,'smallest-magnitude\nactive weights pruned\n(never just-grown)',ha='center',va='top',fontsize=9.5)
t(xs[0]+3*s,ytop-6*s-1.2,'9 active',ha='center',va='top',fontsize=9.5)
t(94,43.2,'Growth and pruning use equal numbers (here 4 and 4),\nso the number of active connections never changes.',ha='center',va='top',fontsize=10.5)

# ---- (c)
t(1,37,'(c)',fontsize=15,fontweight='bold',va='center'); t(6,37,'Training schedule',fontsize=15,va='center')
x0,x1=5,56; ty=29
ax.plot([x0,x1],[ty,ty],color='k',lw=1.2)
for k in range(21):
    xx=x0+(x1-x0)*k/20; ax.plot([xx,xx],[ty-.8,ty+.8],color='k' if k<=12 else '#999',lw=.9)
ax.add_patch(Rectangle((x0+(x1-x0)*.6,ty-1.2),(x1-x0)*.4,2.4,fc='#eee',ec='none',zorder=0))
xf=x0+(x1-x0)*.6; ax.plot([xf,xf],[ty-1.6,ty+1.6],color='k',lw=3)
t((x0+xf)/2,ty+1.8,'prune-grow round every 100 steps\n(ticks not to scale)',ha='center',va='bottom',fontsize=9)
t(x1,ty+1.8,'weights keep training on\na fixed set of connections',ha='right',va='bottom',fontsize=9,color='#777')
# curve axes
cx0,cx1,cy0,cy1=9,56,7,25
ax.plot([cx0,cx0],[cy0,cy1],color='k',lw=1.2); ax.plot([cx0,cx1],[cy0,cy0],color='k',lw=1.2)
xx=np.linspace(0,.6,100); yy=.2*0.5*(1+np.cos(np.pi*xx/.6))
X=cx0+(cx1-cx0)*np.r_[xx,1]; Y=cy0+(cy1-cy0)*np.r_[yy,0]/0.2
ax.add_patch(Rectangle((cx0+(cx1-cx0)*.6,cy0),(cx1-cx0)*.4,cy1-cy0,fc='#e3e3e3',ec='none'))
ax.plot(X,Y,color=BLUE,lw=2.6)
t(cx0-1,cy1,'0.20',ha='right',va='center',fontsize=10); t(cx0-1,cy0,'0',ha='right',va='center',fontsize=10)
t(cx0,cy0-1.8,'0',ha='center',fontsize=10); t(cx0+(cx1-cx0)*.6,cy0-1.8,'12,000',ha='center',fontsize=10); t(cx1,cy0-1.8,'20,000',ha='center',fontsize=10)
t((cx0+cx1)/2,cy0-4.5,'training step',ha='center',fontsize=10.5)
t(2.5,(cy0+cy1)/2,'fraction of connections\nreplaced per round',rotation=90,ha='center',va='center',fontsize=10)
t(cx0+(cx1-cx0)*.8,cy1-2,'step 12,000\n(60% of training):\ntopology frozen',ha='center',va='top',fontsize=9)

# ---- (d)
t(65,37,'(d)',fontsize=15,fontweight='bold',va='center'); t(70,37,'Applied to the toy network',fontsize=15,va='center')
for xx_,lab in [(66,'input x'),(112,'output')]: pass
t(66,31.5,'input x',ha='center',va='center',fontsize=11)
for xb,lab in [(74,'W1'),(88,'ReLU'),(102,'W2')]:
    ax.add_patch(Rectangle((xb,29.8),9,3.4,fc='#dbe8f6',ec='k',lw=1.2)); t(xb+4.5,31.5,lab,ha='center',va='center',fontsize=11)
for a,b in [(69,74),(83,88),(97,102),(111,115)]:
    ax.add_patch(FancyArrowPatch((a,31.5),(b,31.5),arrowstyle='-|>',mutation_scale=11,color='k',lw=1.1))
ax.add_patch(FancyArrowPatch((111,31.5),(115,31.5),arrowstyle='-|>',mutation_scale=11,color='k',lw=1.1))
t(120,31.5,'output',ha='center',va='center',fontsize=11)
s=2.2; gy=26
m1=rand_cells(8,16); m2=rand_cells(8,16)
while set(m2)==set(m1): m2=rand_cells(8,16)
ms=rand_cells(8,16)
for x,m,col,lab in [(67,m1,BLUE,'W1, 75% sparse'),(88,m2,BLUE,'W2, 75% sparse'),(109,ms,LB,'Static control')]:
    grid(x,gy,8,s,{p:col for p in m}); t(x+4*s,gy+1.5,lab,ha='center',fontsize=11)
# lock icon drawn
lx=109+4*s; ly=gy+1.5; lx=109+4*s+8.8
ax.add_patch(Rectangle((lx-1.3,ly-1.2),2.6,2,fc='k',ec='k',zorder=5)); 
from matplotlib.patches import Arc
ax.add_patch(Arc((lx,ly+.8),1.8,2.2,theta1=0,theta2=180,color='k',lw=2,zorder=5))
ax.add_patch(Rectangle((66,0.8),62,5.4,fc='#f4f4f4',ec='k',lw=.8))
t(97,3.5,'Biases stay dense. Pruning and growing rank connections within each weight matrix (local).\nStatic control: same initial random mask as the GSE arm, never rewired (locked).',ha='center',va='center',fontsize=8.6)
fig.savefig('fig2_gse.png',dpi=200,facecolor='white')
# verify counts
print(len(A8),len(m1),len(m2),len(ms),len(act),len(sub),len(grown),len(pruned),set(grown)<=set(sub),set(sub)&set(act),set(pruned)<=set(act))
