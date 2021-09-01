import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import spline, pchip
from matplotlib.legend_handler import HandlerLine2D
from scipy import interpolate
from scipy.interpolate import splrep, splev,interp1d
from matplotlib.pyplot import figure
# Abcisses - nombre de composantes
from scipy.interpolate import make_interp_spline, BSpline
x=np.array([1,2,3,4,6,8,16,32])
x_4=np.array([2,4,8,16,24,32])
x_8=np.array([1,2,3,4,8,16,32])
x_12=np.array([1,2,4,6,16])
x_16=np.array([1,2,3,4,8,16])
x_24=np.array([1,2,4,6])
x_32=np.array([1,2,4])
x_class_unique=np.array([1,1,1])
x_4_fg=[1,2,3,4,6]
x_8_fg=[1,2,3,4,6,8]
x_12_fg=[1,2,3,4,6]
x_16_fg=[1,2,3,4,8,16]
x_24_fg=[1,3]
x_32_fg=[1]

#param 
f4_params = ['3,44K','6,88KK','13,76KK','27,52K','41K','55K']
f8_params = ['6K','12K','18K','24K','48K','96K','193K']
f12_params = ['12,8K','25,6K','51,3K','77K','205K']
f16_params = ['22,2K','44,4K','66,6K','89K','177,8K','355,6K']
f24_params = []
f32_params = ['85,4K','170,8K','341,6K']
f_fsrcnn_params=['[22,3K]']
f4_params_fg = ["0.7K","1.45K","2.15K","2.88K","4.05K"]
f8_params_fg = ["2.4K","4.9K","7.4K","9.8K","14.8K","19.7K"]
f12_params_fg = ["5.2K","10.5K","15.7K","20.9K","31.4K"]
f16_params_fg = ["9K","18K","27K","36K","72.2K","144.4K"]
f24_params_fg = ["19.7K","59K"]
f32_params_fg = ["34,4K"]
# Ordonnées - delta psnr
y_srcnn = np.array([0.5219 for i in range(x.shape[0])])
y_espcn = np.array([0.8179 for i in range(x.shape[0])])
y_fsrcnn = np.array([1.0336 for i in range(x.shape[0])])


y_4 = np.array([0.7787,0.87,0.9636,1.0456,1.0955,1.1681])
y_8 = np.array([0.8023,0.9658,1.0261,1.0698,1.164,1.227,1.3237])
y_12 = np.array([0.9443,1.0624,1.1531,1.2127,1.3264])
y_16=np.array([0.9844,1.1447,1.2051,1.2435,1.2985,1.3522])
y_32=np.array([1.163,1.3202,1.389])
y_class_unique=np.array([1.2707,1.3084,1.3228])

y_4_fg=np.array([0.5388,0.5469,0.6156,0.6428,0.7173])
y_8_fg=np.array([0.7157,0.862,0.9065,0.9278,0.942,0.9562])
y_12_fg=np.array([0.8309,0.9705,0.9975,1.0221,1.0354])
y_16_fg=np.array([0.8558,1.0219,1.0819,1.0799,1.0901,1.085])
y_32_fg=np.array([0.9768])

f_srcnn = make_interp_spline(x, y_srcnn, k=3)  # type: BSpline
f_espcn = make_interp_spline(x, y_espcn, k=3)  # type: BSpline
f_fsrcnn = make_interp_spline(x, y_fsrcnn, k=3)  # type: BSpline

#f_srcnn = interp1d(x, y_srcnn, kind='cubic')
f4 = make_interp_spline(x_4, y_4, k=1)
f8 = make_interp_spline(x_8, y_8, k=1)
f12 = make_interp_spline(x_12, y_12, k=1)
f16 = make_interp_spline(x_16, y_16, k=1)
f32 = make_interp_spline(x_32, y_32, k=1)


f4_fg = make_interp_spline(x_4_fg, y_4_fg, k=1)
f8_fg = make_interp_spline(x_8_fg, y_8_fg, k=1)
f12_fg = make_interp_spline(x_12_fg, y_12_fg, k=1)
f16_fg= make_interp_spline(x_16_fg, y_16_fg, k=1)
f32_fg = make_interp_spline(x_32_fg, y_32_fg, k=1)

xnew = np.linspace(x.min(), x.max(), num=41, endpoint=True)
fig= plt.figure(num=None, figsize=(12, 18), dpi=80, facecolor='w', edgecolor='k')
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()

# SOFA
srcnn_model=ax1.plot(xnew, f_fsrcnn(xnew), '--',color="black",label="Gain fsrcnn paramétrie profonde (22K)")

# Experiments
# Gain Interpolation
data_4 = ax1.plot(x_4, y_4, 'o',color="red",label="Data ( f = 4 )")
model_4 = ax1.plot(xnew, f4(xnew), '-',color="red",label="Model ( f = 4 )")
data_8 = ax1.plot(x_8, y_8, 'o',color="y",label="Data ( f = 8 )")
model_8 = ax1.plot(xnew, f8(xnew), '-',color="y",label="Model ( f = 8 )")
data_12 = ax1.plot(x_12, y_12, 'o',color="b",label="Data ( f = 12 )")
model_12 = ax1.plot(xnew, f12(xnew), '-',color="b",label="Model ( f = 12 )")
data_16 = ax1.plot(x_16, y_16, 'o',color="turquoise",label="Data ( f = 16 )")
model_16 = ax1.plot(xnew, f16(xnew), '-',color="turquoise",label="Model ( f = 16 )")
data_32 = ax1.plot(x_32, y_32, 'o',color="orange",label="Data ( f = 32 )")
model_32 = ax1.plot(xnew, f32(xnew), '-',color="orange",label="Model ( f = 32 )")
data_class_unique = ax1.plot(x_class_unique, y_class_unique, 'o',color="black",label="Data ( f = 48,64,96 / c = 1 )")

data_4_fg = ax2.plot(x_4_fg, y_4_fg, 'v',color="red",label="Filtres prédéfinis Data ( f = 4 )")
model_4_fg = ax2.plot(xnew, f4_fg(xnew), '--',color="red",label="Model  ( f = 4 )")
data_8_fg = ax2.plot(x_8_fg, y_8_fg, 'v',color="y",label="Filtres prédéfinis ( f = 8 )")
model_8_fg = ax2.plot(xnew, f8_fg(xnew), '--',color="y",label="Model  ( f = 8 )")
data_12_fg = ax2.plot(x_12_fg, y_12_fg, 'v',color="b",label="Filtres prédéfinis ( f = 12 )")
model_12_fg = ax2.plot(xnew, f12_fg(xnew), '--',color="b",label="Model  ( f = 12 )")
data_16_fg = ax2.plot(x_16_fg, y_16_fg, 'v',color="turquoise",label="Filtres prédéfinis ( f = 16 )")
model_16_fg = ax2.plot(xnew, f16_fg(xnew), '--',color="turquoise",label="Model  ( f = 16 )")
data_32_fg = ax2.plot(x_32_fg, y_32_fg, 'v',color="orange",label="Filtres prédéfinis ( f = 32 )")
model_32_fg = ax2.plot(xnew, f32_fg(xnew), '--',color="orange",label="Model  ( f = 32 )")

# Nbre params
for i, txt in enumerate(y_4):
    ax1.annotate(f4_params[i],  (x_4[i], y_4[i]),color='red', xytext=(5, 2),
                        textcoords="offset points",
                        ha="right",
                        va="bottom")

for i, txt in enumerate(y_8):
    ax1.annotate(f8_params[i], (x_8[i], y_8[i]), color='y', xytext=(5, 2),
                        textcoords="offset points",
                        ha="right",
                        va="bottom")
    
for i, txt in enumerate(y_12):
    ax1.annotate(f12_params[i], (x_12[i], y_12[i]), color='b', xytext=(5, 2),
                        textcoords="offset points",
                        ha="right",
                        va="bottom")
for i, txt in enumerate(y_16):
    ax1.annotate(f16_params[i], (x_16[i], y_16[i]), color='turquoise', xytext=(5, 2),
                        textcoords="offset points",
                        ha="right",
                        va="bottom")

for i, txt in enumerate(y_32):
    ax1.annotate(f32_params[i], (x_32[i], y_32[i]), color='orange', xytext=(5, 2),
                        textcoords="offset points",
                        ha="right",
                        va="bottom")
for i, txt in enumerate(y_class_unique):
    ax1.annotate(y_class_unique[i], (x_class_unique[i], y_class_unique[i]), color='black', xytext=(5, 2),
                        textcoords="offset points",
                        ha="right",
                        va="bottom")

for i, txt in enumerate(y_4_fg):
    ax1.annotate(f4_params_fg[i],  (x_4_fg[i], y_4_fg[i]),color='red', xytext=(5, 2),
                        textcoords="offset points",
                        ha="right",
                        va="bottom")

for i, txt in enumerate(y_8_fg):
    ax1.annotate(f8_params_fg[i], (x_8_fg[i], y_8_fg[i]), color='y', xytext=(5, 2),
                        textcoords="offset points",
                        ha="right",
                        va="bottom")
    
for i, txt in enumerate(y_12_fg):
    ax1.annotate(f12_params_fg[i], (x_12_fg[i], y_12_fg[i]), color='b', xytext=(5, 2),
                        textcoords="offset points",
                        ha="right",
                        va="bottom")
for i, txt in enumerate(y_16_fg):
    ax1.annotate(f16_params_fg[i], (x_16_fg[i], y_16_fg[i]), color='turquoise', xytext=(5, 2),
                        textcoords="offset points",
                        ha="right",
                        va="bottom")

for i, txt in enumerate(y_32_fg):
    ax1.annotate(f32_params_fg[i], (x_32[i], y_32[i]), color='orange', xytext=(5, 2),
                        textcoords="offset points",
                        ha="right",
                        va="bottom")

    
ax1.annotate(f_fsrcnn_params[0], (x[4], y_fsrcnn[4]), xytext=(5, 2),
                        textcoords="offset points",
                        ha="right",
                        va="bottom")


plt.suptitle('Experience_2_lr', fontsize=12)
plt.xlabel('Nombre de classes', fontsize=12)
ax1.set_ylabel('Delta psnr', fontsize=12)
ax2.set_ylabel('Corrélation', fontsize=12)
ax1.set_ylim(0.4,1.5)
ax2.set_ylim(0.4,1.5)
ax1.legend( loc=2)
ax2.legend( loc=1)
plt.savefig("/export/home/durand192/Documents/filtres_f(c)_exp2_lr.pdf", bbox_inches='tight')
plt.show()


# MFG - EXP 2 - F4 ---------------------------------------------
xnew = np.linspace(x.min(), x.max(), num=41, endpoint=True)
fig= plt.figure(num=None, figsize=(12, 18), dpi=80, facecolor='w', edgecolor='k')
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()

# SOFA
srcnn_model=ax1.plot(xnew, f_fsrcnn(xnew), '--',color="black",label="Gain fsrcnn paramétrie profonde (22K)")

# Experiments
# Gain Interpolation
data = ax1.plot(x_4, y_4, 'o',color="red",label="Data Filtres entraînés ( f = 4 )")
model = ax1.plot(xnew, f4(xnew), '-',color="red",label="Model Filtres entraînés ( f = 4 )")
data_fg = ax2.plot(x_4_fg, y_4_fg, 'v',color="red",label="Filtres prédéfinis Data ( f = 4 )")
#model_fg = ax2.plot(xnew, f4_fg(xnew), '--',color="red",label=" Filtres prédéfinis Model  ( f = 4 )")
# Nbre params
for i, txt in enumerate(y_4):
    ax1.annotate(f4_params[i],  (x_4[i], y_4[i]),color='red', xytext=(5, 2),
                        textcoords="offset points",
                        ha="right",
                        va="bottom")

for i, txt in enumerate(y_4_fg):
    ax1.annotate(f4_params_fg[i],  (x_4_fg[i], y_4_fg[i]),color='red', xytext=(5, 2),
                        textcoords="offset points",
                        ha="right",
                        va="bottom")
    
ax1.annotate(f_fsrcnn_params[0], (x[4], y_fsrcnn[4]), xytext=(5, 2),
                        textcoords="offset points",
                        ha="right",
                        va="bottom")


plt.suptitle('F4', fontsize=12)
plt.xlabel('Nombre de classes', fontsize=12)
ax1.set_ylabel('Delta psnr', fontsize=12)
ax2.set_ylabel('Delta psnr', fontsize=12)
ax1.set_ylim(0.4,1.5)
ax2.set_ylim(0.4,1.5)
ax1.legend( loc=2)
ax2.legend( loc=1)
plt.savefig("/export/home/durand192/Documents/filtres_f(c)_4.pdf", bbox_inches='tight')
plt.show()

# MFG - EXP 2 - F8 ---------------------------------------------
xnew = np.linspace(x.min(), x.max(), num=41, endpoint=True)
fig= plt.figure(num=None, figsize=(12, 18), dpi=80, facecolor='w', edgecolor='k')
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()

# SOFA
srcnn_model=ax1.plot(xnew, f_fsrcnn(xnew), '--',color="black",label="Gain fsrcnn paramétrie profonde (22K)")

# Experiments
# Gain Interpolation
data = ax1.plot(x_8, y_8, 'o',color="blue",label="Data Filtres entraînés ( f = 8 )")
model = ax1.plot(xnew, f8(xnew), '-',color="blue",label="Model Filtres entraînés ( f = 8 )")
data_fg = ax2.plot(x_8_fg, y_8_fg, 'v',color="blue",label="Data Filtres prédéfinis ( f = 8 )")
model_fg = ax2.plot(xnew, f8_fg(xnew), '--',color="blue",label="Model Filtres prédéfinis  ( f = 8 )")
# Nbre params
for i, txt in enumerate(y_8):
    ax1.annotate(f8_params[i],  (x_8[i], y_8[i]),color='blue', xytext=(5, 2),
                        textcoords="offset points",
                        ha="right",
                        va="bottom")

for i, txt in enumerate(y_8_fg):
    ax1.annotate(f8_params_fg[i],  (x_8_fg[i], y_8_fg[i]),color='blue', xytext=(5, 2),
                        textcoords="offset points",
                        ha="right",
                        va="bottom")
    
ax1.annotate(f_fsrcnn_params[0], (x[4], y_fsrcnn[4]), xytext=(5, 2),
                        textcoords="offset points",
                        ha="right",
                        va="bottom")


plt.suptitle('F8', fontsize=12)
plt.xlabel('Nombre de classes', fontsize=12)
ax1.set_ylabel('Delta psnr', fontsize=12)
ax2.set_ylabel('Delta psnr', fontsize=12)
ax1.set_ylim(0.4,1.5)
ax2.set_ylim(0.4,1.5)
ax1.legend( loc=2)
ax2.legend( loc=1)
plt.savefig("/export/home/durand192/Documents/filtres_f(c)_8.pdf", bbox_inches='tight')
plt.show()


# MFG - EXP 2 - F 12 ---------------------------------------------
xnew = np.linspace(x.min(), x.max(), num=41, endpoint=True)
fig= plt.figure(num=None, figsize=(12, 18), dpi=80, facecolor='w', edgecolor='k')
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()

# SOFA
srcnn_model=ax1.plot(xnew, f_fsrcnn(xnew), '--',color="black",label="Gain fsrcnn paramétrie profonde (22K)")

# Experiments
# Gain Interpolation
data = ax1.plot(x_12, y_12, 'o',color="green",label="Data Filtres entraînés ( f = 12 )")
model = ax1.plot(xnew, f12(xnew), '-',color="green",label="Model Filtres entraînés ( f = 12 )")
data_fg = ax2.plot(x_12_fg, y_12_fg, 'v',color="green",label="Data Filtres prédéfinis ( f = 12 )")
model_fg = ax2.plot(xnew, f12_fg(xnew), '--',color="green",label="Model Filtres prédéfinis ( f = 12 )")
# Nbre params
for i, txt in enumerate(y_12):
    ax1.annotate(f12_params[i],  (x_12[i], y_12[i]),color='green', xytext=(5, 2),
                        textcoords="offset points",
                        ha="right",
                        va="bottom")

for i, txt in enumerate(y_12_fg):
    ax1.annotate(f12_params_fg[i],  (x_12_fg[i], y_12_fg[i]),color='green', xytext=(5, 2),
                        textcoords="offset points",
                        ha="right",
                        va="bottom")
    
ax1.annotate(f_fsrcnn_params[0], (x[6], y_fsrcnn[4]), xytext=(5, 2),
                        textcoords="offset points",
                        ha="right",
                        va="bottom")


plt.suptitle('F12', fontsize=12)
plt.xlabel('Nombre de classes', fontsize=12)
ax1.set_ylabel('Delta psnr', fontsize=12)
ax2.set_ylabel('Delta psnr', fontsize=12)
ax1.set_ylim(0.4,1.5)
ax2.set_ylim(0.4,1.5)
ax1.legend( loc=2)
ax2.legend( loc=1)
plt.savefig("/export/home/durand192/Documents/filtres_f(c)_12.pdf", bbox_inches='tight')
plt.show()

# MFG - EXP 2 - F 16 ---------------------------------------------
xnew = np.linspace(x.min(), x.max(), num=41, endpoint=True)
fig= plt.figure(num=None, figsize=(12, 18), dpi=80, facecolor='w', edgecolor='k')
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()

# SOFA
srcnn_model=ax1.plot(xnew, f_fsrcnn(xnew), '--',color="black",label="Gain fsrcnn paramétrie profonde (22K)")

# Experiments
# Gain Interpolation
data = ax1.plot(x_16, y_16, 'o',color="brown",label="Data Filtres entraînés ( f = 12 )")
model = ax1.plot(xnew, f16(xnew), '-',color="brown",label="Model Filtres entraînés ( f = 12 )")
data_fg = ax2.plot(x_16_fg, y_16_fg, 'v',color="brown",label="Data Filtres prédéfinis ( f = 12 )")
model_fg = ax2.plot(xnew, f16_fg(xnew), '--',color="brown",label="Model Filtres prédéfinis ( f = 12 )")
# Nbre params
for i, txt in enumerate(y_16):
    ax1.annotate(f16_params[i],  (x_16[i], y_16[i]),color='brown', xytext=(5, 2),
                        textcoords="offset points",
                        ha="right",
                        va="bottom")

for i, txt in enumerate(y_16_fg):
    ax1.annotate(f16_params_fg[i],  (x_16_fg[i], y_16_fg[i]),color='brown', xytext=(5, 2),
                        textcoords="offset points",
                        ha="right",
                        va="bottom")
    
ax1.annotate(f_fsrcnn_params[0], (x[6], y_fsrcnn[4]), xytext=(5, 2),
                        textcoords="offset points",
                        ha="right",
                        va="bottom")


plt.suptitle('F16', fontsize=12)
plt.xlabel('Nombre de classes', fontsize=12)
ax1.set_ylabel('Delta psnr', fontsize=12)
ax2.set_ylabel('Delta psnr', fontsize=12)
ax1.set_ylim(0.4,1.5)
ax2.set_ylim(0.4,1.5)
ax1.legend( loc=2)
ax2.legend( loc=1)
plt.savefig("/export/home/durand192/Documents/filtres_f(c)_16.pdf", bbox_inches='tight')
plt.show()