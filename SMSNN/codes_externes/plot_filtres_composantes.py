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
x=np.array([1,2,3,4,8,16,32])
x_4=np.array([2,4,8,16,24,32])
x_8=np.array([1,2,3,4,8,16,32])
x_12=np.array([1,2,4,6,16])
x_16=np.array([1,2,3,4])
x_24=np.array([1,2,4,6])
x_32=np.array([1,2,4])
x_class_unique=np.array([1,1,1])

x_4_cor=[2,4,8,16,24,32]
x_8_cor=[2,3,4,8,16,32]
x_12_cor=[2,4,6,16]
x_16_cor=[2,3,4]
x_24_cor=[2,4,6]
x_32_cor=[2,4]

#param 
f4_params = ['3,44K','6,88KK','13,76KK','27,52K','41K','55K']
f8_params = ['6K','12K','18K','24K','48K','96K','193K']
f12_params = ['12,8K','25,6K','51,3K','77K','205K']
f16_params = ['22,2K','44,4K','66,6K','89K']
f24_params = ['48,7K','97,4K','194,8K','292,2K']
f32_params = ['85,4K','170,8K','341,6K']
f_fsrcnn_params=['[22,3K]']
# Ordonnées - delta psnr
y_srcnn = np.array([0.5219 for i in range(x.shape[0])])
y_espcn = np.array([0.8179 for i in range(x.shape[0])])
y_fsrcnn = np.array([1.0336 for i in range(x.shape[0])])

y_12 = np.array([0.9443,1.0624,1.1531,1.2127,1.3264])
y_4 = np.array([0.7787,0.87,0.9636,1.0456,1.0955,1.1681])
y_8 = np.array([0.8023,0.9658,1.0261,1.0698,1.164,1.227,1.3237])
y_16=np.array([0.9844,1.1447,1.2051,1.2435])
y_24=np.array([1.1333,1.226,1.3257,1.3431])
y_32=np.array([1.163,1.3202,1.389])
y_class_unique=np.array([1.2707,1.3084,1.3228])

y_4_cor=np.array([0.6815,0.1233,0.1537,0.092,0.0812,0.0795])
y_8_cor=np.array([0.3228,0.1752,0.1623,0.1503,0.0979,0.0771])
y_12_cor=np.array([0.3168,0.2422,0.2239,0.1231])
y_16_cor=np.array([0.473,0.3826,0.2996])
y_24_cor=np.array([0.3494,0.329,0.3094])
y_32_cor=np.array([0.5347,0.3827])

f_srcnn = make_interp_spline(x, y_srcnn, k=3)  # type: BSpline
f_espcn = make_interp_spline(x, y_espcn, k=3)  # type: BSpline
f_fsrcnn = make_interp_spline(x, y_fsrcnn, k=3)  # type: BSpline

#f_srcnn = interp1d(x, y_srcnn, kind='cubic')
f4 = make_interp_spline(x_4, y_4, k=1)
f8 = make_interp_spline(x_8, y_8, k=1)
f12 = make_interp_spline(x_12, y_12, k=1)
f16 = make_interp_spline(x_16, y_16, k=1)
f24 = make_interp_spline(x_24, y_24, k=1)
f32 = make_interp_spline(x_32, y_32, k=1)


f4_cor = make_interp_spline(x_4_cor, y_4_cor, k=1)
f8_cor = make_interp_spline(x_8_cor, y_8_cor, k=1)
f12_cor = make_interp_spline(x_12_cor, y_12_cor, k=1)
f16_cor = make_interp_spline(x_16_cor, y_16_cor, k=1)
f24_cor = make_interp_spline(x_24_cor, y_24_cor, k=1)
f32_cor = make_interp_spline(x_32_cor, y_32_cor, k=1)

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
data_24 = ax1.plot(x_24, y_24, 'o',color="g",label="Data ( f = 24 )")
model_24 = ax1.plot(xnew, f24(xnew), '-',color="g",label="Model ( f = 24 )")
data_32 = ax1.plot(x_32, y_32, 'o',color="orange",label="Data ( f = 32 )")
model_32 = ax1.plot(xnew, f32(xnew), '-',color="orange",label="Model ( f = 32 )")
data_class_unique = ax1.plot(x_class_unique, y_class_unique, 'o',color="black",label="Data ( f = 48,64,96 / c = 1 )")

data_4_cor = ax2.plot(x_4_cor, y_4_cor, 'v',color="red",label="Correlation Data ( f = 4 )")
model_4_cor = ax2.plot(xnew, f4_cor(xnew), '--',color="red",label="Model Correlation ( f = 4 )")
data_8_cor = ax2.plot(x_8_cor, y_8_cor, 'v',color="y",label="Correlation Data ( f = 8 )")
model_8_cor = ax2.plot(xnew, f8_cor(xnew), '--',color="y",label="Model Correlation ( f = 8 )")
data_12_cor = ax2.plot(x_12_cor, y_12_cor, 'v',color="b",label="Correlation Data ( f = 12 )")
model_12_cor = ax2.plot(xnew, f12_cor(xnew), '--',color="b",label="Model Correlation ( f = 12 )")
data_16_cor = ax2.plot(x_16_cor, y_16_cor, 'v',color="turquoise",label="Correlation Data ( f = 16 )")
model_16_cor = ax2.plot(xnew, f16_cor(xnew), '--',color="turquoise",label="Model Correlation ( f = 16 )")
data_24_cor = ax2.plot(x_24_cor, y_24_cor, 'v',color="g",label="Correlation Data ( f = 24 )")
model_24_cor = ax2.plot(xnew, f24_cor(xnew), '--',color="g",label="Model Correlation ( f = 24 )")
data_32_cor = ax2.plot(x_32_cor, y_32_cor, 'v',color="orange",label="Correlation Data ( f = 32 )")
model_32_cor = ax2.plot(xnew, f32_cor(xnew), '--',color="orange",label="Model Correlation ( f = 32 )")

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
for i, txt in enumerate(y_24):
    ax1.annotate(f24_params[i], (x_24[i], y_24[i]), color='g', xytext=(5, 2),
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

ax1.annotate(f_fsrcnn_params[0], (x[4], y_fsrcnn[4]), xytext=(5, 2),
                        textcoords="offset points",
                        ha="right",
                        va="bottom")


plt.suptitle('Experience_2_lr', fontsize=12)
plt.xlabel('Nombre de classes', fontsize=12)
ax1.set_ylabel('Delta psnr', fontsize=12)
ax2.set_ylabel('Corrélation', fontsize=12)
ax1.set_ylim(0,2)
ax2.set_ylim(0,3.2)
ax1.legend( loc=2)
ax2.legend( loc=1)
plt.savefig("/export/home/durand192/Documents/filtres_f(c)_exp2_lr.pdf", bbox_inches='tight')
plt.show()
# FDP FDP FDP FDP DE MATPLOTLIB

