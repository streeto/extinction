from pycasso import fitsQ3DataCube
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages
from pystarlight.util.velocity_fix import SpectraVelocityFixer
from pystarlight.util.redenninglaws import Cardelli_RedLaw, Calzetti_RedLaw
from pycasso.util import get_polygon_mask
import argparse

parser = argparse.ArgumentParser(description='Compute reddening laws.')
parser.add_argument('db', type=str, nargs=1,
                   help='CALIFA superFITS.')
parser.add_argument('--fixk', dest='fixKinematics', action='store_true',
                    help='Fix kinematics.')
parser.add_argument('--vd', dest='targetVd', default=0.0, type=float,
                    help='Velocity dispersion to be convolved with the spectra (if --fixk).')
parser.add_argument('--lll', dest='lambdaLowerLimit', default=0.0, type=float,
                    help='Lower limit in lambda when fitting.')
parser.add_argument('--outdir', dest='outdir', default='.', type=str,
                    help='Output directory for figures.')

args = parser.parse_args()

print 'Opening file %s' % args.db[0]
K = fitsQ3DataCube(args.db[0])
calId = K.califaID
pipeVer = K.header['pipe vers']
sn_vor = K.header['sn_vor']

# TODO: Parameterize the dust regions.
dust_lane_poly = {
                  'K0708': [(21.322580645161288, 37.41935483870968), (25.758064516129032, 41.145161290322584), (28.951612903225808, 39.548387096774192), (32.322580645161288, 39.370967741935488), (34.451612903225808, 40.08064516129032), (37.29032258064516, 39.016129032258064), (39.951612903225808, 38.12903225806452), (42.79032258064516, 38.12903225806452), (44.91935483870968, 36.354838709677416), (48.467741935483872, 33.161290322580648), (49.70967741935484, 31.91935483870968), (50.064516129032256, 30.5), (54.5, 28.725806451612904), (55.91935483870968, 25.887096774193552), (49.70967741935484, 27.661290322580648), (45.62903225806452, 28.016129032258064), (43.677419354838712, 27.661290322580648), (40.661290322580648, 28.193548387096776), (38.532258064516128, 27.838709677419352), (37.645161290322584, 29.258064516129032), (36.225806451612904, 30.322580645161288), (35.693548387096776, 31.387096774193552), (35.693548387096776, 32.983870967741936), (33.564516129032256, 33.161290322580648), (33.20967741935484, 31.741935483870968), (30.903225806451616, 32.806451612903224), (28.241935483870968, 34.935483870967744), (26.112903225806448, 36.354838709677416), (24.693548387096776, 37.41935483870968), (23.451612903225808, 36.354838709677416)],
                  'K0850': [(30.025345622119819, 59.555299539170505), (29.361751152073733, 56.735023041474648), (31.186635944700459, 54.744239631336399), (30.191244239631335, 52.255760368663601), (32.016129032258064, 50.596774193548384), (34.006912442396313, 50.099078341013822), (33.011520737327189, 47.942396313364057), (32.679723502304142, 44.790322580645167), (30.523041474654381, 43.29723502304148), (27.702764976958523, 39.481566820276498), (29.029953917050694, 37.490783410138249), (32.845622119815673, 33.675115207373281), (33.841013824884797, 31.518433179723505), (33.343317972350235, 28.698156682027655), (32.34792626728111, 26.541474654377883), (26.707373271889399, 27.039170506912445), (27.370967741935484, 22.559907834101384), (23.555299539170509, 19.076036866359448), (24.716589861751149, 16.91935483870968), (27.205069124423961, 15.260368663594473), (25.711981566820274, 12.605990783410142), (28.698156682027648, 9.9516129032258078), (26.873271889400922, 6.4677419354838719), (28.864055299539171, 2.8179723502304164), (32.016129032258064, 4.4769585253456246), (35.997695852534562, 5.4723502304147473), (39.315668202764982, 3.6474654377880213), (38.486175115207374, 7.2972350230414769), (41.804147465437794, 10.781105990783413), (41.97004608294931, 14.099078341013829), (39.813364055299544, 15.592165898617512), (38.983870967741936, 18.246543778801847), (40.974654377880185, 21.56451612903226), (41.472350230414747, 26.873271889400922), (42.633640552995388, 32.182027649769594), (44.458525345622121, 36.329493087557609), (43.794930875576043, 40.974654377880185), (43.794930875576043, 45.453917050691246), (43.29723502304148, 47.776497695852541), (46.781105990783416, 47.776497695852541), (46.781105990783416, 51.758064516129039), (42.633640552995388, 53.582949308755758), (42.799539170506918, 55.739631336405523), (43.29723502304148, 58.891705069124427), (44.956221198156683, 60.882488479262676), (43.960829493087559, 63.70276497695852), (44.956221198156683, 67.684331797235018), (42.799539170506918, 69.675115207373267), (39.149769585253452, 68.18202764976958), (36.661290322580641, 66.191244239631331), (34.172811059907829, 69.011520737327189), (30.357142857142858, 68.18202764976958), (32.845622119815673, 63.370967741935488), (31.020737327188943, 61.546082949308754)],
                  'K0925': [(15.675115207373272, 18.926267281105993), (16.394009216589865, 26.294930875576036), (19.808755760368669, 32.764976958525345), (23.762672811059907, 39.953917050691246), (25.559907834101388, 43.1889400921659), (25.91935483870968, 47.142857142857146), (29.334101382488484, 48.940092165898619), (33.647465437788021, 50.377880184331801), (37.781105990783409, 51.456221198156683), (46.048387096774192, 55.769585253456228), (48.744239631336406, 54.152073732718904), (48.384792626728114, 51.815668202764982), (46.228110599078342, 51.096774193548399), (46.228110599078342, 48.041474654377886), (42.813364055299544, 47.142857142857146), (43.352534562211986, 45.705069124423964), (45.509216589861751, 44.447004608294932), (43.891705069124427, 43.009216589861758), (41.735023041474655, 43.009216589861758), (36.882488479262676, 39.055299539170512), (32.389400921658989, 34.562211981566826), (29.693548387096776, 29.70967741935484), (26.099078341013829, 25.216589861751153), (24.481566820276498, 22.52073732718894), (24.301843317972356, 18.387096774193552), (22.324884792626733, 16.230414746543779), (20.527649769585253, 17.668202764976957), (17.831797235023039, 18.027649769585253), (16.394009216589865, 17.129032258064516)],
                  }
dust_region_poly = {
                    'K0708': [(32.854838709677416, 37.596774193548384), (34.274193548387096, 38.12903225806452), (36.048387096774192, 38.12903225806452), (37.29032258064516, 36.887096774193552), (40.306451612903224, 37.064516129032256), (42.967741935483872, 36.0), (45.451612903225808, 32.983870967741936), (41.016129032258064, 32.451612903225808), (40.12903225806452, 33.693548387096776), (35.516129032258064, 34.225806451612904), (33.032258064516128, 34.758064516129032), (31.967741935483872, 36.177419354838712)],
                    'K0850': [(37.822580645161295, 47.278801843317979), (36.163594470046078, 45.28801843317973), (36.661290322580641, 42.799539170506918), (35.168202764976954, 42.301843317972356), (35.5, 40.476958525345623), (35.168202764976954, 37.656682027649779), (35.334101382488484, 35.168202764976968), (35.665898617511516, 32.016129032258064), (34.836405529953922, 29.195852534562217), (33.841013824884797, 27.205069124423968), (33.177419354838705, 22.725806451612907), (32.845622119815673, 20.735023041474658), (33.509216589861751, 18.910138248847929), (32.016129032258064, 17.085253456221199), (31.352534562211982, 14.928571428571431), (32.34792626728111, 13.269585253456224), (34.338709677419359, 12.440092165898619), (35.831797235023046, 15.426267281105993), (35.997695852534562, 18.412442396313367), (37.490783410138249, 20.403225806451616), (37.490783410138249, 22.394009216589861), (38.65207373271889, 24.882488479262673), (39.647465437788014, 27.70276497695853), (40.311059907834107, 32.182027649769594), (40.476958525345623, 36.329493087557609), (39.813364055299544, 37.822580645161295), (40.476958525345623, 39.813364055299544), (40.808755760368669, 46.615207373271886), (39.97926267281106, 47.444700460829495)],
                    'K0925':[(20.168202764976961, 24.857142857142861), (22.504608294930875, 25.216589861751153), (22.504608294930875, 27.373271889400925), (20.34792626728111, 29.170506912442399), (22.864055299539174, 32.046082949308754), (22.864055299539174, 34.023041474654377), (24.481566820276498, 33.304147465437794), (26.458525345622121, 34.202764976958527), (26.458525345622121, 36.359447004608299), (26.458525345622121, 37.79723502304148), (27.177419354838712, 39.414746543778804), (29.154377880184335, 39.235023041474655), (30.412442396313367, 39.774193548387103), (30.412442396313367, 42.110599078341018), (32.20967741935484, 42.47004608294931), (32.748847926267281, 44.447004608294932), (34.546082949308754, 44.447004608294932), (34.905529953917053, 42.110599078341018), (33.647465437788021, 41.211981566820278), (34.905529953917053, 39.594470046082954), (33.827188940092171, 38.156682027649772), (31.490783410138249, 38.156682027649772), (31.311059907834107, 36.359447004608299), (30.951612903225808, 33.843317972350235), (29.873271889400925, 32.046082949308754), (28.435483870967744, 29.70967741935484), (26.997695852534562, 28.271889400921658), (25.91935483870968, 26.474654377880185), (25.02073732718894, 25.036866359447004), (23.223502304147466, 23.778801843317972), (21.605990783410142, 21.981566820276498), (20.34792626728111, 22.52073732718894)],
                    }

# Plot customization
ylim_A_lambda = {'K0925': (0.5, 1.6),
                 'K0708': (0, 0.4),
                 'K0850': (0.25, 0.85),
                 }
ylim_A_V = {'K0925': (0, 2.1),
            'K0708': (-0.2, 0.7),
            'K0850': (-0.1, 1.0),
            }
ylim_at = {'K0925': (9.25, 10.25),
           'K0708': (9.25, 10.25),
           'K0850': (9.25, 10.25),
           }
ylim_Dn4000 = {'K0925': (1.0, 2.0),
               'K0708': (1.0, 2.0),
               'K0850': (1.0, 2.0),
           }

vmin_A_V = {'K0925': 0.0,
            'K0708': 0.0,
            'K0850': 0.0,
            }
vmax_A_V = {'K0925': 2.0,
            'K0708': 1.0,
            'K0850': 1.0,
            }

vmin_color = {'K0925': 1.0,
              'K0708': 1.0,
              'K0850': 0.5,
              }
vmax_color = {'K0925': 2.0,
              'K0708': 1.5,
              'K0850': 1.1,
              }

vmin_at = {'K0925': 9.25,
           'K0708': 9.25,
           'K0850': 9.25,
           }
vmax_at = {'K0925': 10.25,
           'K0708': 10.25,
           'K0850': 10.25,
           }

vmin_v0 = {'K0925': -250,
           'K0708': -250,
           'K0850': -250,
           }
vmax_v0 = {'K0925': 250,
           'K0708': 250,
           'K0850': 250,
           }

vmin_vd = {'K0925': 0,
           'K0708': 0,
           'K0850': 0,
           }
vmax_vd = {'K0925': 300,
           'K0708': 300,
           'K0850': 300,
           }

# Compute bit masks.
dust_lane__yx = get_polygon_mask(dust_lane_poly[calId], K.qSignal.shape) & (K.pixelDistance__yx > 3) & K.qMask
dust_region__yx = get_polygon_mask(dust_region_poly[calId], K.qSignal.shape) & (K.pixelDistance__yx > 3) & K.qMask

K.color__yx = K.fluxRatio__yx(6100, 6500, 4000, 4500)

print 'Finding geometry...'
for i in xrange(10):
    img = np.ma.masked_array(K.qSignal, mask=~K.qMask)
    img[dust_lane__yx] = np.ma.masked
    img, mask = K.fillImage(img)
    pa, ba = K.getEllipseParams(img, mask)
    K.setGeometry(pa, ba)
pdf = PdfPages('%s/%s_%s_v%02d.pdf' % (args.outdir, calId, pipeVer, sn_vor))

plt.ioff()
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8 
plt.rcParams['legend.fontsize'] = 6
plt.rcParams['legend.labelspacing'] = 0.3 
plt.rcParams['legend.frameon'] = False 
lw = 0.7

# Measured from latex
fig_ptwidth = 345.0
fig_width = fig_ptwidth / 72
fig_height = 0.9 * fig_width

# Plot some diagnostic maps.
f = plt.figure(1, figsize=(fig_width, fig_height))
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
plt.suptitle('Maps - %s (%s) - pipeline %s - Voronoi %d' % (K.galaxyName, calId, pipeVer, sn_vor), fontsize=10)

mask_image = np.zeros_like(K.qSignal)
mask_image[dust_lane__yx] = 2.0
mask_image[dust_region__yx] = 3.0

ax_im0 = plt.subplot(gs[0, 0])
im = ax_im0.imshow(np.log10(K.qSignal), cmap='Blues')
ax_im0.contour(mask_image, levels=[1.9, 2.9], colors=['gray', 'k'], linewidths=1.5)
ax_im0.set_xticks([])
ax_im0.set_yticks([])
plt.colorbar(im, ax=ax_im0)
ax_im0.set_title(r'$\log F_{5635\AA}$')

ax_im1 = plt.subplot(gs[0, 1])
im = ax_im1.imshow(K.color__yx, cmap='RdBu_r', vmin=vmin_color[calId], vmax=vmax_color[calId])
ax_im1.contour(mask_image, levels=[1.9, 2.9], colors=['gray', 'k'], linewidths=1.5)
ax_im1.set_xticks([])
ax_im1.set_yticks([])
plt.colorbar(im, ax=ax_im1)
ax_im1.set_title(r'"Color" ($F_{6300\AA} / F_{4250\AA})$')

ax_im2 = plt.subplot(gs[1, 0])
im = ax_im2.imshow(K.A_V__yx, cmap='Reds', vmin=vmin_A_V[calId], vmax=vmax_A_V[calId])
ax_im2.contour(mask_image, levels=[1.9, 2.9], colors=['gray', 'k'], linewidths=1.5)
ax_im2.set_xticks([])
ax_im2.set_yticks([])
plt.colorbar(im, ax=ax_im2)
ax_im2.set_title('$A_V$')

ax_im3 = plt.subplot(gs[1, 1])
im = ax_im3.imshow(K.at_flux__yx, cmap='Reds', vmin=vmin_at[calId], vmax=vmax_at[calId])
ax_im3.contour(mask_image, levels=[1.9, 2.9], colors=['gray', 'k'], linewidths=1.5)
ax_im3.set_xticks([])
ax_im3.set_yticks([])
plt.colorbar(im, ax=ax_im3)
ax_im3.set_title(r'$\langle \log t \rangle_L$')

gs.tight_layout(f, rect=[0, 0, 1, 0.97])
pdf.savefig(f)
plt.savefig('%s/maps_%s_%s_v%02d.pdf' % (args.outdir, calId, pipeVer, sn_vor))  


mean_starlight_A_V = K.A_V__yx[dust_region__yx].mean()
print 'Mean starlight A_V:', mean_starlight_A_V

# Fix kinematics
if args.fixKinematics:
    print 'Fixing kinematics...'
    vfix = SpectraVelocityFixer(K.l_obs, v_0=K.v_0, v_d=K.v_d, nproc=2)
    f_obs__lyx = K.zoneToYX(vfix(K.f_obs, target_vd=args.targetVd), extensive=True, surface_density=False)
    f_flag__lyx = K.zoneToYX(vfix(K.f_flag, target_vd=args.targetVd) > 0.0, extensive=False)
else:
    f_obs__lyx = K.f_obs__lyx.copy()
    f_flag__lyx = K.f_flag__lyx > 0.0

npix = dust_region__yx.sum()
bad_lambda = f_flag__lyx[..., dust_region__yx].sum(axis=1) > (npix / 2)

# Flag the flagged
f_obs__lyx[f_flag__lyx] = np.ma.masked

# Repeat the trick to fill the spectra at the dusty pixels.
print 'Computing intrinsic spectra...'
f_clean__lyx = f_obs__lyx.copy()
f_clean__lyx[:, dust_lane__yx] = np.ma.masked
for i in xrange(K.Nl_obs):
    if bad_lambda[i]: continue
    img = f_clean__lyx[i].copy()
    img[dust_lane__yx] = np.ma.masked
    img, _ = K.fillImage(img)
    f_clean__lyx[i, dust_lane__yx] = img[dust_lane__yx]

# Extinction in each pixel.
tau__lz = np.log(f_clean__lyx[..., dust_lane__yx] / f_obs__lyx[..., dust_lane__yx])
almost_one = np.log10(np.exp(1.0)) * 2.5
A__lz = tau__lz * almost_one

# Integrated dust lane spectra.
dusty_int_spec = np.median(f_obs__lyx[..., dust_region__yx], axis=1)
clean_int_spec = np.median(f_clean__lyx[..., dust_region__yx], axis=1)
dusty_int_spec[bad_lambda] = np.ma.masked
clean_int_spec[bad_lambda] = np.ma.masked

# Plot the spectra and extinction.
fig_height = 0.9 * fig_width
f = plt.figure(2, figsize=(fig_width, fig_height))
gs = gridspec.GridSpec(2, 1, width_ratios=[1, ], height_ratios=[1, 1])
plt.suptitle('Extinction - %s (%s) - pipeline %s - Voronoi %d' % (K.galaxyName, calId, pipeVer, sn_vor), fontsize=10)

ax_spec = plt.subplot(gs[0, 0])
ax_spec.plot(K.l_obs, dusty_int_spec, 'r-', lw=lw, label='dust lane')
ax_spec.plot(K.l_obs, clean_int_spec, 'b-', lw=lw, label='intrinsic')
ax_spec.set_xticklabels([])
ax_spec.set_ylabel(r'Flux')
ax_spec.set_xlim(K.l_obs.min(), K.l_obs.max())
ax_spec.set_ylim(0.0, clean_int_spec.max())
ax_spec.legend(loc='lower right')

# tau_lambda of the integrated spectra in the dusty pixels.
tau__l = np.log(clean_int_spec / dusty_int_spec)
A__l = tau__l * almost_one

# Fit reddening laws
from astropy.modeling.models import custom_model_1d 

@custom_model_1d
def ccm_reddening(l, R_V=3.1, A_V=1.0):
    return A_V * Cardelli_RedLaw(l, R_V)

@custom_model_1d
def cal_reddening(l, R_V=4.05, A_V=1.0):
    return A_V * Calzetti_RedLaw(l, R_V)

from astropy.modeling.fitting import NonLinearLSQFitter
fit = NonLinearLSQFitter()

fitmask = ~bad_lambda & (K.l_obs > args.lambdaLowerLimit)

# CCM law
ccm_A__l = fit(ccm_reddening(R_V=3.1, A_V=mean_starlight_A_V), K.l_obs[fitmask], A__l[fitmask])
print ccm_A__l

# CCM law with R_V = 3.1
_model = ccm_reddening(R_V=3.1, A_V=mean_starlight_A_V)
_model.R_V.fixed = True
RV31ccm_A__l = fit(_model, K.l_obs[fitmask], A__l[fitmask])
print RV31ccm_A__l

# Calzetti law
cal_A__l = fit(cal_reddening(R_V=4.05, A_V=mean_starlight_A_V), K.l_obs[fitmask], A__l[fitmask])
print cal_A__l

# Calzetti law with R_V = 4.05
_model = cal_reddening(R_V=4.05, A_V=mean_starlight_A_V)
_model.R_V.fixed = True
RV405cal_A__l = fit(_model, K.l_obs[fitmask], A__l[fitmask])
print RV405cal_A__l

# Plot the extinction curves.
ax_ext = plt.subplot(gs[1, 0])
ax_ext.plot(K.l_obs, A__l, 'k-', markeredgecolor='none',
            label=r'Observed')
ax_ext.plot(K.l_obs, ccm_A__l(K.l_obs), 'b--', lw=lw, markeredgecolor='none',
            label=r'CCM fit, $R_V = %.2f$, $A_V = %.2f$)' % (ccm_A__l.R_V.value, ccm_A__l.A_V.value))
ax_ext.plot(K.l_obs, RV31ccm_A__l(K.l_obs), 'b:', lw=lw, markeredgecolor='none',
            label=r'CCM fit, $R_V = %.2f$, $A_V = %.2f$)' % (RV31ccm_A__l.R_V.value, RV31ccm_A__l.A_V.value))
ax_ext.plot(K.l_obs, cal_A__l(K.l_obs), 'r--', lw=lw, markeredgecolor='none',
            label=r'CAL fit, $R_V = %.2f$, $A_V = %.2f$)' % (cal_A__l.R_V.value, cal_A__l.A_V.value))
ax_ext.plot(K.l_obs, RV405cal_A__l(K.l_obs), 'r:', lw=lw, markeredgecolor='none',
            label=r'CAL fit, $R_V = %.2f$, $A_V = %.2f$)' % (RV405cal_A__l.R_V.value, RV405cal_A__l.A_V.value))

ax_ext.set_xlim(K.l_obs.min(), K.l_obs.max())
ax_ext.set_ylim(ylim_A_lambda[calId][0], ylim_A_lambda[calId][1])
ax_ext.set_xlabel(r'wavelength $[\AA]$')
ax_ext.set_ylabel(r'$A_\lambda$')
ax_ext.legend(loc='lower left')

gs.tight_layout(f, rect=[0, -0.04, 1, 0.99])  
pdf.savefig(f)
plt.savefig('%s/spectra_%s_%s_v%02d.pdf' % (args.outdir, calId, pipeVer, sn_vor))  


# Plot radial profiles
distance__yx = np.ma.masked_where(dust_lane__yx | ~K.qMask, K.pixelDistance__yx, copy=True)
A_V__yx = np.ma.masked_where(dust_lane__yx | ~K.qMask, K.A_V__yx, copy=True)
at_flux__yx = np.ma.masked_where(dust_lane__yx | ~K.qMask, K.at_flux__yx, copy=True)
LobnSD__yx = np.ma.masked_where(dust_lane__yx | ~K.qMask, K.LobnSD__yx, copy=True)
Dn4000__yx = np.ma.masked_where(dust_lane__yx | ~K.qMask, K.Dn4000__yx, copy=True)
color__yx = np.ma.masked_where(dust_lane__yx | ~K.qMask, K.color__yx, copy=True)
bins = np.arange(31)
bins_center = bins[:-1] + 0.5

lane_distance__yx = np.ma.masked_where(~dust_lane__yx, K.pixelDistance__yx, copy=True)
lane_A_V__yx = np.ma.masked_where(~dust_lane__yx, K.A_V__yx, copy=True)
lane_at_flux__yx = np.ma.masked_where(~dust_lane__yx, K.at_flux__yx, copy=True)
lane_LobnSD__yx = np.ma.masked_where(~dust_lane__yx, K.LobnSD__yx, copy=True)
lane_Dn4000__yx = np.ma.masked_where(~dust_lane__yx, K.Dn4000__yx, copy=True)
lane_color__yx = np.ma.masked_where(~dust_lane__yx, K.color__yx, copy=True)
bins_lane = np.arange(lane_distance__yx.min(), lane_distance__yx.max() + 1)
bins_lane_center = bins_lane[:-1] + 0.5

n_lane_pix = (A__lz.shape[1])
fit_A_V__z = np.empty((n_lane_pix,))
fit_A_V__yx = np.ma.masked_all_like(K.A_V__yx)
for zz in xrange(n_lane_pix):
#     _model = cal_reddening(R_V=cal_A__l.R_V.value, A_V=cal_A__l.A_V.value)
    _model = ccm_reddening(R_V=3.1, A_V=ccm_A__l.A_V.value)
    _model.R_V.fixed = True
    _ccm_A__l = fit(_model, K.l_obs[fitmask], A__lz[fitmask, zz])
    if fit.fit_info['ierr'] in [1, 2, 3, 4]:
        fit_A_V__z[zz] = _ccm_A__l.A_V.value
fit_A_V__yx[dust_lane__yx] = fit_A_V__z

pdf.savefig(f)
rp_mode = 'mean'
lane_A_V__r = K.radialProfile(lane_A_V__yx, bin_r=bins_lane, r__yx=lane_distance__yx, rad_scale=1, mode=rp_mode, mask=dust_region__yx)
fit_A_V__r = K.radialProfile(fit_A_V__yx, bin_r=bins_lane, r__yx=lane_distance__yx, rad_scale=1, mode=rp_mode, mask=dust_region__yx)
lane_at_flux__r = K.radialProfile(lane_at_flux__yx * lane_LobnSD__yx, bin_r=bins_lane, r__yx=lane_distance__yx, rad_scale=1, mode='sum', mask=dust_region__yx) \
    / K.radialProfile(lane_LobnSD__yx, bin_r=bins_lane, r__yx=lane_distance__yx, rad_scale=1, mode='sum', mask=dust_region__yx)
lane_Dn4000__r = K.radialProfile(lane_Dn4000__yx, bin_r=bins_lane, r__yx=lane_distance__yx, rad_scale=1, mode=rp_mode, mask=dust_region__yx)
lane_color__r = K.radialProfile(lane_color__yx, bin_r=bins_lane, r__yx=lane_distance__yx, rad_scale=1, mode=rp_mode, mask=dust_region__yx)

A_V__r = K.radialProfile(A_V__yx, bin_r=bins, r__yx=distance__yx, rad_scale=1, mode=rp_mode)
at_flux__r = K.radialProfile(at_flux__yx * LobnSD__yx, bin_r=bins, r__yx=distance__yx, rad_scale=1, mode='sum') \
    / K.radialProfile(LobnSD__yx, bin_r=bins, r__yx=distance__yx, rad_scale=1, mode='sum')
Dn4000__r = K.radialProfile(Dn4000__yx, bin_r=bins, r__yx=distance__yx, rad_scale=1, mode=rp_mode)
color__r = K.radialProfile(color__yx, bin_r=bins, r__yx=distance__yx, rad_scale=1, mode=rp_mode)

plt.rcParams['legend.fontsize'] = 6
fig_height = fig_width * 0.75
f = plt.figure(4, figsize=(fig_width, fig_height))
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
plt.suptitle('Radial profiles - %s (%s) - pipeline %s - Voronoi %d' % (K.galaxyName, calId, pipeVer, sn_vor), fontsize=10)

ax_A_V = plt.subplot(gs[0, 0])
ax_A_V.plot(bins_lane_center, lane_A_V__r, 'k-', lw=lw, label='dust lane (starlight)')
ax_A_V.plot(bins_lane_center, fit_A_V__r, 'k:', lw=lw, label='dust lane (fitted)')
ax_A_V.plot(bins_lane_center, lane_A_V__r - fit_A_V__r, 'r--', lw=lw, label=r'dust lane (starlight - fitted)')
ax_A_V.plot(bins_center, A_V__r, 'k--', lw=lw, label='rest of galaxy (starlight)')
# ax_A_V.set_xlabel(r'radius [arcsec]')
ax_A_V.set_xticklabels([])
ax_A_V.set_ylabel(r'$A_V$ [mag]')
ax_A_V.set_xlim(0, bins.max())
ax_A_V.set_ylim(ylim_A_V[calId][0], ylim_A_V[calId][1])
ax_A_V.legend()

ax_at = plt.subplot(gs[0, 1])
ax_at.plot(bins_lane_center, lane_at_flux__r, 'k-', lw=lw)
ax_at.plot(bins_center, at_flux__r, 'k--', lw=lw)
# ax_at.set_xlabel(r'radius [arcsec]')
ax_at.set_xticklabels([])
ax_at.set_ylabel(r'$\langle \log t \rangle_L$')
ax_at.set_xlim(0, bins.max())
ax_at.set_ylim(ylim_at[calId][0], ylim_at[calId][1])

ax_Dn = plt.subplot(gs[1, 0])
ax_Dn.plot(bins_lane_center, lane_Dn4000__r, 'k-', lw=lw)
ax_Dn.plot(bins_center, Dn4000__r, 'k--', lw=lw)
ax_Dn.set_xlabel(r'radius [arcsec]')
ax_Dn.set_ylabel(r'$D_n(4000)$')
ax_Dn.set_xlim(0, bins.max())
ax_Dn.set_ylim(ylim_Dn4000[calId][0], ylim_Dn4000[calId][1])

ax_cl = plt.subplot(gs[1, 1])
ax_cl.plot(bins_lane_center, lane_color__r, 'k-', lw=lw)
ax_cl.plot(bins_center, color__r, 'k--', lw=lw)
ax_cl.set_xlabel(r'radius [arcsec]')
ax_cl.set_ylabel(r'Flux ratio $(6300 / 4250)$')
ax_cl.set_xlim(0, bins.max())

gs.tight_layout(f, rect=[0, -0.04, 1, 0.99])  
pdf.savefig(f)
plt.savefig('%s/radprof_%s_%s_v%02d.pdf' % (args.outdir, calId, pipeVer, sn_vor))


fig_height = 0.5 * fig_width

# Plot some diagnostic maps.
f = plt.figure(9, figsize=(fig_width, fig_height))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], height_ratios=[1])
plt.suptitle('Kinematics - %s (%s) - pipeline %s - Voronoi %s' % (K.galaxyName, calId, pipeVer, sn_vor), fontsize=10)

ax_im0 = plt.subplot(gs[0, 0])
im = ax_im0.imshow(K.v_0__yx, cmap='RdBu', vmin=vmin_v0[calId], vmax=vmax_v0[calId])
ax_im0.set_xticks([])
ax_im0.set_yticks([])
plt.colorbar(im, ax=ax_im0)
ax_im0.set_title(r'$v_0\ [km/s]$')

ax_im1 = plt.subplot(gs[0, 1])
im = ax_im1.imshow(K.v_d__yx, cmap='OrRd', vmin=vmin_vd[calId], vmax=vmax_vd[calId])
ax_im1.set_xticks([])
ax_im1.set_yticks([])
plt.colorbar(im, ax=ax_im1)
ax_im1.set_title(r'$v_d\ [km/s]$')

gs.tight_layout(f, rect=[0, 0, 1, 0.97])
pdf.savefig(f)
plt.savefig('%s/kinematics_%s_%s_v%02d.pdf' % (args.outdir, calId, pipeVer, sn_vor))  

pdf.close()

