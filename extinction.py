from pycasso import fitsQ3DataCube
import numpy as np
import matplotlib.pyplot as plt
from pystarlight.util.velocity_fix import SpectraVelocityFixer
from matplotlib import gridspec
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
                  'K0708': [(25.403225806451616, 41.322580645161288), (30.016129032258064, 39.370967741935488), (33.20967741935484, 40.79032258064516), (38.532258064516128, 37.951612903225808), (41.016129032258064, 38.661290322580648), (43.854838709677423, 37.064516129032256), (45.983870967741936, 35.645161290322584), (47.758064516129032, 33.516129032258064), (50.41935483870968, 31.387096774193552), (51.661290322580641, 29.79032258064516), (53.967741935483872, 26.951612903225808), (52.548387096774192, 26.064516129032256), (50.241935483870968, 27.661290322580648), (46.161290322580648, 29.08064516129032), (44.032258064516128, 30.322580645161288), (40.306451612903224, 32.451612903225808), (35.693548387096776, 33.870967741935488), (29.129032258064512, 34.58064516129032), (25.403225806451616, 36.532258064516128), (21.5, 37.596774193548384), (22.20967741935484, 39.903225806451616)],
                  'K0925': [(19.089861751152078, 27.013824884792626), (19.62903225806452, 31.327188940092171), (21.426267281105993, 35.281105990783409), (23.223502304147466, 40.133640552995395), (25.91935483870968, 43.1889400921659), (26.997695852534562, 45.345622119815673), (26.099078341013829, 47.142857142857146), (28.255760368663594, 48.940092165898619), (30.412442396313367, 49.299539170506918), (32.20967741935484, 50.018433179723502), (34.905529953917053, 48.041474654377886), (35.624423963133644, 49.479262672811068), (37.241935483870968, 50.557603686635943), (36.882488479262676, 52.714285714285722), (46.228110599078342, 56.488479262672811), (48.025345622119815, 55.230414746543786), (46.767281105990783, 51.995391705069125), (46.228110599078342, 49.479262672811068), (43.891705069124427, 48.041474654377886), (42.094470046082954, 48.221198156682028), (41.375576036866363, 46.783410138248854), (40.656682027649772, 43.009216589861758), (37.781105990783409, 39.774193548387103), (33.647465437788021, 35.640552995391708), (31.490783410138249, 31.866359447004612), (28.794930875576036, 27.912442396313367), (26.81797235023042, 23.599078341013822), (25.02073732718894, 19.824884792626726), (22.684331797235025, 16.230414746543779), (20.34792626728111, 14.433179723502302), (16.394009216589865, 16.05069124423963), (14.237327188940091, 19.285714285714285), (15.495391705069126, 22.700460829493089), (18.370967741935488, 24.317972350230413)],
                  }
dust_region_poly = {
                    'K0708': [(33.91935483870968, 37.596774193548384), (36.225806451612904, 37.774193548387096), (38.70967741935484, 37.241935483870968), (40.661290322580648, 37.241935483870968), (43.145161290322584, 35.645161290322584), (44.741935483870968, 33.870967741935488), (42.258064516129032, 33.338709677419352), (39.596774193548384, 33.693548387096776), (38.0, 34.758064516129032), (35.870967741935488, 35.112903225806456), (34.274193548387096, 35.112903225806456), (33.387096774193552, 36.177419354838712)],
                    'K0925':[(22.504608294930875, 31.866359447004612), (25.559907834101388, 34.023041474654377), (26.997695852534562, 38.336405529953922), (28.974654377880185, 40.313364055299544), (32.569124423963139, 44.447004608294932), (34.725806451612904, 43.1889400921659), (34.006912442396313, 38.875576036866363), (31.311059907834107, 37.437788018433181), (31.311059907834107, 33.843317972350235), (28.974654377880185, 31.506912442396313), (26.63824884792627, 28.990783410138249), (25.02073732718894, 27.552995391705068), (23.582949308755765, 29.170506912442399)],
                    }

# Plot customization
ylim_A_lambda = {'K0925': (0.5, 1.6),
                 'K0708': (0, 0.4),
                 }
ylim_A_V = {'K0925': (0, 2.1),
            'K0708': (-0.2, 0.7),
            }
ylim_at = {'K0925': (9.25, 10.25),
           'K0708': (9.25, 10.25),
           }
ylim_Dn4000 = {'K0925': (1.0, 2.0),
               'K0708': (1.0, 2.0),
           }

vmin_A_V = {'K0925': 0.0,
            'K0708': 0.0,
            }
vmax_A_V = {'K0925': 2.0,
            'K0708': 1.0,
            }

vmin_color = {'K0925': 1.0,
              'K0708': 1.0,
              }
vmax_color = {'K0925': 2.0,
              'K0708': 1.5,
              }

vmin_at = {'K0925': 9.25,
           'K0708': 9.25,
           }
vmax_at = {'K0925': 10.25,
           'K0708': 10.25,
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
plt.savefig('%s/maps_%s_%s_v%02d.pdf' % (args.outdir, calId, pipeVer, sn_vor))  
# plt.show()


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
plt.savefig('%s/spectra_%s_%s_v%02d.pdf' % (args.outdir, calId, pipeVer, sn_vor))  
# plt.show()

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

rp_mode = 'mean'
lane_A_V__r = K.radialProfile(lane_A_V__yx, bin_r=bins_lane, r__yx=lane_distance__yx, rad_scale=1, mode=rp_mode)
fit_A_V__r = K.radialProfile(fit_A_V__yx, bin_r=bins_lane, r__yx=lane_distance__yx, rad_scale=1, mode=rp_mode)
lane_at_flux__r = K.radialProfile(lane_at_flux__yx * lane_LobnSD__yx, bin_r=bins_lane, r__yx=lane_distance__yx, rad_scale=1, mode='sum') \
    / K.radialProfile(lane_LobnSD__yx, bin_r=bins_lane, r__yx=lane_distance__yx, rad_scale=1, mode='sum')
lane_Dn4000__r = K.radialProfile(lane_Dn4000__yx, bin_r=bins_lane, r__yx=lane_distance__yx, rad_scale=1, mode=rp_mode)
lane_color__r = K.radialProfile(lane_color__yx, bin_r=bins_lane, r__yx=lane_distance__yx, rad_scale=1, mode=rp_mode)

A_V__r = K.radialProfile(A_V__yx, bin_r=bins, r__yx=distance__yx, rad_scale=1, mode=rp_mode)
at_flux__r = K.radialProfile(at_flux__yx * LobnSD__yx, bin_r=bins, r__yx=distance__yx, rad_scale=1, mode='sum') \
    / K.radialProfile(LobnSD__yx, bin_r=bins, r__yx=distance__yx, rad_scale=1, mode='sum')
Dn4000__r = K.radialProfile(Dn4000__yx, bin_r=bins, r__yx=distance__yx, rad_scale=1, mode=rp_mode)
color__r = K.radialProfile(color__yx, bin_r=bins, r__yx=distance__yx, rad_scale=1, mode=rp_mode)

plt.rcParams['legend.fontsize'] = 6
fig_height = fig_width * 0.75
f = plt.figure(3, figsize=(fig_width, fig_height))
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
plt.savefig('%s/radprof_%s_%s_v%02d.pdf' % (args.outdir, calId, pipeVer, sn_vor))
# plt.show()
