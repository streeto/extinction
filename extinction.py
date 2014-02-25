from pycasso import fitsQ3DataCube
import numpy as np
import matplotlib.pyplot as plt
from pystarlight.util.velocity_fix import SpectraVelocityFixer
from matplotlib import gridspec
from pystarlight.util.redenninglaws import Cardelli_RedLaw, Calzetti_RedLaw
from pycasso.util import get_polygon_mask
import argparse

parser = argparse.ArgumentParser(description='Compute redening laws.')
parser.add_argument('db', type=str, nargs=1,
                   help='CALIFA superFITS.')
parser.add_argument('--fixk', dest='fixKinematics', action='store_true',
                    help='Fix kinematics.')
parser.add_argument('--vd', dest='targetVd', default=0.0, type=float,
                    help='Velocity dispersion to be convolved with the spectra (if --fixk).')
parser.add_argument('--lll', dest='lambdaLowerLimit', default=0.0, type=float,
                    help='Lower limit in lambda when fitting.')
parser.add_argument('--qbick', dest='qbickId', type=str,
                    help='Qbick ID (for display purposes).')

args = parser.parse_args()

print 'Opening file %s' % args.db[0]
K = fitsQ3DataCube(args.db[0])

# TODO: Parameterize the dust regions.
dust_lane_poly = {'K0708': [(23, 40),
                            (43, 40),
                            (57, 29),
                            (52, 25),
                            (41, 33),
                            (26, 35),
                            ],
                  'K0925': [(9, 19),
                            (20, 41),
                            (29, 48),
                            (42, 51),
                            (45, 44),
                            (36, 39),
                            (31, 33),
                            (30, 30),
                            (17, 9),
                            ],
                  }
dust_region_poly = {'K0708': [(39, 38),
                              (46, 34),
                              (41, 32),
                              (39, 34),
                              ],
                    'K0925': [(18, 29),
                              (25, 34),
                              (25, 41),
                              (34, 40),
                              (31, 35),
                              (28, 28),
                              (22, 24),
                              ],
                    }


# Compute bit masks.
dust_lane__yx = get_polygon_mask(dust_lane_poly[K.califaID], K.qSignal.shape) & (K.pixelDistance__yx > 3) & K.qMask
dust_region__yx = get_polygon_mask(dust_region_poly[K.califaID], K.qSignal.shape) & (K.pixelDistance__yx > 3) & K.qMask

print 'Finding geometry...'
for i in xrange(10):
    img = np.ma.masked_array(K.qSignal, mask=~K.qMask)
    img[dust_lane__yx] = np.ma.masked
    img, mask = K.fillImage(img)
    pa, ba = K.getEllipseParams(img, mask)
    K.setGeometry(pa, ba)


# Plot some diagnostic maps.
plt.ioff()
plt.rcParams['legend.fontsize'] = 10
f = plt.figure(1, figsize=(7,7))
gs = gridspec.GridSpec(2, 2, width_ratios=[1,1], height_ratios=[1,1])
plt.suptitle('%s (%s) - %s' % (K.califaID, K.galaxyName, args.qbickId))

mask_image = np.zeros_like(K.qSignal)
mask_image[dust_lane__yx] = 2.0
mask_image[dust_region__yx] = 3.0

ax_im0 = plt.subplot(gs[0,0])
im = ax_im0.imshow(K.qSignal, cmap='jet')
ax_im0.contour(mask_image, levels=[1.9, 2.9], colors=['gray', 'k'], linewidths=2)
ax_im0.set_xticks([])
ax_im0.set_yticks([])
plt.colorbar(im, ax=ax_im0)
ax_im0.set_title(r'Image @ $5635 \AA$')

ax_im1 = plt.subplot(gs[0,1])
im = ax_im1.imshow(K.fluxRatio__yx(4000, 4500, 6100, 6500), cmap='jet')
ax_im1.contour(mask_image, levels=[1.9, 2.9], colors=['gray', 'k'], linewidths=2)
ax_im1.set_xticks([])
ax_im1.set_yticks([])
plt.colorbar(im, ax=ax_im1)
ax_im1.set_title(r'Flux ratio ($F_{6300\AA} / F_{4250\AA})$')

ax_im2 = plt.subplot(gs[1,0])
im = ax_im2.imshow(K.A_V__yx, cmap='jet')
ax_im2.contour(mask_image, levels=[1.9, 2.9], colors=['gray', 'k'], linewidths=2)
ax_im2.set_xticks([])
ax_im2.set_yticks([])
plt.colorbar(im, ax=ax_im2)
ax_im2.set_title('$A_V$')

ax_im3 = plt.subplot(gs[1,1])
im = ax_im3.imshow(K.at_flux__yx, cmap='jet')
ax_im3.contour(mask_image, levels=[1.9, 2.9], colors=['gray', 'k'], linewidths=2)
ax_im3.set_xticks([])
ax_im3.set_yticks([])
plt.colorbar(im, ax=ax_im3)
ax_im3.set_title(r'$\langle \log t \rangle_L$')

gs.tight_layout(f, rect=[0, 0.03, 1, 0.97])
plt.savefig('maps_%s_%s.png' % (K.califaID, args.qbickId))  
plt.show()


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
bad_lambda = f_flag__lyx[...,dust_region__yx].sum(axis=1) > (npix / 2)

# Flag the flagged
f_obs__lyx[f_flag__lyx] = np.ma.masked

# Repeat the trick to fill the spectra at the dusty pixels.
print 'Computing intrinsic spectra...'
f_clean__lyx = f_obs__lyx.copy()
f_clean__lyx[:,dust_lane__yx] = np.ma.masked
for i in xrange(K.Nl_obs):
    if bad_lambda[i]: continue
    img = f_clean__lyx[i].copy()
    img[dust_lane__yx] = np.ma.masked
    img, _ = K.fillImage(img)
    f_clean__lyx[i, dust_lane__yx] = img[dust_lane__yx]

# Extinction in each pixel.
tau__lz = np.log(f_clean__lyx[...,dust_lane__yx] / f_obs__lyx[...,dust_lane__yx])
almost_one = np.log10(np.exp(1.0)) * 2.5
A__lz = tau__lz * almost_one

# Integrated dust lane spectra.
dusty_int_spec = np.median(f_obs__lyx[..., dust_region__yx], axis=1)
clean_int_spec = np.median(f_clean__lyx[..., dust_region__yx], axis=1)
dusty_int_spec[bad_lambda] = np.ma.masked
clean_int_spec[bad_lambda] = np.ma.masked

# Plot the spectra and extinction.
f = plt.figure(1, figsize=(7,7))
gs = gridspec.GridSpec(2, 1, width_ratios=[1,], height_ratios=[1,1])
plt.suptitle('%s (%s) - %s' % (K.califaID, K.galaxyName, args.qbickId))

ax_spec = plt.subplot(gs[0,0])
ax_spec.plot(K.l_obs, dusty_int_spec, 'r-', label='dust lane')
ax_spec.plot(K.l_obs, clean_int_spec, 'b-', label='intrinsic')
ax_spec.set_xticklabels([])
ax_spec.set_ylabel(r'Flux')
ax_spec.set_xlim(K.l_obs.min(), K.l_obs.max())
ax_spec.set_ylim(0.0, clean_int_spec.max())
ax_spec.legend()

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
ax_ext = plt.subplot(gs[1,0])
ax_ext.plot(K.l_obs, A__l, 'k-', markeredgecolor='none',
            label=r'Observed')
ax_ext.plot(K.l_obs, ccm_A__l(K.l_obs), 'b--', markeredgecolor='none',
            label=r'CCM fit, $R_V = %.2f$, $A_V = %.2f$)' % (ccm_A__l.R_V.value, ccm_A__l.A_V.value))
ax_ext.plot(K.l_obs, RV31ccm_A__l(K.l_obs), 'b:', markeredgecolor='none',
            label=r'CCM fit, $R_V = %.2f$, $A_V = %.2f$)' % (RV31ccm_A__l.R_V.value, RV31ccm_A__l.A_V.value))
ax_ext.plot(K.l_obs, cal_A__l(K.l_obs), 'r--', markeredgecolor='none',
            label=r'CAL fit, $R_V = %.2f$, $A_V = %.2f$)' % (cal_A__l.R_V.value, cal_A__l.A_V.value))
ax_ext.plot(K.l_obs, RV405cal_A__l(K.l_obs), 'r:', markeredgecolor='none',
            label=r'CAL fit, $R_V = %.2f$, $A_V = %.2f$)' % (RV405cal_A__l.R_V.value, RV405cal_A__l.A_V.value))

ax_ext.set_xlim(K.l_obs.min(), K.l_obs.max())
ax_ext.set_ylim(0.0, ccm_A__l.A_V.value * 2.0)
ax_ext.set_xlabel(r'wavelength $[\AA]$')
ax_ext.set_ylabel(r'$A_\lambda$')
ax_ext.legend()

gs.tight_layout(f, rect=[0, 0.03, 1, 0.97])  
plt.savefig('spectra_%s_%s.png' % (K.califaID, args.qbickId))  
plt.show()

# Plot radial profiles
distance__yx = np.ma.masked_where(dust_lane__yx | ~K.qMask, K.pixelDistance__yx, copy=True)
A_V__yx = np.ma.masked_where(dust_lane__yx | ~K.qMask, K.A_V__yx, copy=True)
at_flux__yx = np.ma.masked_where(dust_lane__yx | ~K.qMask, K.at_flux__yx, copy=True)
Dn4000__yx = np.ma.masked_where(dust_lane__yx | ~K.qMask, K.Dn4000__yx, copy=True)
bins = np.arange(distance__yx.max() + 1)
bins_center = bins[:-1] + 0.5

lane_distance__yx = np.ma.masked_where(~dust_lane__yx, K.pixelDistance__yx, copy=True)
lane_A_V__yx = np.ma.masked_where(~dust_lane__yx, K.A_V__yx, copy=True)
lane_at_flux__yx = np.ma.masked_where(~dust_lane__yx, K.at_flux__yx, copy=True)
lane_Dn4000__yx = np.ma.masked_where(~dust_lane__yx, K.Dn4000__yx, copy=True)
bins_lane = np.arange(lane_distance__yx.min(), lane_distance__yx.max() + 1)
bins_lane_center = bins_lane[:-1] + 0.5

n_lane_pix = (A__lz.shape[1])
fit_A_V__z = np.empty((n_lane_pix,))
fit_A_V__yx = np.ma.masked_all_like(K.A_V__yx)
for zz in xrange(n_lane_pix):
    _model = cal_reddening(R_V=cal_A__l.R_V.value, A_V=cal_A__l.A_V.value)
    _model.R_V.fixed = True
    _cal_A__l = fit(_model, K.l_obs[fitmask], A__lz[fitmask, zz])
    if fit.fit_info['ierr'] in [1,2,3,4]:
        fit_A_V__z[zz] = _cal_A__l.A_V.value
fit_A_V__yx[dust_lane__yx] = fit_A_V__z

rp_mode = 'mean'
lane_A_V__r = K.radialProfile(lane_A_V__yx, bin_r=bins_lane, r__yx=lane_distance__yx, rad_scale=1, mode=rp_mode)
fit_A_V__r = K.radialProfile(fit_A_V__yx, bin_r=bins_lane, r__yx=lane_distance__yx, rad_scale=1, mode=rp_mode)
lane_at_flux__r = K.radialProfile(lane_at_flux__yx, bin_r=bins_lane, r__yx=lane_distance__yx, rad_scale=1, mode=rp_mode)
lane_Dn4000__r = K.radialProfile(lane_Dn4000__yx, bin_r=bins_lane, r__yx=lane_distance__yx, rad_scale=1, mode=rp_mode)

A_V__r = K.radialProfile(A_V__yx, bin_r=bins, r__yx=distance__yx, rad_scale=1, mode=rp_mode)
at_flux__r = K.radialProfile(at_flux__yx, bin_r=bins, r__yx=distance__yx, rad_scale=1, mode=rp_mode)
Dn4000__r = K.radialProfile(Dn4000__yx, bin_r=bins, r__yx=distance__yx, rad_scale=1, mode=rp_mode)

f = plt.figure(2, figsize=(8,10))
gs = gridspec.GridSpec(3, 1, width_ratios=[1,], height_ratios=[1,1,1])
plt.suptitle('%s (%s) - %s' % (K.califaID, K.galaxyName, args.qbickId))

ax_A_V = plt.subplot(gs[0,0])
ax_A_V.plot(bins_lane_center, lane_A_V__r, 'k-', label='dust lane (starlight)')
ax_A_V.plot(bins_lane_center, fit_A_V__r, 'k:', label='dust lane (fitted)')
ax_A_V.plot(bins_center, A_V__r, 'k--', label='rest of galaxy (starlight)')
ax_A_V.set_xlabel(r'radius [arcsec]')
ax_A_V.set_ylabel(r'$A_V$ [mag]')
ax_A_V.legend()

ax_at = plt.subplot(gs[1,0])
ax_at.plot(bins_lane_center, lane_at_flux__r, 'k-')
ax_at.plot(bins_center, at_flux__r, 'k--')
ax_at.set_xlabel(r'radius [arcsec]')
ax_at.set_ylabel(r'$\langle \log t \rangle_L$')

ax_Dn = plt.subplot(gs[2,0])
ax_Dn.plot(bins_lane_center, lane_Dn4000__r, 'k-')
ax_Dn.plot(bins_center, Dn4000__r, 'k--')
ax_Dn.set_xlabel(r'radius [arcsec]')
ax_Dn.set_ylabel(r'$D_n(4000)$')

gs.tight_layout(f, rect=[0, 0.03, 1, 0.97])  
plt.savefig('radprof_%s_%s.png' % (K.califaID, args.qbickId))  
plt.show()
