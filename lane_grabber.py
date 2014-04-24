from pycasso import fitsQ3DataCube
import matplotlib.pyplot as plt
import argparse
from matplotlib.lines import Line2D

################################################################################        

class PolyGrabber:
    
    def __init__(self, fig):
        self.fig = fig
        self.line = Line2D([],[], lw=2.0, color='k', figure=self.fig)
        self.axes = self.fig.axes[0]
        self.axes.add_line(self.line)
        self.curImage = self.axes.images[0]
        self.raiseImage('1')
        self.redraw()
        

    def redraw(self):
        self.fig.canvas.draw()
        
        
    def connect(self):
        self.fig.canvas.mpl_connect('button_press_event', self.onButtonPress)
        self.fig.canvas.mpl_connect('key_press_event', self.onKeyPress)
        
        
    def onButtonPress(self, ev):
        if ev.button == 1:
            self.addPoint(ev.xdata, ev.ydata)
        elif ev.button == 3:
            self.delLastPoint()
        else:
            return
        self.redraw()


    def onKeyPress(self, ev):
        if ev.key in ['1', '2', '3', '4', '5']:
            self.raiseImage(ev.key)
        elif ev.key == 'z':
            self.changeCLim(dmin=-0.05)
        elif ev.key == 'x':
            self.changeCLim(dmin=0.05)
        elif ev.key == 'c':
            self.changeCLim(dmax=-0.05)
        elif ev.key == 'v':
            self.changeCLim(dmax=0.05)
        elif ev.key == 'enter':
            self.dumpPoly()
        elif ev.key == ' ':
            vmin, vmax = self.curImage.get_clim()
            print 'vmin=%.2f, vmax=%.2f' % (vmin, vmax)            
        self.redraw()

    
    def changeCLim(self, dmin=0.0, dmax=0.0):
        vmin, vmax = self.curImage.get_clim()
        rng = vmax - vmin
        vmin += rng * dmin
        vmax += rng * dmax
        self.curImage.set_clim(vmin, vmax)


    def raiseImage(self, key):
        try:
            ev_id = int(key) - 1
        except:
            pass
        for i in xrange(len(self.axes.images)):
            im = self.axes.images[i]
            if i == ev_id:
                im.set_alpha(1.0)
                self.curImage = im
                self.axes.set_title(im.get_label())
            else:
                im.set_alpha(0.0)


    def addPoint(self, x, y):
        xd = list(self.line.get_xdata())
        yd = list(self.line.get_ydata())
        xd.append(x)
        yd.append(y)
        self.line.set_data(xd, yd)

    
    def delLastPoint(self):
        xd = list(self.line.get_xdata())
        yd = list(self.line.get_ydata())
        if len(xd) == 0:
            return
        xd.pop()
        yd.pop()
        self.line.set_data(xd, yd)

    def dumpPoly(self):
        xd = list(self.line.get_xdata())
        yd = list(self.line.get_ydata())
        if len(xd) < 3:
            return
        print zip(xd, yd)

################################################################################        
    
parser = argparse.ArgumentParser(description='Compute redening laws.')
parser.add_argument('db', type=str, nargs=1,
                   help='CALIFA superFITS.')

args = parser.parse_args()

print 'Opening file %s' % args.db[0]
K = fitsQ3DataCube(args.db[0])

plt.ioff()
f = plt.figure(1, figsize=(7,7))
ax = f.add_subplot(111)
ax.imshow(K.qSignal, cmap='Blues', label=r'Image @ $5635 \AA$')
ax.imshow(K.A_V__yx, cmap='Reds', vmin=0.0, vmax=2.0, label=r'$A_V$')
ax.imshow(K.at_flux__yx, cmap='Reds', vmin=9.0, vmax=11.0, label=r'$\langle \log t \rangle_L$')
ax.imshow(K.Dn4000__yx, cmap='Reds', vmin=0.0, vmax=2.0, label=r'$D_n(4000)$')
im = ax.imshow(K.fluxRatio__yx(6100, 6500, 4000, 4500), vmin=0.5, vmax=2.0, cmap='RdBu_r',
               label=r'Flux ratio ($F_{6300\AA} / F_{4250\AA})$')
ax.set_title(im.get_label())

grabber = PolyGrabber(f)
grabber.connect()

print '''Use the keys 1-5 to cycle between the images.
Left click adds a vertex in the current mouse position.
Right click deletes the last point.

The keys z, x decrease or increase the vmin of the current image.
The keys c, v decrease or increase the vmax of the current image.
 
Press <space> to print vmin & vmax of the current image.

Press <enter> when done, it will print the polygon as a python list.



'''
plt.show()
