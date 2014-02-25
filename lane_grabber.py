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
        self.fig.canvas.draw()


    def onKeyPress(self, ev):
        if ev.key in ['1', '2', '3', '4', '5']:
            self.raiseImage(ev.key)
        elif ev.key == 'enter':
            self.dumpPoly()
        self.fig.canvas.draw()


    def raiseImage(self, key):
        try:
            ev_id = int(key) - 1
        except:
            pass
        for i in xrange(len(self.axes.images)):
            if i == ev_id:
                self.axes.images[i].set_alpha(1.0)
            else:
                self.axes.images[i].set_alpha(0.0)


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
ax.imshow(K.qSignal, cmap='jet')
ax.imshow(K.A_V__yx, cmap='jet')
im = ax.imshow(K.fluxRatio__yx(4000, 4500, 6100, 6500), cmap='jet')
ax.set_title(r'Image @ $5635 \AA$')

grabber = PolyGrabber(f)
grabber.connect()

plt.show()
