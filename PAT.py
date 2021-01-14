# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 12:24:33 2020
From command line run:
	pip install bokeh
	bokeh serve C:\Path\to\file\PAT.py
Open in browser: "localhost:5006"
"""
__author__ = "Jacob Harris"
import numpy as np
from bokeh.layouts import column, row, gridplot
from bokeh.models import Slider, Rect, Div, HoverTool, SingleIntervalTicker, LinearAxis
from bokeh.plotting import ColumnDataSource, figure, curdoc
from bokeh.models.widgets import RadioButtonGroup

k=2*np.pi
dbmin = 20 # lowest shown value on log scale
elmax = 10 # max number of elements
theta = np.pi/180*np.arange(0, 360, 1) # X-axis of graph
psi = np.pi*np.arange(-2,2,0.01) # cartesian x
plotsize = 300
description = Div(text="""Interactive tool built to show the ideal pattern created by identical elments
 alligned along the y-axis with uniform spacing and phase shift. <i> by Jacob Harris</i>""")

def Currents(currents, d, beta):
    """Input array of element currents"""
    def AFun(phi):
        """Returns a function representing the array factor"""
        Psi = k*d*np.cos(phi)+beta
        af = 0j
        for i in range(len(currents)):
            af += currents[i]*np.exp(1j*Psi*i)
        return abs(af)
    return AFun
def CurP(currents):
    """AF as a function of Psi"""
    def AFun(Psi):
        af = 0j
        for i in range(len(currents)):
            af += currents[i]*np.exp(1j*Psi*i)
        return abs(af)
    return AFun

def polardict(r):
    """Creates polar plot for ColumnDataSourse
    x,y for plot; r,t for hover_tool"""
    x = r*np.sin(theta)
    y = r*np.cos(theta)
    return dict(x=x, y=y, r=r, t=theta*180/np.pi)

def centered(elmts):
    """Returns a numpy array of specified length centered at zero"""
    return np.arange(elmts)-(elmts-1)/2

#Interactive Elements
element_slider = Slider(start=1, end=elmax, value=4, step=1, title="Number of Elements")
space_slider = Slider(start=0, end=2, value=0.5, step=.05, title="Spacing (wavelengths)")
phase_slider = Slider(start=-1, end=1, value=0, step=.05, title="Phase Shift (pi radians)")
amplitude_radio = RadioButtonGroup(labels=['Uniform', 'Triangular', 'Raised Cos', 'Double RC', 'Parabola'], active=0)
amplitude = {
        0: lambda elmts: np.ones(elmts),
        1: lambda elmts: (elmts+1)/2-abs(centered(elmts)),
        2: lambda elmts: 1+np.cos(k*(centered(elmts))/elmts),
        3: lambda elmts: 2+np.cos(k*(centered(elmts))/elmts),
        4: lambda elmts: 1-(2*centered(elmts)/elmts)**2
        }
element_radio = RadioButtonGroup(labels=['Isotropic', 'HW Dipole', 'Endfire', 'Kraken V'], active=0)
e_patterns = {
        0: np.ones(len(theta)),
        1: np.sin(theta)**2,
        2: Currents([1,1,1,1,1], 0.4, 0.8*np.pi)(theta+np.pi/2)/5,
        3: Currents([1,1,1,1,1,1], 0.65, 10/180*np.pi)(theta)/6,
        }

d_abs   = ColumnDataSource()
d_norm  = ColumnDataSource()
d_log   = ColumnDataSource()
d_car   = ColumnDataSource()
d_elem  = ColumnDataSource()
d_prod  = ColumnDataSource()
d_amp   = ColumnDataSource()
d_box   = ColumnDataSource()

def callback(attr, old, new):
    """Calculations run at each button change"""
    v_e = element_slider.value
    space = space_slider.value
    beta = np.pi*phase_slider.value
    
    amp = amplitude.get(amplitude_radio.active)
    cur = amp(v_e)
    cur = cur/cur.max() # Normalize
    d_amp.data = dict(x=[f'E{x}' for x in range(v_e)], y=cur)
    
    r = Currents(cur, space, beta)(theta)
    d_abs.data = polardict(r)
    rn = r/r.max()
    d_norm.data = polardict(rn)
    rl = 10*np.log10(rn)+dbmin
    rl = np.where(rl>0, rl, 0)#hide neg values
    d_log.data = polardict(rl)
    re = e_patterns.get(element_radio.active)
    d_elem.data = polardict(re)
    d_prod.data = polardict(re*rn)
    
    afp = CurP(cur)(psi)
    d_car.data = dict(x=psi, y=afp/afp.max())
    d_box.data = dict(x=[beta], w=[2*k*space], y=[0.5], h=[4])#Data must be col
    
#Initialize
callback('','','')

#Plot
p_abs = figure(plot_width=plotsize, aspect_ratio=1, title="Array Factor",
              x_range=[-10, 10], y_range=[-10, 10], tools='pan,wheel_zoom,reset,save')
p_abs.add_tools(HoverTool(tooltips=[('magnitude', '@r'), ('angle (deg)', '@t')]))
p_abs.line('x', 'y', source=d_abs, line_width=3, line_alpha=0.6)
p_abs.annulus(x=0, y=0, inner_radius=10, outer_radius=20, fill_color='#ffffff', line_color='#000000')
p_abs.min_border=20

p_norm = figure(plot_width=plotsize, aspect_ratio=1, title="Normalized",
              x_range=[-1, 1], y_range=[-1, 1], tools='pan,wheel_zoom,reset,save')
p_norm.add_tools(HoverTool(tooltips=[('magnitude', '@r'), ('angle (deg)', '@t')]))
p_norm.annulus(x=0, y=0, inner_radius=1, outer_radius=2, fill_color='#ffffff', line_color='#000000')
p_norm.line('x', 'y', source=d_norm, line_width=3, line_alpha=0.6)
p_norm.min_border=20

p_log = figure(plot_width=plotsize, aspect_ratio=1, title="Logarithmic (dB)",
              x_range=[-dbmin, dbmin], y_range=[-dbmin, dbmin], tools='pan,wheel_zoom,reset,save')
p_log.add_tools(HoverTool(tooltips=[('magnitude', '@r'), ('angle (deg)', '@t')]))
p_log.annulus(x=0, y=0, inner_radius=dbmin, outer_radius=dbmin+10, fill_color='#ffffff', line_color='#000000')
p_log.line('x', 'y', source=d_log, line_width=3, line_alpha=0.6)
p_log.min_border=20

p_car = figure(plot_width=plotsize, aspect_ratio=1, title="Cartesian", x_axis_type=None,
              x_range=[-k, k], y_range=[0, 1], tools='hover,pan,wheel_zoom,reset,save')
p_car.add_layout(LinearAxis(ticker=SingleIntervalTicker(interval=np.pi, num_minor_ticks=10)), 'below')
p_car.line('x', 'y', source=d_car, line_width=3, line_alpha=0.6)
glyph = Rect(x="x", y="y", width="w", height="h", fill_alpha=0.2, fill_color="#cab2d6")
p_car.add_glyph(d_box, glyph)
p_car.min_border=20

p_elem = figure(plot_width=plotsize, aspect_ratio=1, title="Element Pattern",
              x_range=[-1, 1], y_range=[-1, 1], tools='pan,wheel_zoom,reset,save')
p_elem.add_tools(HoverTool(tooltips=[('magnitude', '@r'), ('angle (deg)', '@t')]))
p_elem.line('x', 'y', source=d_elem, line_width=3, line_alpha=0.6, line_color="orange")
p_elem.line('x', 'y', source=d_prod, line_width=3, line_alpha=0.6)
p_elem.min_border=20

e_names = [f'E{x}' for x in range(elmax)]#x_range needs list of strings
p_amp = figure(plot_width=plotsize, aspect_ratio=1, title="Amplitude",
              x_range=e_names, y_range=[0, 1.2], tools='hover,pan,wheel_zoom,reset,save')
p_amp.vbar(source=d_amp, x='x', top='y', width=0.5)
p_amp.min_border=20

#Running
element_slider.on_change('value', callback)
space_slider.on_change('value', callback)
phase_slider.on_change('value', callback)
amplitude_radio.on_change('active', callback)
element_radio.on_change('active', callback)

#Document
bokeh_doc = curdoc()
grid = gridplot([[p_abs, p_norm, p_log],[p_car, p_elem, p_amp]],
                sizing_mode='scale_both', toolbar_location='right')
bokeh_doc.add_root(column([grid,
                           element_slider, space_slider, phase_slider,
                           Div(text="Amplitude Taper: "), amplitude_radio, 
                           Div(text="Element Pattern: "), element_radio, 
                           description], sizing_mode='scale_width'))
bokeh_doc.title = "Phased Array Tool"

