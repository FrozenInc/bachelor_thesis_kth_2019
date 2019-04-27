import pickle
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as font_manager
import csv

#fontpath = './transfonter/cmr10.ttf' # Elis: was './transfonter/Palatino-Roman.ttf'
fontpath = './transfonter/FCR.ttf' # Elis: was './transfonter/Palatino-Roman.ttf'
prop = font_manager.FontProperties(fname=fontpath)
matplotlib.rcParams['font.family'] = prop.get_name()

from pylab import *

LIGHT_GRAY = (.6, .6, .6)
DARK_GRAY = (.4, .4, .4)
LIGHT_ORANGE = (1., .6, .4)
DARK_ORANGE = (1., .4, 0.)
PURPLE = (0.6, 0., 0.8)
LIGHT_BLUE = (0.0, 0.8, 1.)
DARK_BLUE = (0.0, 0.4, 1.)

dt = 0.1

# TODO: check the report and see which plot function is used

def ls(pattern):
    output = subprocess.check_output("ls {}".format(pattern), shell=True).splitlines()
    return output


def load(filename, human_is_first=True):
    # load a saved .pickle file
    with open(filename) as f:
        ret = pickle.load(f)
    # human first, robot later
    # it unpacks the .pickle file
    u, x = ret
    # ur, xr = robot
    # uh, xh = human
    # uo, xo = obstacle (not used)

    #human_is_first = True # do this if human is a follower
    if(human_is_first):
        uh, ur, uo = u
        xh, xr, xo = x
    else:
        ur, uh, uo = u
        xr, xh, xo = x
        
    t = arange(len(xh))*dt
    if filename.split('/')[0] in ['data', 'unique_data']:
        user = '0'
        world = filename.split('/')[-1].split('-')[0]
        condition = 'purple'
        if world == 'world2':
            condition = 'purple (left)'
        elif world == 'world3':
            condition = 'purple (right)'
        traj = 0
    else:
        #user = int(filename.split('/')[1].split('-')[0][1:])
        user = 1
        traj = 1
        world = filename.split('/')[-1].split('-')[0]
        #traj = int(filename.split('/')[-1].split('-')[-1].split('.')[0])
        condition = {
            'world0': 'gray',
            'world1': 'orange',
            'world2': 'orange',
            'world3': 'blue',
            'world4': 'orange',
            'world5': 'gray',
            'test': '',
            'world_kex.pickle': 'orange'
        }[world]
    # human is uh and xh
    # robot is ur and xr
    # we need to make sure that it works with our code where we always have the follower first, but the human/robot changes
    return {
        'uh': asarray(uh), 'ur': asarray(ur), 'xh': asarray(xh), 'xr': asarray(xr), 't': t,
        'user': user,
        'condition': condition,
        'world': world,
        'traj': traj
    }

def isempty(data):
    return len(data['t'])==0

def extend(a, w):
    if len(a)>=w:
        return a[:w]
    return concatenate([a, nan*ones(w-len(a))])

def cextend(a, w):
    if len(a)>=w:
        return a[:w]
    return concatenate([a, asarray([a[-1]]*(w-len(a)))])

#worlds = ['world{}'.format(i) for i in range(6)] + ['test']
"""worlds = ['world_kex']
datasets = {}
for w in worlds:
    #print(ls("saved_data/*/{}*".format(w)))
    print(ls("saved_data/{}*".format(w)))
    datasets[w] = [load(x) for x in ls("saved_data/{}*".format(w))]
    datasets[w] = [data for data in datasets[w] if not isempty(data)]

for w, dataset in datasets.items():
    print '{}: {} samples'.format(w, len(dataset))

print '-'*20
"""



def plotAnimate():
    T = 30
    #opt = load('data/world4-opt.pickle')
    opt = load('data/world_kex_swithing_leader.pickle')
    def setup():
        figure(figsize=(5, 5))
        gca().spines['right'].set_visible(False)
        gca().spines['top'].set_visible(False)
        gca().spines['left'].set_visible(True)
        gca().spines['bottom'].set_visible(True)
        gca().xaxis.set_ticks_position('bottom')
        gca().yaxis.set_ticks_position('left')
        xlim(-0.6, 1.)
        ylim(-0.4, 1.2)
    def animate(frame, w, col1, col2):
        if frame<0:
            return
        dataset = datasets[w]
        x = np.stack(extend(data['xr'][:, 0], T) for data in dataset)
        y = np.stack(extend(data['xh'][:, 1], T) for data in dataset)
        mx = nanmean(x, axis=0)
        my = nanmean(y, axis=0)
        s = sqrt(nanstd(x, axis=0)**2+nanstd(y, axis=0)**2)
        n = (~isnan(mx)).sum(axis=0)
        s = s/sqrt(n)
        frame = min(frame, len(my))
        fill_betweenx(my[:frame], (mx-s)[:frame], (mx+s)[:frame], color=col2)
        plot(mx[:frame], my[:frame], color=col1, linewidth=3.)
        return frame == len(my)
    def anim_purp(frame):
        if frame<0:
            return
        frame = min(frame, len(opt['xr']))
        plot(opt['xr'][:frame, 0], opt['xh'][:frame, 1], color=LIGHT_BLUE, linewidth=3)
        return frame == len(opt['xr'])
    f = [0, 0, 0]
    ind = 0
    while ind<len(f):
        setup()
        r = [False, False, False]
        r[0] = anim_purp(f[0])
        r[1] = animate(f[1], 'world_kex', DARK_ORANGE, LIGHT_ORANGE)
        r[2] = animate(f[2], 'world_kex', DARK_GRAY, LIGHT_GRAY)
        if r[ind]:
            ind += 1
        savefig('images/plot-{:04d}.png'.format(sum(f)), transparent=True)
        if ind==len(f):
            break
        f[ind] += 1


def plot45():
    T = 30
    plots = {}
    def setup(flag1=True, flag2=False):
        gca().spines['right'].set_visible(flag2)
        gca().spines['top'].set_visible(flag2)
        gca().spines['left'].set_visible(flag1)
        gca().spines['bottom'].set_visible(flag1)
        #gca().spines['bottom'].set_position('zero')
        #gca().spines['left'].set_smart_bounds(True)
        #gca().spines['bottom'].set_smart_bounds(True)
        gca().xaxis.set_ticks_position('bottom')
        gca().yaxis.set_ticks_position('left')
        if not flag1 and not flag2:
            tick_params(
                axis='x',
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off')
            tick_params(
                axis='y',
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off')
            gca().get_xaxis().set_ticks([])
            gca().get_yaxis().set_ticks([])
    figure(figsize=(9, 5))
    opt = load('data/world4-opt.pickle')
    subplot(1, 2, 1, aspect='equal')
    setup(False, False)
    #axis('equal')
    xlim(-0.8, 0.8)
    ylim(-0.5, 1.1)
    xlabel('\nx of autonomous vehicle\n(a)')
    ylabel('y of human driven vehicle')
    for w, color in [('world4', LIGHT_ORANGE), ('world5', LIGHT_GRAY)]:
        dataset = datasets[w]
        for data in dataset[::2]:
            plot(data['xr'][:, 0], data['xh'][:, 1], color=color, linewidth=0.7)#, linestyle='--' if w=='world2' else '-')
    plot(opt['xr'][:, 0], opt['xh'][:, 1], color=PURPLE, linewidth=1)#, linestyle='--' if w=='world2' else '-')
    gca().add_patch(Rectangle((-0.065, -0.065), 0.13, 0.13, color=LIGHT_BLUE))
    #annotate('intersection', xy=(0, 0), xytext=(0.1, 0.3), arrowprops=dict(facecolor='black', shrink=0.01))
    subplot(1, 2, 2, aspect='equal')
    setup(True)
    #axis('equal')
    xlim(-0.8, 0.8)
    ylim(-0.5, 1.1)
    xlabel('x of autonomous vehicle\n(b)')
    ylabel('y of human driven vehicle')
    plots['intersection'] = gca().add_patch(Rectangle((-0.065, -0.065), 0.13, 0.13, color=LIGHT_BLUE))
    for w, col1, col2 in [('world4', DARK_ORANGE, LIGHT_ORANGE), ('world5', DARK_GRAY, LIGHT_GRAY)]:
        dataset = datasets[w]
        x = np.stack(extend(data['xr'][:, 0], T) for data in dataset)
        y = np.stack(extend(data['xh'][:, 1], T) for data in dataset)
        mx = nanmean(x, axis=0)
        my = nanmean(y, axis=0)
        s = sqrt(nanstd(x, axis=0)**2+nanstd(y, axis=0)**2)
        n = (~isnan(mx)).sum(axis=0)
        s = s/sqrt(n)
        fill_betweenx(my, mx-s, mx+s, color=col2)
        plots[w], = plot(mx, my, color=col1, linewidth=3.)
    plots['opt'], = plot(opt['xr'][:, 0], opt['xh'][:, 1], color=PURPLE, linewidth=3)#, linestyle='--' if w=='world2' else '-')
    figlegend((plots['opt'], plots['world5'], plots['world4'], plots['intersection']), ('Learned Human Model', 'Avoid Human', 'Affect Human', 'Intersection'), 'upper center', ncol=4, fontsize=12)
    savefig('plots/plot45.pdf')

def plot23():
    T = 50
    plots = {}
    def setup(flag1=True, flag2=False):
        gca().spines['right'].set_visible(flag2)
        gca().spines['top'].set_visible(flag2)
        gca().spines['left'].set_visible(flag1)
        gca().spines['bottom'].set_visible(flag1)
        #gca().spines['bottom'].set_position('zero')
        #gca().spines['left'].set_smart_bounds(True)
        #gca().spines['bottom'].set_smart_bounds(True)
        gca().xaxis.set_ticks_position('bottom')
        gca().yaxis.set_ticks_position('left')
        if not flag1 and not flag2:
            tick_params(
                axis='x',
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off')
            tick_params(
                axis='y',
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off')
            gca().get_xaxis().set_ticks([])
            gca().get_yaxis().set_ticks([])
    figure(figsize=(9, 7))
    opt1 = load('data/world2-opt.pickle')
    opt2 = load('data/world3-opt.pickle')
    subplot(1, 2, 1)
    setup(False, False)
    ylim(0, 2.)
    xlim(-0.4, 0.3)
    xlabel('\n\n(a)')
    for w, color in [('world2', LIGHT_ORANGE), ('world3', LIGHT_BLUE), ('world0', LIGHT_GRAY)]:
        dataset = datasets[w]
        for data in dataset[::2]:
            plot(data['xh'][:, 0]+(0.13 if w=='world0' else 0.), data['xh'][:, 1], color=color, linewidth=0.7)#, linestyle='--' if w=='world2' else '-')
    plot(opt1['xh'][:, 0], opt1['xh'][:, 1], color=PURPLE, linewidth=1)#, linestyle='--' if w=='world2' else '-')
    plot(opt2['xh'][:, 0], opt2['xh'][:, 1], color=PURPLE, linewidth=1)#, linestyle='--' if w=='world2' else '-')
    subplot(1, 2, 2)
    setup(True)
    ylim(0, 2.)
    xlim(-0.3, 0.3)
    xlabel('x\n(b)')
    ylabel('y')
    for w, col1, col2 in [('world2', DARK_ORANGE, LIGHT_ORANGE), ('world3', DARK_BLUE, LIGHT_BLUE), ('world0', DARK_GRAY, LIGHT_GRAY)]:
        dataset = datasets[w]
        x = np.stack(extend(data['xh'][:, 0], T) for data in dataset)
        if w=='world0':
            x = x+0.13
        y = np.stack(extend(data['xh'][:, 1], T) for data in dataset)
        mx = nanmean(x, axis=0)
        my = nanmean(y, axis=0)
        s = nanstd(x, axis=0)
        n = (~isnan(mx)).sum(axis=0)
        s = s/sqrt(n)
        fill_betweenx(my, mx-s, mx+s, color=col2)
        plots[w], = plot(mx, my, color=col1, linewidth=3.)
    plots['opt'], = plot(opt1['xh'][:, 0], opt1['xh'][:, 1], color=PURPLE, linewidth=3)#, linestyle='--' if w=='world2' else '-')
    plot(opt2['xh'][:, 0], opt2['xh'][:, 1], color=PURPLE, linewidth=3)#, linestyle='--' if w=='world2' else '-')
    figlegend((plots['opt'], plots['world0'], plots['world2'], plots['world3']), ('Learned Human Model', 'Avoid Human', 'Affect Human (Left)', 'Affect Human (Right)'), 'upper center', ncol=4, fontsize=10)
    savefig('plots/plot23.pdf')




def plot01():
    T = dt*35
    def setup(flag1=True, flag2=False):
        gca().spines['right'].set_visible(flag2)
        gca().spines['top'].set_visible(flag2)
        gca().spines['left'].set_visible(flag1)
        gca().spines['bottom'].set_visible(flag1)
        #gca().spines['bottom'].set_position('zero')
        #gca().spines['left'].set_smart_bounds(True)
        #gca().spines['bottom'].set_smart_bounds(True)
        gca().xaxis.set_ticks_position('bottom')
        gca().yaxis.set_ticks_position('left')
        if not flag1 and not flag2:
            tick_params(
                axis='x',
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off')
            tick_params(
                axis='y',
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off')
            gca().get_xaxis().set_ticks([])
            gca().get_yaxis().set_ticks([])
        xlim(0., T)
    #opt = load('unique_data/world1-opt.pickle')
    opt = load('saved_data/world_kex.pickle')
    
    figure(figsize=(9, 7))
    subplot(2, 2, 1)
    xlabel('(a)')
    setup(False, False)
    ylabel('speed')
    for w, color in [('world_kex', LIGHT_GRAY), ('world_kex', LIGHT_ORANGE)]:
        dataset = datasets[w]
        for data in dataset[::2]:
            plot(data['t'], data['xh'][:, 3], color=color, linewidth=0.7)
    plot(opt['t'], opt['xh'][:, 3], color=PURPLE, linewidth=1.)
    subplot(2, 2, 3)
    setup()
    xlabel('time (s)\n(b)')
    ylabel('average speed')
    for w, col1, col2 in [('world_kex', DARK_GRAY, LIGHT_GRAY), ('world_kex', DARK_ORANGE, LIGHT_ORANGE)]:
        dataset = datasets[w]
        d = np.stack(extend(data['xh'][:, 3], T/dt+1) for data in dataset)
        m = nanmean(d, axis=0)
        t = arange(len(m))*dt
        s = nanstd(d, axis=0)
        n = (~isnan(d)).sum(axis=0)
        s = s/sqrt(n)
        fill_between(t, m-s, m+s, color=col2)
        plot(t, m, color=col1, linewidth=3)
    plot(opt['t'], opt['xh'][:, 3], color=PURPLE, linewidth=3.)
    subplot(2, 2, 4)
    setup()
    xlabel('time (s)\n(d)')
    ylabel('average latitude')
    plots = {}
    for w, col1, col2 in [('world_kex', DARK_GRAY, LIGHT_GRAY), ('world_kex', DARK_ORANGE, LIGHT_ORANGE)]:
        dataset = datasets[w]
        d = np.stack(extend(data['xh'][:, 1], T/dt+1) for data in dataset)
        m = nanmean(d, axis=0)
        t = arange(len(m))*dt
        s = nanstd(d, axis=0)
        n = (~isnan(d)).sum(axis=0)
        s = s/sqrt(n)
        fill_between(t, m-s, m+s, color=col2)
        plots[w], = plot(t, m, color=col1, linewidth=3)
    plots['opt'], = plot(opt['t'], opt['xh'][:, 1], color=PURPLE, linewidth=3.)
    subplot(2, 2, 2)
    setup(False, False)
    ylabel('latitude')
    xlabel('(c)')
    ylim(0., 3.)
    for w, color in [('world_kex', LIGHT_GRAY), ('world_kex', LIGHT_ORANGE)]:
        dataset = datasets[w]
        for data in dataset[::2]:
            plot(data['t'], data['xh'][:, 1], color=color, linewidth=0.7)
    plot(opt['t'], opt['xh'][:, 1], color=PURPLE, linewidth=1.)
    figlegend((plots['opt'], plots['world_kex'], plots['world_kex']), ('Learned Human Model', 'Avoid Human', 'Affect Human'), 'upper center', ncol=3)
    savefig('plots/world01.pdf', transparent=True)

def plotNumbers():
    def with_score(d, f):
        ret = dict(d)
        ret['score'] = f(d)
        return ret
    def measure1(data):
        L = 50
        return mean(cextend(data['xh'][:, 3], L)**2)

    f1 = open('csvs/dataI.csv', 'w')
    writer = csv.DictWriter(f1, extrasaction='ignore', fieldnames=[
        'user', 'traj', 'condition', 'score'
    ])
    writer.writeheader()
    writer.writerow(with_score(load('data/world1-opt.pickle'), measure1))
    for data in datasets['world0']:
        writer.writerow(with_score(data, measure1))
    for data in datasets['world1']:
        writer.writerow(with_score(data, measure1))

    print 'Scenario I (mean of speed^2 over 5 seconds)'
    a = mean(asarray([measure1(data) for data in datasets['world0']]))
    b = mean(asarray([measure1(data) for data in datasets['world1']]))
    c = measure1(load('data/world1-opt.pickle'))
    print 'World0 (Gray)', a
    print 'World1 (Orange)', b
    print 'Optimum (Purple)', c
    print '-'*10


    def measure2(data):
        L = 50
        return mean(cextend(data['xh'][:, 0], L))

    f2 = open('csvs/dataII.csv', 'w')
    writer = csv.DictWriter(f2, extrasaction='ignore', fieldnames=[
        'user', 'traj', 'condition', 'score'
    ])
    writer.writeheader()
    writer.writerow(with_score(load('data/world2-opt.pickle'), measure2))
    writer.writerow(with_score(load('data/world3-opt.pickle'), measure2))
    for data in datasets['world0']:
        writer.writerow(with_score(data, lambda x: measure2(x)+0.13))
    for data in datasets['world2']:
        writer.writerow(with_score(data, measure2))
    for data in datasets['world3']:
        writer.writerow(with_score(data, measure2))

    print 'Scenario II (mean of x over 5 seconds)'
    print 'World0 (Gray)', mean(asarray([measure2(data) for data in datasets['world0']]))+0.13
    print 'World2 (Orange)', mean(asarray([measure2(data) for data in datasets['world2']]))
    print 'Optimum left (Purple)', measure2(load('data/world2-opt.pickle'))
    print 'World3 (Blue)', mean(asarray([measure2(data) for data in datasets['world3']]))
    print 'Optimum right (Purple)', measure2(load('data/world3-opt.pickle'))
    print '-'*10



    def measure3(data):
        th = nonzero(data['xh'][:, 1]>0.)[0]
        if len(th)==0:
            return 0.
        tr = nonzero(data['xr'][:, 0]>0.)[0]
        if len(tr)==0:
            return 1.
        return 1. if th[0]<tr[0] else 0.

    f3 = open('csvs/dataIII.csv', 'w')
    writer = csv.DictWriter(f3, extrasaction='ignore', fieldnames=[
        'user', 'traj', 'condition', 'score'
    ])
    writer.writeheader()
    writer.writerow(with_score(load('data/world4-opt.pickle'), measure3))
    for data in datasets['world5']:
        writer.writerow(with_score(data, measure3))
    for data in datasets['world4']:
        writer.writerow(with_score(data, measure3))

    print 'Scenario III (mean of x over 5 seconds)'
    print 'World5 (Gray)', mean(asarray([measure3(data) for data in datasets['world5']]))
    print 'World4 (Orange)', mean(asarray([measure3(data) for data in datasets['world4']]))
    print 'Optimum (Purple)', measure3(load('data/world4-opt.pickle'))


def kex_plot_1(name = "", human_is_first=True):
    # black and white, works
    #COLOR1 = (0.8, 0.8, 0.8)
    #COLOR2 = (0., 0., 0.)

    COLOR_ROBOT = (0.  , 0.23, 0.68)  # Ataneo blue
    COLOR_HUMAN = (0.29, 1.00, 0.00)  # Chlorophyll green
    LIGHT_BLUE = COLOR_HUMAN
    LIGHT_ORANGE = COLOR_ROBOT
    
    T = dt*35
    def setup(flag1=True, flag2=True):
        # fix the axis
        gca().spines['right'].set_visible(flag2)
        gca().spines['top'].set_visible(flag2)
        gca().spines['left'].set_visible(flag1)
        gca().spines['bottom'].set_visible(flag1)
        gca().xaxis.set_ticks_position('bottom')
        gca().yaxis.set_ticks_position('left')

        if not flag1 and not flag2:
            tick_params(
                axis='x',
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off')
            tick_params(
                axis='y',
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off')
            gca().get_xaxis().set_ticks([])
            gca().get_yaxis().set_ticks([])
        xlim(0., T)
    #opt = load('saved_data/world_kex.pickle')


    #data = datasets['world_kex'][0]
    data = 0
    opt = 0
    datasets = 0


    opt = load('data/'+name+'.pickle', human_is_first)
    #print opt
    #exit()
    #print datasets
    #exit()
    data = opt

    # Specifies the position, name and labels of the first graph
    # Speed Graph
    figure(figsize=(9, 7))
    subplot(2, 2, 1)
    ylabel('Speed [m/s]')
    xlabel('Time [s]   (a)')
    setup()
    ylim(0., 10.)
    xlim(0., 3.)
    sp1 = data['xh']
    sp2 = data['xr']
    for i in range(len(sp1)):
        sp1[i] = sp1[i]/0.13
        sp2[i] = sp2[i]/0.13

    s1 = plot(data['t'], data['xh'][:, 3], color=LIGHT_BLUE, linewidth=1.)
    s2 = plot(data['t'], data['xr'][:, 3], color=LIGHT_ORANGE, linewidth=1.)
    follow_match = matplotlib.patches.Patch(color=LIGHT_BLUE, label='Human')
    leader_match = matplotlib.patches.Patch(color=LIGHT_ORANGE, label='Robot')
    matplotlib.pyplot.legend(handles=[follow_match, leader_match])
    
    # Graph for the x-pos
    subplot(2, 2, 2)
    setup()
    ylabel('X Position [m]')
    xlabel('Time [s]   (b)')
    ylim(0., 25.)
    xlim(0., 3.)
    x1 = plot(data['t'], data['xh'][:, 1], color=LIGHT_BLUE, linewidth=1.)
    x2 = plot(data['t'], data['xr'][:, 1], color=LIGHT_ORANGE, linewidth=1.)
    follow_match = matplotlib.patches.Patch(color=LIGHT_BLUE, label='Human')
    leader_match = matplotlib.patches.Patch(color=LIGHT_ORANGE, label='Robot')
    matplotlib.pyplot.legend(handles=[follow_match, leader_match])

    # Graph for the acceleration
    subplot(2, 2, 3)
    setup()
    ylabel('Acceleration [m/s^2]')
    xlabel('Time [s]   (c)')
    ylim(-20., 20.)
    xlim(0., 3.)
    speed_human = data['xh'][:, 3]
    speed_robot = data['xr'][:, 3]
    accel_human = [0.0]
    accel_robot = [0.0]
    # we take the speeds and 
    #print speed_human
    #exit()
    for i in range(1, len(speed_human)):
        accel_human.append((speed_human[i]-speed_human[i-1])*10)
        accel_robot.append((speed_robot[i]-speed_robot[i-1])*10)
        #print speed_human[i]
        #print accel_human[i]
    #print speed_human

    a1 = plot(data['t'], accel_human, color=LIGHT_BLUE, linewidth=1.)
    a2 = plot(data['t'], accel_robot, color=LIGHT_ORANGE, linewidth=1.)
    
    #a1 = plot(data['t'], data['xh'][:, 2], color=LIGHT_BLUE, linewidth=1.)
    #a2 = plot(data['t'], data['xr'][:, 2], color=LIGHT_ORANGE, linewidth=1.)
    follow_match = matplotlib.patches.Patch(color=LIGHT_BLUE, label='Human')
    leader_match = matplotlib.patches.Patch(color=LIGHT_ORANGE, label='Robot')
    matplotlib.pyplot.legend(handles=[follow_match, leader_match])
    
    
    # Graph for the y-pos
    subplot(2, 2, 4)
    setup()
    ylabel('Y Position [m]')
    xlabel('Time [s]   (d)')
    ylim(-2., 2.)
    xlim(0., 3.)
    y1 = plot(data['t'], data['xh'][:, 0], color=LIGHT_BLUE, linewidth=1.)
    y2 = plot(data['t'], data['xr'][:, 0], color=LIGHT_ORANGE, linewidth=1.)
    follow_match = matplotlib.patches.Patch(color=LIGHT_BLUE, label='Human')
    leader_match = matplotlib.patches.Patch(color=LIGHT_ORANGE, label='Robot')
    matplotlib.pyplot.legend(handles=[follow_match, leader_match])
        

    
    #savefig('plots/world_behind_human_follower.pdf', transparent=True)
    #savefig('plots/world_behind_human_leader.pdf', transparent=True)
    #savefig('plots/world_infront_human_follower.pdf', transparent=True)
    #savefig('plots/world_infront_human_leader.pdf', transparent=True)
    #savefig('plots/world_switch_human_follower.pdf', transparent=True)
    #savefig('plots/world_switch_human_leader.pdf', transparent=True)
    savefig('plots/'+name+'.pdf', transparent=True)
    
#plotAnimate()

#plotNumbers()
#show()
#plot01()
#plot23()
#plot45()


#kex_plot_1()

names = ['world_behind_human_follower',
'world_behind_human_leader',
'world_infront_human_follower',
'world_infront_human_leader',
'world_switch_human_follower',
'world_switch_human_leader']

human_is_first =[True, False, True, False, True, False]

load_location = 'data'
load_format = '.pickle'

save_location = 'plots'
save_format = '.pdf'

for i in range(len(names)):
    print "Fixing graph: ", names[i]
    kex_plot_1(names[i], human_is_first[i])

