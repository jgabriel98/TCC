import matplotlib.pyplot as plt
from  matplotlib.cbook import mplDeprecation

import warnings;
warnings.filterwarnings('ignore', category=mplDeprecation)

def plot_data(data_list: list, tick, labels=None, legends=None, verticalLineAt=None, colors=None, blocking=True):
    plt.figure(figsize=(18, 9))
    plt.style.use('ggplot')
    if type(data_list) is not list:
        data_list = [data_list]
    
    lines = []
    
    for i in range(len(data_list)):
        data = data_list[i]
        label = legends[i] if legends and i<len(legends) else None
        color = colors[i] if colors and i<len(colors) else None
        if hasattr(data, 'reshape'):
            lines.append(plt.plot(range(data.shape[0]), data.reshape(-1), c=color, alpha=0.6 ,label=label)[0])
            plt.xticks(range(0, data.shape[0], tick), labels, rotation=45)
        else:
            lines.append(plt.plot(range(len(data)), data, c=color, alpha=0.6, label=label)[0])
            plt.xticks(range(0, len(data), tick), labels, rotation=45)


    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Mid Price', fontsize=18)

    if legends:
        leg = plt.legend()
        leg.get_frame().set_alpha(0.4)
        line_dict = dict()
        for legline, origline in zip(leg.get_lines(), lines):
            legline.set_picker(5)  # 5 pts tolerance
            line_dict[legline] = origline
        
        def onpick(event):
            # on the pick event, find the orig line corresponding to the
            # legend proxy line, and toggle the visibility
            legline = event.artist
            origline = line_dict[legline]
            vis = not origline.get_visible()
            origline.set_visible(vis)
            # Change the alpha on the line in the legend so we can see what lines
            # have been toggled
            if vis:
                legline.set_alpha(1.0)
            else:
                legline.set_alpha(0.2)
            plt.gcf().canvas.draw()

        plt.connect('pick_event', onpick)
    if verticalLineAt:
        plt.axvline(verticalLineAt, linestyle='--', c='0.5')
    plt.show(block=blocking)