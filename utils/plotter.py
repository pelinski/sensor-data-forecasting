import torch
from bokeh.plotting import figure
from bokeh.models.widgets import Panel, Tabs
from bokeh.models import Legend
from bokeh.embed import file_html
from bokeh.resources import CDN
import numpy as np


def get_html_plot(inputs, outputs, targets):
    """Plots model predictions and targets for a given batch in html

    Args:
        outputs (torch.Tensor): Model outputs with shape (batch_size, seq_length, out_size)
        targets (torch.Tensor): Model targets with shape (batch_size, seq_length, out_size)

    Returns:
        file_html (string): html string with the plot
    """

    num_sensors = outputs.shape[2]
    num_plots = outputs.shape[0]

    sensor_tabs = []
    for sensor in range(num_sensors):

        seq_tabs = []
        for seq_idx in range(len(outputs)):
            x_in = np.arange(0, len(inputs[seq_idx, :, sensor]))
            x_out = np.arange(0, len(outputs[seq_idx, :, sensor]))
            _inputs = inputs[seq_idx, :,
                             sensor].flatten().detach().cpu().numpy()
            _outputs = outputs[seq_idx, :,
                               sensor].flatten().detach().cpu().numpy()
            _targets = targets[seq_idx, :,
                               sensor].flatten().detach().cpu().numpy()

            _ = figure(plot_width=550, plot_height=300,
                       name="sensor "+str(sensor+1))
            _.xaxis.axis_label = "Frames elapsed"
            _.yaxis.axis_label = "Amplitude"
            l_in = _.line(x_in, _inputs, color="green")
            l_out = _.line(x_in[-1]+x_out, _outputs,
                           color="blue")
            l_tgt = _.line(x_in[-1]+x_out, _targets,
                           color="red")
            legend = Legend(items=[("Input",   [l_in]),
                            ("Output", [l_out]), ("Target",   [l_tgt])])
            _.add_layout(legend, 'right')

            seq_tab = Panel(child=_, title=str(seq_idx+1))
            seq_tabs.append(seq_tab)

        sensor_tab = Panel(child=Tabs(tabs=seq_tabs),
                           title="Sensor "+str(sensor+1))
        sensor_tabs.append(sensor_tab)

    html = file_html(Tabs(tabs=sensor_tabs), CDN, "sensor-forecasting")

    return html
