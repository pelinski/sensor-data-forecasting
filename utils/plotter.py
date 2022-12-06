import torch
from bokeh.plotting import figure
from bokeh.models.widgets import Panel, Tabs
from bokeh.embed import file_html
from bokeh.resources import CDN
import numpy as np


def get_html_plot(outputs, targets):
    """Plots model predictions and targets for a given batch in html

    Args:
        outputs (torch.Tensor): Model outputs with shape (batch_size, seq_length, out_size)
        targets (torch.Tensor): Model targets with shape (batch_size, seq_length, out_size)

    Returns:
        file_html (string): html string with the plot
    """

    num_sensors = outputs[0].shape[2]
    sensor_tabs = []

    for sensor in range(num_sensors):

        batch_tabs = []
        for batch_idx in range(len(outputs)):

            outputs_batch = outputs[batch_idx]
            targets_batch = targets[batch_idx]

            outputs_r = torch.reshape(outputs_batch[:, :, sensor], (
                outputs_batch.shape[0]*outputs_batch.shape[1], 1)).squeeze().detach().cpu().numpy()
            targets_r = torch.reshape(targets_batch[:, :, sensor], (
                targets_batch.shape[0]*targets_batch.shape[1], 1)).squeeze().detach().cpu().numpy()

            x = np.arange(0, len(outputs_r))

            _ = figure(plot_width=550, plot_height=300,
                       name="sensor "+str(sensor+1))
            _.xaxis.axis_label = "Frames elapsed"
            _.yaxis.axis_label = "Amplitude"
            _.line(x, outputs_r, color="blue", legend_label="Output")
            _.line(x, targets_r, color="red", legend_label="Targets")
            _.legend.location = "top_left"

            batch_tab = Panel(child=_, title=str(batch_idx+1))
            batch_tabs.append(batch_tab)

        sensor_tab = Panel(child=Tabs(tabs=batch_tabs),
                           title="Sensor "+str(sensor+1))
        sensor_tabs.append(sensor_tab)

    html = file_html(Tabs(tabs=sensor_tabs), CDN, "sensor-forecasting")

    return html
