from IPython.core.display_functions import clear_output

from vadbp.variant_comparator import DataType
from ipywidgets import HTML, VBox, HBox, Output, Layout, SelectMultiple, Button, ToggleButtons, Tab
from ipyevents import Event
from IPython.display import display


class VisualVariantComparator:

    def __init__(self, varcop):
        self.varcop = varcop

        self.graph = HTML()
        self.info = Output()
        self.log = HTML()
        self.plot_tabs = Tab(layout={'display': 'block', 'width': '75%'})

        self.data_type_toggle = ToggleButtons(
            options=['Continuous', 'Categorical']
        )

        self.default_data_type = self.get_selected_data_type()

        self.measurements = SelectMultiple(
            options=varcop.get_columns_for_type(self.default_data_type),
            value=varcop.get_columns_for_type(self.default_data_type),
            rows=30,
            layout=Layout(max_width='150px')

        )
        self.select_all_measurements = Button(
            description='Select All',
            icon='check'
        )

        self.graph.value = self.varcop.visualize(attributes=self.measurements.value,
                                                 data_type=self.get_selected_data_type())

        self.events = Event(source=self.graph, watched_events=['click'],
                            watched_node_attributes=['data-type', 'data-id', 'data-id-left', 'data-id-right'])

        self.events.on_dom_event(self.handle_graph_click_event)
        self.data_type_toggle.observe(self.handle_data_type_update_event, names='value')
        self.measurements.observe(self.handle_measurements_update_event, names='value')
        self.select_all_measurements.on_click(self.handle_select_all_measurements)

        self.tool = HBox([VBox([self.data_type_toggle, self.select_all_measurements, self.measurements], layout=Layout(width='15%')),
                          VBox([self.info, self.plot_tabs, self.graph], layout=Layout(width='80%'))])

    def show(self):
        display(self.tool)

    def get_selected_data_type(self):
        if self.data_type_toggle.value == 'Categorical':
            return DataType.CATEGORICAL
        return DataType.CONTINUOUS

    #@info.capture(clear_output=True)
    def handle_graph_click_event(self, event):
        with self.info:
            clear_output()
        self.plot_tabs.layout.display = 'none'

        self.log.value = f"You clicked a {event}"
        df_to_display = None
        plots_to_display = []
        if event['data-type'] == 'node':
            df_to_display = self.varcop.dataframe_for_node(event['data-id'], attributes=self.measurements.value,
                                                           data_type=self.get_selected_data_type())
            plots_to_display = [self.varcop.plot_for_node_attribute(event['data-id'], attribute,
                                                                    data_type=self.get_selected_data_type())
                                for attribute in df_to_display.reset_index()['variable'].tolist()]
        if event['data-type'] == 'edge':
            df_to_display = self.varcop.dataframe_for_edge(event['data-id-left'], event['data-id-right'],
                                                           attributes=self.measurements.value,
                                                           data_type=self.get_selected_data_type())
            plots_to_display = [
                self.varcop.plot_for_edge_attribute(event['data-id-left'], event['data-id-right'], attribute,
                                                    data_type=self.get_selected_data_type())
                for
                attribute in df_to_display.reset_index()['variable'].tolist()]
        if df_to_display is not None:
            self.plot_tabs.layout.display = 'block'
            with self.info:
                display(df_to_display)
            outputs = [Output() for plot in plots_to_display]
            for i, plot in enumerate(plots_to_display):
                with outputs[i]:
                    display(plot)
            self.plot_tabs.children = outputs
            [self.plot_tabs.set_title(i, tab) for i, tab in enumerate(df_to_display.reset_index()['variable'].tolist())]

    #@info.capture(clear_output=True)
    def handle_data_type_update_event(self, change):
        with self.info:
            clear_output()
        self.plot_tabs.layout.display = 'none'

        new_data_type = self.get_selected_data_type()
        self.measurements.options = self.varcop.get_columns_for_type(new_data_type)
        self.measurements.value = self.varcop.get_columns_for_type(new_data_type)
        self.graph.value = self.varcop.visualize(attributes=self.measurements.value, data_type=new_data_type)

    #@info.capture(clear_output=True)
    def handle_measurements_update_event(self, change):
        with self.info:
            clear_output()
        self.plot_tabs.layout.display = 'none'

        self.graph.value = self.varcop.visualize(attributes=change['new'],
                                                 data_type=self.get_selected_data_type())

    def handle_select_all_measurements(self, event):
        self.measurements.value = self.measurements.options
