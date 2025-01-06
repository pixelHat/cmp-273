from dataclasses import dataclass
import pyarrow.parquet as pq
import plotly.graph_objects as go
from plotly_resampler import FigureResampler


@dataclass
class StarPU:
    file_name: str

    def __init__(self, file_name: str, interval: float):
        self.file_name = file_name
        self.interval = interval
        self.starpu = pq.read_table(
            file_name,
            columns=["ResourceId", "Value", "Start", "End", "Duration"],
        ).to_pandas()

    def tasks_by_resource(self, resourceId: list[str]):
        return self.starpu[self.starpu["ResourceId"].isin(resourceId)]

    @property
    def resourcesId(self):
        return self.starpu["ResourceId"].unique()

    def chart(self):
        COLORS = [
            "#8dd3c7",
            "#ffffb3",
            "#bebada",
            "#fb8072",
            "#80b1d3",
            "#fdb462",
            "#b3de69",
            "#fccde5",
            "#d9d9d9",
            "#bc80bd",
        ]

        df = self.tasks_by_resource(self.resourcesId)
        df["Task"] = df["ResourceId"]
        task_names = df["Task"].unique()
        task_positions = {task: index for index, task in enumerate(task_names)}
        types_of_tasks = df["Value"].unique()
        # TODO: generate better colors
        task_colors = {
            task: COLORS[index] for (index, task) in enumerate(types_of_tasks)
        }

        # Create the figure
        fig = FigureResampler(go.Figure())

        legend_entries = set()

        for task in types_of_tasks:
            task_data = df[df["Value"] == task][df["Duration"] > self.interval]
            for _, row in task_data.iterrows():
                show_legend = task not in legend_entries
                fig.add_trace(
                    go.Bar(
                        y=[task_positions[row["Task"]]],
                        x=[row["End"] - row["Start"]],
                        orientation="h",
                        name=task,
                        legendgroup=task,
                        showlegend=show_legend,
                        base=row["Start"],
                        marker_color=task_colors[row["Value"]],
                    ),
                )

                if show_legend:
                    legend_entries.add(task)  # Add the task to the legend tracking set

        fig.update_layout(
            title="Task Timeline",
            xaxis_title="Time (milliseconds)",
            yaxis_title="Workers",
            barmode="relative",
            yaxis=dict(
                tickvals=list(range(len(df["ResourceId"].unique()))),
                ticktext=df["ResourceId"].unique(),
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.5
            ),
        )
        fig.update_yaxes(autorange="reversed")

        fig.update_xaxes(
            tickmode="linear",  # Set tick mode to linear
            tick0=0,  # Starting point
            dtick=10000,  # Interval between ticks
            tickformat=",",  # Format to show full numbers without abbreviations
        )

        return fig
