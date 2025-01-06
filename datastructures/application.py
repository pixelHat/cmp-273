from dataclasses import dataclass
from functools import cached_property
import pyarrow.parquet as pq
import plotly.graph_objects as go


@dataclass
class Application:
    file_name: str
    highlighted: list[str]
    cpe: bool

    def __init__(
        self,
        file_name: str,
        dag_file_name: str,
        should_display_outliers: bool = False,
        show_idless_cpu: bool = False,
        show_abe: bool = False,
    ):
        self.file_name = file_name
        self.application = pq.read_table(file_name).to_pandas()
        self.dag = pq.read_table(
            dag_file_name, columns=["JobId", "Dependent"]
        ).to_pandas()
        self.total_time = max(self.application["End"])
        self.cpe = True
        self.should_display_outliers = should_display_outliers
        self.highlighted = []
        self.show_idless_cpu = show_idless_cpu
        self.show_abe = show_abe

    def tasks_by_resource(self, resourceId: list[str]):
        return self.application[self.application["ResourceId"].isin(resourceId)]

    def idelles_resource_time(self, resource_tasks):
        return round(
            (
                1
                - sum(
                    [
                        task["End"] - task["Start"]
                        for _, task in resource_tasks.iterrows()
                    ]
                )
                / self.total_time
            )
            * 100,
            2,
        )

    @property
    def resourcesId(self):
        return self.application["ResourceId"].unique()

    @property
    def chart(self):
        df = self.tasks_by_resource(self.resourcesId).sort_values(
            by="Value", ascending=True
        )
        df["Task"] = df["ResourceId"]
        task_names = df["Task"].unique()
        task_positions = {task: index for index, task in enumerate(task_names)}
        types_of_tasks = df["Value"].unique()
        task_colors = {
            "lapack_dgeqrt": "#8dd3c7",
            "lapack_dlarfb": "#ffffb3",
            "lapack_dtpqrt": "#bebada",
            "lapack_dtpmqrt": "#fb8072",
        }

        # Create the figure
        fig = go.Figure()

        legend_entries = set()

        bars = []
        for task in types_of_tasks:
            task_data = df[df["Value"] == task]
            for _, row in task_data.iterrows():
                show_legend = task not in legend_entries

                if self.should_display_outliers:
                    opacity = 1 if row["Outlier"] else 0.3
                else:
                    opacity = (
                        1
                        if len(self.highlighted) == 0
                        else (1 if row["JobId"] in self.highlighted else 0.3)
                    )

                bar = go.Bar(
                    y=[task_positions[row["Task"]]],
                    x=[row["Duration"]],
                    orientation="h",
                    name=task,
                    legendgroup=task,
                    showlegend=show_legend,
                    opacity=opacity,
                    base=row["Start"],
                    marker_color=task_colors[row["Value"]],
                    hovertext=row["Duration"],
                    customdata=[{"id": row["JobId"]}],
                    selected=dict(marker=dict(opacity=1)),
                    unselected=dict(marker=dict(opacity=1)),
                    uid=row["JobId"],
                )

                bars.append(bar)

                if show_legend:
                    legend_entries.add(task)  # Add the task to the legend tracking set
        fig.add_traces(bars)

        fig.update_yaxes(autorange="reversed")

        fig.update_xaxes(
            tickmode="linear",
            tick0=0,
            dtick=10000,
            tickformat=",",
        )

        # ABE
        if self.show_abe:
            fig.add_vrect(
                x0=self.abe,
                x1=self.abe,
                label=dict(
                    text=f"ABE {self.abe}",
                    font=dict(size=16, color="white"),
                    textangle=-90,
                ),
                line=dict(color="rgba(128, 128, 128, 0.8)", width=20),
            )

        # Idle CPU
        if self.show_idless_cpu:
            for index, resourceId in enumerate(self.resourcesId):
                fig.add_annotation(
                    text=f"{self.idelles_resource_time(self.tasks_by_resource([resourceId]))}%",
                    x=1,
                    y=index,
                    font=dict(color="black"),
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=2,
                    align="center",
                    showarrow=False,
                    xanchor="left",
                )
        fig.update_layout(
            title="Application",
            xaxis_title="Time (milliseconds)",
            yaxis_title="Workers",
            barmode="relative",
            yaxis=dict(
                tickvals=list(range(len(df["ResourceId"].unique()))),
                ticktext=df["ResourceId"].unique(),
            ),
            newshape_line_color="cyan",
            editrevision=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.1,
                xanchor="center",
                x=0.5,
            ),
            newshape=dict(
                fillcolor="rgba(0,0, 0,0.2)",
                label=dict(
                    text="A1",
                    textposition="top center",
                    yanchor="bottom",
                    texttemplate="A%{x0:d}",
                ),
            ),
        )
        return fig

    @cached_property
    def abe(self):
        return int(
            self.application["Duration"].mean()
            * self.application["Duration"].size
            / len(self.resourcesId),
        )

    def highlight(self, bars: list[str]):
        self.highlighted = list(bars)

    def highlight_task_depedency(self, id: str):
        if self.should_display_outliers:
            return
        kids = self.dag[self.dag["Dependent"] == id]
        self.highlighted = list([str(j) for j in kids["JobId"]]) + [id]
