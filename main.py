import streamlit as st

from datastructures import Application, StarPU, Scheduler

st.set_page_config(
    page_title="StarPU Runtime Performance Analisys",
    page_icon="📈",
    layout="wide",
)


def application_on_select(key: str, should_display_outliers=False):
    if should_display_outliers:
        del st.session_state[key]


def toggles_panel():
    should_display_abe = st.toggle("ABE", value=True)
    should_display_outliers = st.toggle("Outliers")
    should_display_cpu_idless = st.toggle("CPU Idleness")
    st.session_state["should_display_abe"] = should_display_abe
    st.session_state["should_display_outliers"] = should_display_outliers
    st.session_state["should_display_cpu_idless"] = should_display_cpu_idless


@st.fragment
def application_panel(file_name: str, dag_file_name: str, key: str):
    should_display_abe = st.session_state.get("should_display_abe", False)
    should_display_outliers = st.session_state.get("should_display_outliers", False)
    should_display_cpu_idless = st.session_state.get("should_display_cpu_idless", False)
    gantt = Application(
        file_name,
        dag_file_name,
        should_display_outliers=should_display_outliers,
        show_idless_cpu=should_display_cpu_idless,
        show_abe=should_display_abe,
    )

    click_data = st.session_state.get(key, None)
    if (
        click_data
        and len(click_data["selection"]["points"]) > 0
        and not should_display_outliers
    ):
        id = click_data["selection"]["points"][0]["customdata"]["id"]
        gantt.highlight_task_depedency(id)
    else:
        gantt.highlight([])

    st.plotly_chart(
        gantt.chart,
        use_container_width=True,
        on_select=lambda: application_on_select(key),
        selection_mode=["points"],
        key=key,
        config={
            "newshape_line_color": "red",
            "modeBarButtonsToAdd": [
                "drawline",
                "drawrect",
                "eraseshape",
            ],
        },
    )


@st.fragment
def starpu_panel(file_name: str, key: str):
    columns = st.columns(4, gap="medium")
    with columns[0]:
        interval = st.number_input(
            "Remove tasks with lifespan less than",
            value=1.0,
            format="%0.1f",
            key=f"number_input_{key}",
        )
    starpu_gantt = StarPU(file_name, interval)
    st.plotly_chart(starpu_gantt.chart(), use_container_width=True, key=key)


@st.fragment
def scheduler_panels(file_name: str, key_submitted: str, key_ready: str):
    scheduler = Scheduler(file_name)
    st.plotly_chart(scheduler.submitted_panel, key=key_submitted)
    st.plotly_chart(scheduler.ready_panel, key=key_ready)


options = st.multiselect(
    "Dataset",
    ["lws", "dmada", "dmdas"],
    ["dmdas"],
)

datasets = {
    "lws": "datasets/lws",
    "dmada": "datasets/dmada",
    "dmdas": "datasets/dmdas",
}


if len(options) > 0:
    toggles_panel()
    columns = st.columns((1) * len(options), gap="medium")

    for index, col in enumerate(columns):
        with col:
            folder = datasets[options[index]]
            application_panel(
                f"{folder}/application.parquet", f"{folder}/dag.parquet", f"st_{index}"
            )
            starpu_panel(f"{folder}/starpu.parquet", f"starpu_{index}")
            scheduler_panels(
                f"{folder}/variable.parquet",
                f"submitted_{index}",
                f"ready_{index}",
            )
else:
    st.write("Choose one dataset, please!")
