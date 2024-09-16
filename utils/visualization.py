from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
from utils import change_province, change_json
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import geopandas as gpd
import folium
from utils import machine_learning as ml


def interactive_table(df: pd.DataFrame):
    """Creates an st-aggrid interactive table based on a dataframe.

    Args:
        df (pd.DataFrame]): Source dataframe

    Returns:
        dict: The selected row
    """
    options = GridOptionsBuilder.from_dataframe(
        df,
        enableRowGroup=True,
        enableValue=True,
        enablePivot=True
    )

    options.configure_side_bar()

    options.configure_selection("single")
    selection = AgGrid(
        df,
        enable_enterprise_modules=True,
        gridOptions=options.build(),
        theme="light",
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=True,
    )

    return selection


def get_chart_line(data, x_label, y_label, z_label, xlabel, ylabel, title):
    lines = (
        alt.Chart(data,
                  title=title,
                  width=500,
                  height=300)
        .mark_line()
        .encode(
            x=alt.X(x_label, type="nominal", title=xlabel),
            y=alt.Y(y_label, type="quantitative", title=ylabel),
            color=alt.Color(z_label, type="nominal", title=""),
            order=alt.Order(z_label, sort="descending"),
            tooltip=[
                alt.Tooltip(x_label, title=x_label),
                alt.Tooltip(y_label, title=y_label)]
        )
    )

    return lines.interactive()


def get_bar_vertical(chart_data, x_label, y_label, z_label, xlabel, ylabel, title):
    # Horizontal stacked bar chart
    chart = (
        alt.Chart(chart_data,
                  title=title,
                  width=700,
                  height=400)
        .mark_bar()
        .encode(
            x=alt.X(x_label, type="nominal", title=xlabel),
            y=alt.Y(y_label, type="quantitative", title=ylabel),
            color=alt.Color(z_label, type="nominal", title=""),
            order=alt.Order(z_label, sort="descending"),
            tooltip=[
                alt.Tooltip(x_label, title=x_label),
                alt.Tooltip(y_label, title=y_label),
            ]
        )
    )

    return chart.interactive()


def get_bar_vertical_1(chart_datas, titles):
    chart_datas.reset_index(drop=True, inplace=True)

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax = chart_datas['value'].plot(kind='line',
                                   marker='*', color='black', ms=10)
    chart_datas['value'].plot(kind='bar', ax=ax,
                              xlim=ax.get_xlim(), ylim=ax.get_ylim())

    if len(chart_datas) == 2:
        ax.set_xticklabels(('2020', '2021'))
    elif len(chart_datas) == 3:
        ax.set_xticklabels(('2019', '2020', '2021'))
    elif len(chart_datas) == 4:
        ax.set_xticklabels(('2018', '2019', '2020', '2021'))
    elif len(chart_datas) == 5:
        ax.set_xticklabels(('2017', '2018', '2019', '2020', '2021'))

    ax.grid(axis='y')
    ax.set_title(titles)
    ax.set_xlabel('Years')
    ax.set_ylabel('Value')

    return fig, ax


def get_bar_vertical_2(chart_data, x_label, y_label, z_label, xlabel, ylabel, title):
    # Horizontal stacked bar chart
    chart = (
        alt.Chart(chart_data,
                  title=title,
                  width=500,
                  height=400)
        .mark_bar()
        .encode(
            x=alt.X(x_label, type="nominal", title=xlabel),
            y=alt.Y(y_label, type="quantitative", title=ylabel),
            color=alt.Color(z_label, type="nominal", title=""),
            order=alt.Order(z_label, sort="descending"),
            tooltip=[
                alt.Tooltip(x_label, title=x_label),
                alt.Tooltip(y_label, title=y_label),
            ]
        )
    )

    return chart.interactive()


def get_bar_horizontal(chart_data, x_label, y_label, z_label, xlabel, ylabel, title):
    # Horizontal stacked bar chart
    chart = (
        alt.Chart(chart_data,
                  title=title,
                  width=700)
        .mark_bar()
        .encode(
            x=alt.X(x_label, type="quantitative", title=xlabel),
            y=alt.Y(y_label, type="nominal", title=ylabel),
            color=alt.Color(z_label, type="nominal", title=""),
            order=alt.Order(z_label, sort="descending"),
            tooltip=[
                alt.Tooltip(x_label, title=x_label),
                alt.Tooltip(y_label, title=y_label),
            ]
        )
    )

    return chart.interactive()


def get_chart_map(dataset, target, title, source):
    file_geo = 'data/Indonesia/BATAS_PROVINSI_DESEMBER_2019_DUKCAPIL.shp'
    df_geo = gpd.read_file(file_geo)
    df_geo = change_province(df_geo)

    data = pd.merge(df_geo, dataset, on=["Provinsi"])

    # Create variables that will be used in some parameters later
    values = int(target)
    values2 = title

    # Create figure and axes for Matplotlib
    fig, ax = plt.subplots(1, figsize=(35, 10))

    # Set the value range for the choropleth map
    vmin, vmax = data[values].min(), data[values].max()

    # Remove the axis as we do not need it
    ax.axis('off')

    # Add labels
    data['coords'] = data['geometry'].apply(lambda x: x.representative_point().coords[:])
    data['coords'] = [coords[0] for coords in data['coords']]
    for idx, row in data.iterrows():
        ann = row['Provinsi']
        ann += '\n'
        ann += str(row[values])
        plt.annotate(text=ann,
                     fontsize=3,
                     xy=row['coords'],
                     horizontalalignment='center')

    # Add a map title
    title = '{}'.format(values2)
    ax.set_title(title, fontdict={'fontsize': '20',
                                  'fontweight': '10'})

    # Create an annotation for the data source
    ax.annotate(str('Source: ' + source), xy=(0.1, .08),
                xycoords='figure fraction',
                horizontalalignment='left',
                verticalalignment='top', fontsize=12, fontweight='bold', color='k')

    # Generate the map
    data.plot(column=values, cmap='Reds', linewidth=0.8, ax=ax, edgecolor='0.8',
              norm=plt.Normalize(vmin=vmin, vmax=vmax), legend=True)

    return fig, ax


def get_folium_map(dataset, target):
    file_geo = "data/Indonesia/BATAS_PROVINSI_DESEMBER_2019_DUKCAPIL.shp"
    df_geo = change_json(file_geo)
    df_geo = change_province(df_geo)

    df_merged = pd.merge(df_geo, dataset, on=["Provinsi"])

    # Create a map object for choropleth map
    map_indo = folium.Map(location=[-2.49607, 117.89587],
                          tiles='OpenStreetMap',
                          zoom_start=5)

    # Set up Choropleth map object with key on Province
    folium.Choropleth(geo_data=df_merged,
                      data=df_merged,
                      columns=['Provinsi', int(target)],
                      key_on='feature.properties.Provinsi',
                      fill_color='YlOrRd',
                      fill_opacity=1,
                      line_opacity=0.2,
                      legend_name='Rate',
                      smooth_factor=0,
                      Highlight=True,
                      line_color='#0000',
                      name='Rate',
                      show=True,
                      overlay=True).add_to(map_indo)

    # Add hover functionality
    # Style function
    style_function = lambda x: {'fillColor': '#ffffff', 'color': '#000000', 'fillOpacity': 0.1, 'weight': 0.1}

    # Highlight function
    highlight_function = lambda x: {'fillColor': '#000000', 'color': '#000000', 'fillOpacity': 0.50, 'weight': 0.1}

    # Create popup tooltip object
    NIL = folium.features.GeoJson(data=df_merged,
                                  style_function=style_function,
                                  control=False,
                                  highlight_function=highlight_function,
                                  tooltip=folium.features.GeoJsonTooltip(
                                      fields=['Provinsi', str(target)],
                                      aliases=['Provinsi', 'Value'],
                                      style=('background-color: white; '
                                             'color: #333333; font-family: arial; '
                                             'font-size: 12px; padding: 10px;')))

    # Add tooltip object to the map
    map_indo.add_child(NIL)
    map_indo.keep_in_front(NIL)

    # Add dark and light mode
    # folium.TileLayer('cartodbdark_matter',
    #                  name='dark mode',
    #                  control=True).add_to(map_indo)
    # folium.TileLayer('cartodbpositron',
    #                  name='light mode',
    #                  control=True).add_to(map_indo)

    # Add a layer controller
    folium.LayerControl(collapsed=False).add_to(map_indo)

    return map_indo


def cross_data(data, select1, select2, select3, title):
    chart = (
        alt.Chart(data,
                  title=title,
                  height=600,
                  width=700)
        .mark_point(filled=True)
        .encode(
            alt.X(select1),
            alt.Y(select2),
            alt.Size(select3),
            alt.Color(select3),
            alt.OpacityValue(1),
            tooltip=[alt.Tooltip('Provinsi'),
                     alt.Tooltip(select1),
                     alt.Tooltip(select2),
                     alt.Tooltip(select3)
                     ]
        ))

    return chart
