import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime

import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

from utils import load_fleet_excel, save_fleet_excel, update_vehicle_by_unit, COLUMNS

st.set_page_config(page_title="Fleet Dashboard", page_icon="🚛", layout="wide")

# Fleet type configuration
FLEET_CONFIG = {
    "OP": {"path": "data/fleet.xlsx", "name": "Open Pit", "icon": ""},
    "UG": {"path": "data/UG.xlsx", "name": "Underground", "icon": ""}
}
SHEET_NAME = "Fleet"

# Initialize fleet type in session state
if "fleet_type" not in st.session_state:
    st.session_state.fleet_type = "OP"

# Get current excel path based on fleet type
def get_excel_path():
    return FLEET_CONFIG[st.session_state.fleet_type]["path"]

# -------------------------
# Cache + helpers
# -------------------------
@st.cache_data(show_spinner=False)
def get_data(fleet_type: str):
    excel_path = FLEET_CONFIG[fleet_type]["path"]
    return load_fleet_excel(excel_path, sheet_name=SHEET_NAME)

def reload_data():
    st.cache_data.clear()

def to_float(x):
    try:
        return float(x)
    except Exception:
        return None

# numpy compatibility
try:
    _ = pd.np
except Exception:
    import numpy as np
    pd.np = np

def jitter_duplicates(df: pd.DataFrame, lat_col="LAT", lon_col="LON", meters=60):
    """
    Visual-only jitter for duplicated coordinates (does NOT touch Excel).
    """
    out = df.copy()
    out[lat_col] = out[lat_col].apply(to_float)
    out[lon_col] = out[lon_col].apply(to_float)
    out = out.dropna(subset=[lat_col, lon_col]).copy()

    dlat = meters / 111_320.0

    grp = out.groupby([lat_col, lon_col], dropna=True)
    for (lat, lon), g in grp:
        if len(g) <= 1:
            continue
        for j, idx in enumerate(g.index):
            angle = (j / len(g)) * 6.283185307  # 2*pi
            out.at[idx, lat_col] = lat + dlat * 0.75 * (pd.np.sin(angle))
            dlon = meters / (111_320.0 * max(0.2, abs(pd.np.cos(pd.np.radians(lat)))))
            out.at[idx, lon_col] = lon + dlon * 0.75 * (pd.np.cos(angle))
    return out

def safe_str(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v)

def format_year(v):
    """Format year values to remove .0 (e.g., 2018.0 -> 2018)"""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    try:
        # Try to convert to int if it's a number
        num = float(v)
        if not pd.isna(num):
            return str(int(num))
    except (ValueError, TypeError):
        pass
    return str(v)

# -------------------------
# Load data
# -------------------------
try:
    df = get_data(st.session_state.fleet_type)
except PermissionError:
    st.error(f"Permission denied: {get_excel_path()}")
    st.info("Close Excel (EXCEL.EXE) if the file is open, then rerun Streamlit.")
    st.stop()

# -------------------------
# Query param navigation (More >>)
# Clicking More>> in popup sends ?unit=F-124
# -------------------------
params = st.query_params
unit_from_url = params.get("unit", None)
if unit_from_url:
    # Force open details view
    st.session_state.selected_unit = unit_from_url
    st.session_state.page = "details"

# -------------------------
# Session defaults
# -------------------------
if "page" not in st.session_state:
    st.session_state.page = "map"
if "selected_unit" not in st.session_state:
    st.session_state.selected_unit = None

# -------------------------
# Sidebar navigation
# -------------------------
st.sidebar.title("🚚 Fleet Dashboard")

# Fleet Type Toggle Switch
st.sidebar.markdown("---")
st.sidebar.markdown("##### Fleet Type")

# Create toggle buttons for OP/UG
toggle_col1, toggle_col2 = st.sidebar.columns(2)

with toggle_col1:
    op_selected = st.session_state.fleet_type == "OP"
    if st.button(
        f" OP",
        use_container_width=True,
        type="primary" if op_selected else "secondary",
        key="btn_op"
    ):
        if st.session_state.fleet_type != "OP":
            st.session_state.fleet_type = "OP"
            st.session_state.selected_unit = None
            st.session_state.page = "map"
            if "map_selected_unit" in st.session_state:
                del st.session_state.map_selected_unit
            st.rerun()

with toggle_col2:
    ug_selected = st.session_state.fleet_type == "UG"
    if st.button(
        f" UG",
        use_container_width=True,
        type="primary" if ug_selected else "secondary",
        key="btn_ug"
    ):
        if st.session_state.fleet_type != "UG":
            st.session_state.fleet_type = "UG"
            st.session_state.selected_unit = None
            st.session_state.page = "map"
            if "map_selected_unit" in st.session_state:
                del st.session_state.map_selected_unit
            st.rerun()

# Show current fleet info
current_fleet = FLEET_CONFIG[st.session_state.fleet_type]
st.sidebar.caption(f"📂 {current_fleet['name']} Fleet")

# Determine the correct radio index based on current page
# details and edit_one pages should keep Map selected
current_page = st.session_state.page
if current_page in ["map", "details", "edit_one"]:
    nav_index = 0
elif current_page == "stats":
    nav_index = 1
elif current_page == "add":
    nav_index = 2
elif current_page == "download":
    nav_index = 3
else:
    nav_index = 2

nav = st.sidebar.radio(
    "Menu",
    ["🌍 Map", "📊 Stats", "➕ Add Vehicle", "📥 Download Fleet"],
    index=nav_index
)

# Only change page if user clicks a different menu item
if nav == "🌍 Map" and st.session_state.page not in ["map", "details", "edit_one"]:
    st.session_state.page = "map"
elif nav == "🌍 Map" and st.session_state.page in ["details", "edit_one"]:
    pass  # Keep current page (details or edit)
elif nav == "🌍 Map":
    st.session_state.page = "map"
elif nav == "📊 Stats":
    st.session_state.page = "stats"
elif nav == "➕ Add Vehicle":
    st.session_state.page = "add"
elif nav == "📥 Download Fleet":
    st.session_state.page = "download"

# -------------------------
# Filters (only for Map page)
# -------------------------
if st.session_state.page in ["map", "details", "edit_one"]:
    st.sidebar.markdown("---")
    st.sidebar.caption("Filters")

    regions = sorted([x for x in df["Region"].dropna().unique()])
    countries = sorted([x for x in df["Country"].dropna().unique()])
    statuses = sorted([x for x in df["Status"].dropna().unique()])
    vehicle_types = sorted([x for x in df["Type"].dropna().unique()])

    f_region = st.sidebar.multiselect("Region", regions)
    f_country = st.sidebar.multiselect("Country", countries)
    f_status = st.sidebar.multiselect("Status", statuses)
    f_type = st.sidebar.multiselect("Type", vehicle_types)
else:
    f_region = []
    f_country = []
    f_status = []
    f_type = []

# Quick search for vehicle
st.sidebar.markdown("---")
st.sidebar.caption("Quick Vehicle Search")
quick_search = st.sidebar.text_input("Enter vehicle Unit", "", key="quick_search")
if st.sidebar.button("🔍 Search Vehicle", use_container_width=True):
    if quick_search.strip():
        # Find exact or partial match
        search_results = df[df["Unit"].astype(str).str.lower().str.contains(quick_search.strip().lower(), na=False)]
        if not search_results.empty:
            st.session_state.selected_unit = str(search_results.iloc[0]["Unit"])
            st.session_state.page = "details"
            st.rerun()
        else:
            st.sidebar.error("Vehicle not found")
    else:
        st.sidebar.warning("Enter a vehicle Unit")

filtered = df.copy()
if f_region:
    filtered = filtered[filtered["Region"].isin(f_region)]
if f_country:
    filtered = filtered[filtered["Country"].isin(f_country)]
if f_status:
    filtered = filtered[filtered["Status"].isin(f_status)]
if f_type:
    filtered = filtered[filtered["Type"].isin(f_type)]

# KPI row
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Vehicles", len(filtered))
k2.metric("Countries", int(filtered["Country"].nunique()))
k3.metric("Regions", int(filtered["Region"].nunique()))
k4.metric("Operative", int((filtered["Status"].astype(str).str.lower() == "operative").sum()))
st.markdown("---")

# =========================
# PAGE: MAP
# =========================
if st.session_state.page == "map":
    fleet_icon = FLEET_CONFIG[st.session_state.fleet_type]["icon"]
    fleet_name = FLEET_CONFIG[st.session_state.fleet_type]["name"]
    st.title(f"{fleet_icon} {fleet_name} Fleet Map")

    map_df = filtered.copy()
    map_df["LAT"] = map_df["LAT"].apply(to_float)
    map_df["LON"] = map_df["LON"].apply(to_float)
    map_df = map_df.dropna(subset=["LAT", "LON"]).copy()

    if map_df.empty:
        st.warning("No vehicles have LAT/LON. Add coordinates to see pins.")
        st.dataframe(filtered.head(30), use_container_width=True)
        st.stop()

    # Jitter duplicates for clickability
    map_df = jitter_duplicates(map_df, meters=60)

    center_lat = float(map_df["LAT"].mean())
    center_lon = float(map_df["LON"].mean())

    m = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=2, 
        tiles="CartoDB positron",
        attr=""  # Remove attribution/credits
    )
    cluster = MarkerCluster(spiderfyOnMaxZoom=True, showCoverageOnHover=False, disableClusteringAtZoom=10)
    cluster.add_to(m)

    # Create simple popup with vehicle info
    for _, r in map_df.iterrows():
        unit = safe_str(r.get("Unit"))
        year = safe_str(r.get("Body Year"))
        condition = safe_str(r.get("Condition"))
        project = safe_str(r.get("Key Account"))
        v_type = safe_str(r.get("Type"))
        status = safe_str(r.get("Status")).lower()
        
        # Determine marker color based on status
        if status == "operative":
            marker_color = "green"
            status_text = "Operative"
        elif status in ["", "unknown", "other"]:
            marker_color = "orange"
            status_text = "Other"
        else:
            marker_color = "red"
            status_text = "Non-Operative"

        # Format year properly
        year_formatted = format_year(year)
        
        # Simple popup - no buttons, just info
        popup_html = f"""
        <div style="width:200px; font-family:Arial;">
          <b style="font-size:14px;">{unit}</b><br>
          <span style="color:{marker_color};">● {status_text}</span><br>
          <small>Type: {v_type} | Year: {year_formatted}</small><br>
          <small>Project: {project}</small>
        </div>
        """

        folium.Marker(
            location=[r["LAT"], r["LON"]],
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=unit,
            icon=folium.Icon(color=marker_color, icon="truck", prefix="fa")
        ).add_to(cluster)

    # Inject CSS directly into map to hide attribution
    hide_attribution_css = """
    <style>
    .leaflet-control-attribution {
        display: none !important;
    }
    </style>
    """
    m.get_root().html.add_child(folium.Element(hide_attribution_css))

    # Display map WITHOUT triggering reruns on interactions
    map_output = st_folium(
        m, 
        width=None, 
        height=500,
        returned_objects=["last_object_clicked_tooltip"]
    )

    # Check if a marker was clicked - update selection
    vehicle_list = sorted(map_df["Unit"].astype(str).unique().tolist())
    
    if map_output and map_output.get("last_object_clicked_tooltip"):
        clicked_unit = map_output["last_object_clicked_tooltip"]
        if clicked_unit in vehicle_list:
            # Only rerun if the selection actually changed
            if st.session_state.get("map_selected_unit") != clicked_unit:
                st.session_state.map_selected_unit = clicked_unit
                st.rerun()
    
    st.markdown("---")
    
    # Check if filters are active (region or country)
    filters_active = bool(f_region) or bool(f_country)
    
    if filters_active:
        # Show list of all filtered vehicles
        st.subheader(f"📋 Filtered Vehicles ({len(map_df)} found)")
        
        # Create a display dataframe with selected columns
        display_df = map_df.copy()
        display_df["Status"] = display_df["Status"].astype(str)
        display_df["Body Year_fmt"] = display_df["Body Year"].apply(format_year)
        display_df["Type_fmt"] = display_df["Type"].astype(str)
        display_df["Country_fmt"] = display_df["Country"].astype(str)
        display_df["Project"] = display_df["Key Account"].astype(str)
        display_df["Condition_fmt"] = display_df["Condition"].astype(str)
        
        # Select and reorder columns for display
        display_table = display_df[["Unit", "Status", "Body Year_fmt", "Type_fmt", "Country_fmt", "Project", "Condition_fmt"]]
        display_table.columns = ["Unit", "Status", "Body Year", "Type", "Country", "Project", "Condition"]
        
        # Color code by status
        def highlight_status(row):
            if str(row['Status']).lower() == 'operative':
                return ['background-color: #d4edda'] * len(row)
            else:
                return ['background-color: #f8d7da'] * len(row)
        
        st.dataframe(
            display_table.style.apply(highlight_status, axis=1),
            use_container_width=True,
            height=400
        )
    
    else:
        # Show FULL vehicle details below if a vehicle is clicked (no filters active)
        if "map_selected_unit" in st.session_state and st.session_state.map_selected_unit:
            selected_vehicle = st.session_state.map_selected_unit
            detail_row = df[df["Unit"].astype(str) == selected_vehicle]
            if not detail_row.empty:
                r = detail_row.iloc[0]
                
                st.subheader(f"📋 Vehicle Details: {selected_vehicle}")
                
                # Status indicator
                status_val = safe_str(r.get("Status")).lower()
                if status_val == "operative":
                    st.success("● Status: Operative")
                else:
                    st.error("● Status: Non-Operative")
                
                # Details in columns
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**📍 Location**")
                    st.write(f"**Region:** {safe_str(r.get('Region'))}")
                    st.write(f"**Country:** {safe_str(r.get('Country'))}")
                    st.write(f"**Coordinates:** {safe_str(r.get('LAT'))}, {safe_str(r.get('LON'))}")
                
                with c2:
                    st.markdown("**🚛 Vehicle Info**")
                    st.write(f"**Unit:** {safe_str(r.get('Unit'))}")
                    st.write(f"**Type:** {safe_str(r.get('Type'))}")
                    st.write(f"**Product:** {safe_str(r.get('Product'))}")
                    st.write(f"**Capacity:** {safe_str(r.get('Capacity (MT)'))} MT")
                
                with c3:
                    st.markdown("**📅 Details**")
                    st.write(f"**Chassis Year:** {format_year(r.get('Chassis Year'))}")
                    st.write(f"**Body Year:** {format_year(r.get('Body Year'))}")
                    st.write(f"**Condition:** {safe_str(r.get('Condition'))}")
                    st.write(f"**Axles:** {safe_str(r.get('Axles'))}")
                
                st.markdown("---")
                
                d1, d2 = st.columns(2)
                with d1:
                    st.markdown("**🏭 Operation**")
                    st.write(f"**Project:** {safe_str(r.get('Key Account'))}")
                    st.write(f"**Operation:** {safe_str(r.get('Operation'))}")
                
                with d2:
                    st.markdown("**⚙️ Engine**")
                    st.write(f"**Engine Type:** {safe_str(r.get('Engine Type'))}")
                    st.write(f"**Engine Number:** {safe_str(r.get('Engine Number'))}")
                    st.write(f"**Chassis:** {safe_str(r.get('Chassis'))}")
                
                # Edit button
                st.markdown("---")
                if st.button("✏️ Edit Vehicle", use_container_width=True, type="primary"):
                    st.session_state.selected_unit = selected_vehicle
                    st.session_state.page = "edit_one"
                    st.rerun()

# =========================
# PAGE: DETAILS (read-only) + Edit button
# =========================
if st.session_state.page == "details":
    unit = st.session_state.selected_unit
    st.title(f"Vehicle Details: {unit}")

    if not unit:
        st.info("No vehicle selected. Go to Map and click More >>.")
        st.stop()

    row_df = df[df["Unit"].astype(str) == str(unit)]
    if row_df.empty:
        st.warning("Vehicle not found in database.")
        st.query_params.clear()
        st.session_state.selected_unit = None
        st.session_state.page = "map"
        st.rerun()

    r = row_df.iloc[0]

    # Action buttons
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 4])
    with btn_col1:
        if st.button("⬅ Back to Map", use_container_width=True):
            st.query_params.clear()
            st.session_state.page = "map"
            st.rerun()
    with btn_col2:
        if st.button("✏️ Edit Vehicle", use_container_width=True, type="primary"):
            st.session_state.page = "edit_one"
            st.rerun()

    st.markdown("---")

    # Same layout as Add Vehicle form
    c1, c2, c3 = st.columns(3)
    with c1:
        st.text_input("Region", value=safe_str(r["Region"]), disabled=True, key="d_region")
        st.text_input("Country", value=safe_str(r["Country"]), disabled=True, key="d_country")
        st.text_input("Unit", value=safe_str(r["Unit"]), disabled=True, key="d_unit")
        st.text_input("Key Account", value=safe_str(r["Key Account"]), disabled=True, key="d_keyaccount")
    with c2:
        st.text_input("Type", value=safe_str(r["Type"]), disabled=True, key="d_type")
        st.text_input("Product", value=safe_str(r["Product"]), disabled=True, key="d_product")
        st.text_input("Status", value=safe_str(r["Status"]), disabled=True, key="d_status")
        st.text_input("Condition", value=safe_str(r["Condition"]), disabled=True, key="d_condition")
    with c3:
        st.text_input("Capacity (MT)", value=safe_str(r["Capacity (MT)"]), disabled=True, key="d_capacity")
        st.text_input("Chassis Year", value=safe_str(r["Chassis Year"]), disabled=True, key="d_yearchassis")
        st.text_input("Body Year", value=safe_str(r["Body Year"]), disabled=True, key="d_yearbody")
        st.text_input("Axles", value=safe_str(r["Axles"]), disabled=True, key="d_axles")

    st.markdown("### Operation & Mechanical")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.text_input("Operation", value=safe_str(r["Operation"]), disabled=True, key="d_operation")
    with d2:
        st.text_input("Engine Type", value=safe_str(r["Engine Type"]), disabled=True, key="d_enginetype")
        st.text_input("Engine Number", value=safe_str(r["Engine Number"]), disabled=True, key="d_enginenumber")
    with d3:
        st.text_input("Chassis", value=safe_str(r["Chassis"]), disabled=True, key="d_chassis")

    st.markdown("### Coordinates")
    e1, e2 = st.columns(2)
    with e1:
        st.text_input("LAT", value=safe_str(r["LAT"]), disabled=True, key="d_lat")
    with e2:
        st.text_input("LON", value=safe_str(r["LON"]), disabled=True, key="d_lon")

# =========================
# PAGE: EDIT ONE (save back to Excel)
# =========================
if st.session_state.page == "edit_one":
    unit = st.session_state.selected_unit
    st.title(f"✏️ Edit Vehicle: {unit}")

    if not unit:
        st.info("No vehicle selected.")
        st.stop()

    row_df = df[df["Unit"].astype(str) == str(unit)]
    if row_df.empty:
        st.warning("Vehicle not found.")
        st.stop()

    r = row_df.iloc[0]

    st.warning("Close Excel before saving, otherwise PermissionError.")

    # Action buttons at top
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 4])
    with btn_col1:
        back_clicked = st.button("⬅ Cancel", use_container_width=True)
    
    st.markdown("---")

    with st.form("edit_vehicle_form"):
        # Same layout as Add Vehicle form
        c1, c2, c3 = st.columns(3)
        with c1:
            REGION = st.text_input("Region", value=safe_str(r["Region"]))
            COUNTRY = st.text_input("Country", value=safe_str(r["Country"]))
            UNIT_DISP = st.text_input("Unit", value=safe_str(r["Unit"]), disabled=True, help="Unit cannot be changed")
            KEY_ACCOUNT = st.text_input("Key Account", value=safe_str(r["Key Account"]))
        with c2:
            TYPE = st.text_input("Type", value=safe_str(r["Type"]))
            PRODUCT = st.text_input("Product", value=safe_str(r["Product"]))
            Status = st.text_input("Status", value=safe_str(r["Status"]))
            Condition = st.text_input("Condition", value=safe_str(r["Condition"]))
        with c3:
            CAPACITY = st.text_input("Capacity (MT)", value=safe_str(r["Capacity (MT)"]))
            YEAR_CHASSIS = st.text_input("Chassis Year", value=safe_str(r["Chassis Year"]))
            YEAR_BODY = st.text_input("Body Year", value=safe_str(r["Body Year"]))
            Axles = st.text_input("Axles", value=safe_str(r["Axles"]))

        st.markdown("### Operation & Mechanical")
        d1, d2, d3 = st.columns(3)
        with d1:
            Operation = st.text_input("Operation", value=safe_str(r["Operation"]))
        with d2:
            Engine_Type = st.text_input("Engine Type", value=safe_str(r["Engine Type"]))
            Engine_Number = st.text_input("Engine Number", value=safe_str(r["Engine Number"]))
        with d3:
            Chassis = st.text_input("Chassis", value=safe_str(r["Chassis"]))

        st.markdown("### Coordinates")
        e1, e2 = st.columns(2)
        with e1:
            LAT = st.text_input("LAT", value=safe_str(r["LAT"]))
        with e2:
            LON = st.text_input("LON", value=safe_str(r["LON"]))

        save_btn = st.form_submit_button("💾 Save changes", type="primary", use_container_width=True)

    if back_clicked:
        st.session_state.page = "details"
        st.rerun()

    if save_btn:
        try:
            updates = {
                "Region": REGION,
                "Country": COUNTRY,
                "Key Account": KEY_ACCOUNT,
                "Type": TYPE,
                "Product": PRODUCT,
                "Status": Status,
                "Condition": Condition,
                "Capacity (MT)": CAPACITY,
                "Chassis Year": YEAR_CHASSIS,
                "Body Year": YEAR_BODY,
                "Operation": Operation,
                "Engine Type": Engine_Type,
                "Engine Number": Engine_Number,
                "Chassis": Chassis,
                "Axles": Axles,
                "LAT": LAT,
                "LON": LON,
            }

            df2 = update_vehicle_by_unit(df, unit, updates)
            save_fleet_excel(df2, get_excel_path(), sheet_name=SHEET_NAME)
            reload_data()

            st.success("Saved ✅ Updated in Excel.")
            st.session_state.page = "details"
            st.rerun()

        except PermissionError:
            st.error("Permission denied. Close Excel and try again.")
        except Exception as e:
            st.error(f"Error: {e}")

# =========================
# PAGE: STATS
# =========================
if st.session_state.page == "stats":
    fleet_icon = FLEET_CONFIG[st.session_state.fleet_type]["icon"]
    fleet_name = FLEET_CONFIG[st.session_state.fleet_type]["name"]
    st.title(f"📊 {fleet_name} Fleet Analytics")
    
    # Use all data for stats (no filters)
    stats_df = df.copy()
    
    # Prepare status classification
    stats_df["Status_Clean"] = stats_df["Status"].astype(str).str.lower()
    stats_df["Is_Operative"] = stats_df["Status_Clean"].apply(lambda x: "Operative" if x == "operative" else "Non-Operative")
    
    # First row: Country and Region charts (Interactive)
    st.markdown("### Fleet Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Vehicles by Country")
        country_counts = stats_df["Country"].fillna("Unknown").value_counts().head(15)
        if not country_counts.empty:
            fig1 = go.Figure(data=[go.Bar(
                x=country_counts.values,
                y=country_counts.index.astype(str),
                orientation='h',
                marker_color='#1f77b4',
                hovertemplate='<b>%{y}</b><br>%{x} vehicles<extra></extra>'
            )])
            fig1.update_layout(
                height=400, 
                showlegend=False, 
                yaxis={'categoryorder':'total ascending'},
                xaxis_title="Number of Vehicles", 
                yaxis_title="",
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=10, r=10, t=10, b=40),
                shapes=[dict(type="rect", xref="paper", yref="paper", x0=0, y0=0, x1=1, y1=1, line=dict(color="black", width=2))]
            )
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("No data")
    
    with col2:
        st.markdown("#### Vehicles by Region")
        region_counts = stats_df["Region"].fillna("Unknown").value_counts()
        if not region_counts.empty:
            fig2 = go.Figure(data=[go.Bar(
                x=region_counts.values,
                y=region_counts.index.astype(str),
                orientation='h',
                marker_color='#ff7f0e',
                hovertemplate='<b>%{y}</b><br>%{x} vehicles<extra></extra>'
            )])
            fig2.update_layout(
                height=400, 
                showlegend=False, 
                yaxis={'categoryorder':'total ascending'},
                xaxis_title="Number of Vehicles", 
                yaxis_title="",
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=10, r=10, t=10, b=40),
                shapes=[dict(type="rect", xref="paper", yref="paper", x0=0, y0=0, x1=1, y1=1, line=dict(color="black", width=2))]
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No data")
    
    st.markdown("---")
    
    # Second row: Three pie charts side by side with selectors above each
    st.markdown("### Operational Status Analysis")
    
    # Three columns for pie charts with selectors above each
    pie_col1, pie_col2, pie_col3 = st.columns(3)
    
    with pie_col1:
        st.markdown("#### Overall Fleet Status")
        # Add placeholder elements to match height with other columns
        st.write("")
        st.write("")
        st.write("")
        
        status_counts = stats_df["Is_Operative"].value_counts()
        
        if not status_counts.empty:
            # Colors: Dark red for Operative, Grey for Non-Operative
            colors_map = {'Operative': '#8B0000', 'Non-Operative': '#808080'}
            colors = [colors_map.get(s, '#808080') for s in status_counts.index]
            
            fig3 = go.Figure(data=[go.Pie(
                labels=status_counts.index,
                values=status_counts.values,
                marker=dict(colors=colors, line=dict(color='black', width=2)),
                textinfo='label+percent',
                hovertemplate='<b>%{label}</b><br>%{value} vehicles<br>%{percent}<extra></extra>'
            )])
            fig3.update_layout(
                height=380, 
                showlegend=True, 
                margin=dict(t=20, b=20, l=20, r=20),
                shapes=[dict(type="rect", xref="paper", yref="paper", x0=0, y0=0, x1=1, y1=1, line=dict(color="black", width=2))]
            )
            st.plotly_chart(fig3, use_container_width=True)
            st.caption(f"Total: {len(stats_df)} vehicles")
        else:
            st.info("No data")
    
    with pie_col2:
        st.markdown("#### Status by Selection")
        search_option = st.radio("By:", ["Country", "Project"], horizontal=True, key="status_filter", label_visibility="collapsed")
        if search_option == "Country":
            available_items = sorted(stats_df["Country"].dropna().unique().tolist())
            if available_items:
                selected_item = st.selectbox("Select", available_items, key="status_select", label_visibility="collapsed")
                filtered_stats = stats_df[stats_df["Country"] == selected_item]
            else:
                selected_item = "N/A"
                filtered_stats = pd.DataFrame()
        else:
            available_items = sorted(stats_df["Key Account"].dropna().unique().tolist())
            if available_items:
                selected_item = st.selectbox("Select", available_items, key="status_select", label_visibility="collapsed")
                filtered_stats = stats_df[stats_df["Key Account"] == selected_item]
            else:
                selected_item = "N/A"
                filtered_stats = pd.DataFrame()
        
        if not filtered_stats.empty:
            filtered_status = filtered_stats["Is_Operative"].value_counts()
            
            # Get vehicle names
            op_vehicles = filtered_stats[filtered_stats["Is_Operative"] == "Operative"]["Unit"].astype(str).tolist()
            nop_vehicles = filtered_stats[filtered_stats["Is_Operative"] == "Non-Operative"]["Unit"].astype(str).tolist()
            
            hover_texts2 = []
            for status in filtered_status.index:
                if status == "Operative":
                    vehicles = op_vehicles[:10]
                    extra = "..." if len(op_vehicles) > 10 else ""
                else:
                    vehicles = nop_vehicles[:10]
                    extra = "..." if len(nop_vehicles) > 10 else ""
                hover_texts2.append(f"<b>{status}</b><br>{filtered_status[status]} vehicles<br><br>Units: {', '.join(vehicles)}{extra}")
            
            colors_map = {'Operative': '#2ecc71', 'Non-Operative': '#e74c3c'}
            colors = [colors_map.get(s, '#808080') for s in filtered_status.index]
            
            fig4 = go.Figure(data=[go.Pie(
                labels=filtered_status.index,
                values=filtered_status.values,
                marker=dict(colors=colors, line=dict(color='black', width=2)),
                textinfo='label+percent',
                hovertemplate='%{customdata}<extra></extra>',
                customdata=hover_texts2
            )])
            fig4.update_layout(
                height=380, 
                showlegend=True, 
                margin=dict(t=20, b=20, l=20, r=20),
                shapes=[dict(type="rect", xref="paper", yref="paper", x0=0, y0=0, x1=1, y1=1, line=dict(color="black", width=2))]
            )
            st.plotly_chart(fig4, use_container_width=True)
            st.caption(f"{selected_item}: {len(filtered_stats)} vehicles")
        else:
            st.info("Select an option")
    
    with pie_col3:
        st.markdown("#### Products by Project")
        st.write("")  # Spacer for alignment
        available_projects = sorted(stats_df["Key Account"].dropna().unique().tolist())
        if available_projects:
            selected_project = st.selectbox("Select", available_projects, key="product_select", label_visibility="collapsed")
            filtered_products = stats_df[stats_df["Key Account"] == selected_project]
        else:
            selected_project = "N/A"
            filtered_products = pd.DataFrame()
        
        if not filtered_products.empty:
            product_counts = filtered_products["Product"].fillna("Unknown").value_counts()
            
            # Get vehicle names for each product
            hover_texts3 = []
            for product in product_counts.index:
                vehicles = filtered_products[filtered_products["Product"] == product]["Unit"].astype(str).tolist()[:10]
                extra = "..." if len(filtered_products[filtered_products["Product"] == product]) > 10 else ""
                hover_texts3.append(f"<b>{product}</b><br>{product_counts[product]} vehicles<br><br>Units: {', '.join(vehicles)}{extra}")
            
            fig5 = go.Figure(data=[go.Pie(
                labels=product_counts.index,
                values=product_counts.values,
                marker=dict(line=dict(color='black', width=2)),
                textinfo='label+percent',
                hovertemplate='%{customdata}<extra></extra>',
                customdata=hover_texts3
            )])
            fig5.update_layout(
                height=380, 
                showlegend=True, 
                margin=dict(t=20, b=20, l=20, r=20),
                shapes=[dict(type="rect", xref="paper", yref="paper", x0=0, y0=0, x1=1, y1=1, line=dict(color="black", width=2))]
            )
            st.plotly_chart(fig5, use_container_width=True)
            st.caption(f"{selected_project}: {len(filtered_products)} vehicles")
        else:
            st.info("Select a project")
    
    st.markdown("---")
    
    # Third row: Year of Issue Analysis
    st.markdown("### Year of Issue Analysis")
    
    year_col1, year_col2 = st.columns([1, 3])
    
    with year_col1:
        st.markdown("#### Filter Options")
        year_filter_option = st.radio("View by:", ["Country", "Project"], key="year_filter")
        
        if year_filter_option == "Country":
            year_available = sorted(stats_df["Country"].dropna().unique().tolist())
            if year_available:
                year_selected = st.selectbox("Select Country", year_available, key="year_select")
                year_filtered = stats_df[stats_df["Country"] == year_selected]
            else:
                year_selected = "N/A"
                year_filtered = pd.DataFrame()
        else:
            year_available = sorted(stats_df["Key Account"].dropna().unique().tolist())
            if year_available:
                year_selected = st.selectbox("Select Project", year_available, key="year_select")
                year_filtered = stats_df[stats_df["Key Account"] == year_selected]
            else:
                year_selected = "N/A"
                year_filtered = pd.DataFrame()
    
    with year_col2:
        st.markdown(f"#### Vehicle Years: {year_selected}")
        
        if not year_filtered.empty:
            # Convert YEAR BODY to numeric
            year_filtered = year_filtered.copy()
            year_filtered["YEAR_NUM"] = pd.to_numeric(year_filtered["Body Year"], errors='coerce')
            year_data = year_filtered.dropna(subset=["YEAR_NUM"])
            
            if not year_data.empty:
                # Group by year and count
                year_counts = year_data.groupby("YEAR_NUM").agg({
                    "Unit": lambda x: list(x.astype(str)),
                    "Key Account": lambda x: list(x.astype(str))
                }).reset_index()
                year_counts["Count"] = year_counts["Unit"].apply(len)
                year_counts = year_counts.sort_values("YEAR_NUM")
                
                # Create hover text
                if year_filter_option == "Country":
                    # Show project names when filtering by country
                    hover_texts_year = []
                    for _, row in year_counts.iterrows():
                        units = row["Unit"][:8]
                        projects = list(set(row["Key Account"]))[:5]
                        extra_units = "..." if len(row["Unit"]) > 8 else ""
                        extra_proj = "..." if len(set(row["Key Account"])) > 5 else ""
                        hover_texts_year.append(
                            f"<b>Year {int(row['YEAR_NUM'])}</b><br>"
                            f"{row['Count']} vehicles<br><br>"
                            f"Projects: {', '.join(projects)}{extra_proj}<br><br>"
                            f"Units: {', '.join(units)}{extra_units}"
                        )
                else:
                    hover_texts_year = []
                    for _, row in year_counts.iterrows():
                        units = row["Unit"][:8]
                        extra = "..." if len(row["Unit"]) > 8 else ""
                        hover_texts_year.append(
                            f"<b>Year {int(row['YEAR_NUM'])}</b><br>"
                            f"{row['Count']} vehicles<br><br>"
                            f"Units: {', '.join(units)}{extra}"
                        )
                
                # Get oldest and newest for annotation
                oldest_year = int(year_data["YEAR_NUM"].min())
                newest_year = int(year_data["YEAR_NUM"].max())
                oldest_vehicle = year_data[year_data["YEAR_NUM"] == oldest_year]["Unit"].iloc[0]
                newest_vehicle = year_data[year_data["YEAR_NUM"] == newest_year]["Unit"].iloc[0]
                
                fig6 = go.Figure(data=[go.Bar(
                    x=year_counts["YEAR_NUM"].astype(int).astype(str),
                    y=year_counts["Count"],
                    marker=dict(
                        color=year_counts["YEAR_NUM"],
                        colorscale='Blues',
                        line=dict(color='black', width=1)
                    ),
                    hovertemplate='%{customdata}<extra></extra>',
                    customdata=hover_texts_year
                )])
                
                fig6.update_layout(
                    height=380,
                    xaxis_title="Year",
                    yaxis_title="Number of Vehicles",
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    margin=dict(l=10, r=10, t=30, b=60),
                    shapes=[dict(type="rect", xref="paper", yref="paper", x0=0, y0=0, x1=1, y1=1, line=dict(color="black", width=2))],
                    annotations=[
                        dict(
                            text=f"📅 <b>Oldest:</b> {oldest_vehicle} ({oldest_year})  |  <b>Newest:</b> {newest_vehicle} ({newest_year})",
                            xref="paper", yref="paper",
                            x=0.5, y=-0.15,
                            showarrow=False,
                            font=dict(size=14),
                            xanchor='center'
                        )
                    ]
                )
                
                st.plotly_chart(fig6, use_container_width=True)
            else:
                st.info("No year data available")
        else:
            st.info("Select an option to view year distribution")

# =========================
# PAGE: ADD VEHICLE
# =========================
if st.session_state.page == "add":
    fleet_name = FLEET_CONFIG[st.session_state.fleet_type]["name"]
    st.title(f"➕ Add Vehicle to {fleet_name} Fleet")
    st.warning("Close Excel before saving.")

    with st.form("add_vehicle_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            REGION = st.text_input("Region")
            COUNTRY = st.text_input("Country")
            UNIT = st.text_input("Unit")
            KEY_ACCOUNT = st.text_input("Key Account")
        with c2:
            TYPE = st.text_input("Type")
            PRODUCT = st.text_input("Product")
            Status = st.text_input("Status")
            Condition = st.text_input("Condition")
        with c3:
            CAPACITY = st.text_input("Capacity (MT)")
            YEAR_CHASSIS = st.text_input("Chassis Year")
            YEAR_BODY = st.text_input("Body Year")
            Axles = st.text_input("Axles")

        st.markdown("### Operation & Mechanical")
        d1, d2, d3 = st.columns(3)
        with d1:
            Operation = st.text_input("Operation")
        with d2:
            Engine_Type = st.text_input("Engine Type")
            Engine_Number = st.text_input("Engine Number")
        with d3:
            Chassis = st.text_input("Chassis")

        st.markdown("### Coordinates")
        e1, e2 = st.columns(2)
        with e1:
            LAT = st.text_input("LAT")
        with e2:
            LON = st.text_input("LON")

        add_btn = st.form_submit_button("💾 Add vehicle")

    if add_btn:
        if not UNIT.strip() or not COUNTRY.strip():
            st.error("Unit and Country are required.")
        else:
            try:
                new_row = {
                    "Region": REGION.strip(),
                    "Country": COUNTRY.strip(),
                    "Unit": UNIT.strip(),
                    "Key Account": KEY_ACCOUNT.strip(),
                    "Type": TYPE.strip(),
                    "Product": PRODUCT.strip(),
                    "Status": Status.strip(),
                    "Condition": Condition.strip(),
                    "Capacity (MT)": CAPACITY.strip(),
                    "Chassis Year": YEAR_CHASSIS.strip(),
                    "Body Year": YEAR_BODY.strip(),
                    "Operation": Operation.strip(),
                    "Engine Type": Engine_Type.strip(),
                    "Engine Number": Engine_Number.strip(),
                    "Chassis": Chassis.strip(),
                    "Axles": Axles.strip(),
                    "LAT": LAT.strip(),
                    "LON": LON.strip(),
                }
                df2 = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                df2 = df2[COLUMNS]

                save_fleet_excel(df2, get_excel_path(), sheet_name=SHEET_NAME)
                reload_data()
                st.success("Added ✅ Saved to Excel.")
            except PermissionError:
                st.error("Permission denied. Close Excel and try again.")
            except Exception as e:
                st.error(f"Error: {e}")

# =========================
# PAGE: DOWNLOAD FLEET
# =========================
if st.session_state.page == "download":
    fleet_name = FLEET_CONFIG[st.session_state.fleet_type]["name"]
    st.title(f"📥 Download {fleet_name} Fleet Data")
    
    st.info(f"Total vehicles in **{fleet_name}** database: **{len(df)}**")
    
    # Show preview of data
    st.markdown("### Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown("---")
    
    # Convert dataframe to Excel in memory
    from io import BytesIO
    
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Fleet')
    excel_data = output.getvalue()
    
    fleet_prefix = st.session_state.fleet_type.lower()
    st.download_button(
        label=f"📥 Download Complete {fleet_name} Fleet (Excel)",
        data=excel_data,
        file_name=datetime.now().strftime(f"{fleet_prefix}_fleet_%d_%m_%Y.xlsx"),
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        type="primary"
    )
    
    st.caption(f"This will download all {fleet_name} fleet data including all columns and vehicles.")

