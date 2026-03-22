"""
Calgary Open Data — ML / DS Portfolio Index
Streamlit app that serves as a landing page for all 16 projects.
"""

import os
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from portfolio_config import (
    PROJECTS,
    CATEGORIES,
    SKILLS_MATRIX,
    TOOLS,
    DOMAINS,
    COLORS,
    GITHUB_BASE,
    GITHUB_USER,
)

# ── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Calgary ML/DS Portfolio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load CSS ────────────────────────────────────────────────────────────────
_css_path = os.path.join(os.path.dirname(__file__), "assets", "portfolio_style.css")
if os.path.exists(_css_path):
    with open(_css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ── Helpers ─────────────────────────────────────────────────────────────────
def _github_url(folder: str) -> str:
    return f"{GITHUB_BASE}/tree/master/{folder}"


def _category_badges(cats: list[str]) -> str:
    html = ""
    for cat in cats:
        info = CATEGORIES.get(cat, {"color": "#6c757d"})
        html += (
            f'<span class="category-badge" '
            f'style="background:{info["color"]}">{cat}</span>'
        )
    return html


def _tech_pills(stack: list[str]) -> str:
    return "".join(f'<span class="tech-pill">{t}</span>' for t in stack)


def _navigate_to_detail(project_number: int) -> None:
    """Set session state so the Details page opens with this project."""
    st.session_state["detail_project"] = project_number
    st.session_state["page"] = "Project Details"


# ── Sidebar navigation ─────────────────────────────────────────────────────
st.sidebar.title("Portfolio Navigator")
st.sidebar.markdown("---")

PAGE_OPTIONS = [
    "Home",
    "Projects Gallery",
    "Project Details",
    "Skills & Tech",
    "About",
]

# Allow session_state override (e.g. from Gallery → Details)
default_idx = 0
if "page" in st.session_state:
    stored = st.session_state["page"]
    if stored in PAGE_OPTIONS:
        default_idx = PAGE_OPTIONS.index(stored)

page = st.sidebar.radio("Navigate", PAGE_OPTIONS, index=default_idx)
st.session_state["page"] = page

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"[GitHub Repository]({GITHUB_BASE})",
)
st.sidebar.caption("Built on Calgary Open Data")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ═════════════════════════════════════════════════════════════════════════════
def page_home() -> None:
    # Hero
    st.markdown(
        """
        <div class="hero-section">
            <h1>Calgary Open Data ML/DS Portfolio</h1>
            <p>
                16 end-to-end machine learning and data science projects
                built entirely on City of Calgary Open Data, covering
                regression, classification, clustering, time series,
                NLP, anomaly detection &amp; more.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI row
    kpis = [
        ("16", "Projects"),
        ("1M+", "Data Records"),
        ("7", "ML Technique Areas"),
        ("16", "Interactive Apps"),
    ]
    cols = st.columns(4)
    for col, (val, label) in zip(cols, kpis):
        col.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-value">{val}</div>
                <div class="kpi-label">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")  # spacer

    # Category quick-nav
    st.markdown('<p class="section-header">Explore by Category</p>', unsafe_allow_html=True)
    badges_html = ""
    for name, info in CATEGORIES.items():
        count = sum(1 for p in PROJECTS if name in p["categories"])
        if count:
            badges_html += (
                f'<span class="cat-nav" style="background:{info["color"]}">'
                f'{info["icon"]} {name} ({count})</span>'
            )
    st.markdown(badges_html, unsafe_allow_html=True)

    st.markdown("")

    # Featured projects (pick 3 diverse ones)
    st.markdown('<p class="section-header">Featured Projects</p>', unsafe_allow_html=True)
    featured_indices = [0, 3, 9]  # Permit Cost, River Flow, Water Quality
    feat_cols = st.columns(3)
    for col, idx in zip(feat_cols, featured_indices):
        p = PROJECTS[idx]
        with col:
            st.markdown(
                f"""
                <div class="featured-card">
                    <h4>
                        <span class="project-number">{p["number"]}</span>
                        {p["title"]}
                    </h4>
                    <p>{p["tagline"]}</p>
                    {_category_badges(p["categories"])}
                    <div style="margin-top:0.6rem">{_tech_pills(p["tech_stack"][:4])}</div>
                    <div class="dataset-info">{p["dataset"]} &middot; {p["dataset_size"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("")

    # Tech ribbon
    st.markdown('<p class="section-header">Core Technology Stack</p>', unsafe_allow_html=True)
    all_tech = [
        "Python", "Streamlit", "Plotly", "scikit-learn", "XGBoost",
        "TensorFlow", "Prophet", "pandas", "NumPy", "Socrata API",
    ]
    ribbon_html = '<div class="tech-ribbon">'
    for t in all_tech:
        ribbon_html += f'<span class="tech-ribbon-item">{t}</span>'
    ribbon_html += "</div>"
    st.markdown(ribbon_html, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PROJECTS GALLERY
# ═════════════════════════════════════════════════════════════════════════════
def page_gallery() -> None:
    st.markdown('<p class="section-header">Projects Gallery</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">Browse, filter, and explore all 16 projects</p>',
        unsafe_allow_html=True,
    )

    # Sidebar filters
    st.sidebar.markdown("### Filters")
    all_cats = sorted({c for p in PROJECTS for c in p["categories"]})
    selected_cats = st.sidebar.multiselect("Category", all_cats, default=[])
    search = st.sidebar.text_input("Search projects", "")

    # Filter logic
    filtered = PROJECTS
    if selected_cats:
        filtered = [p for p in filtered if any(c in selected_cats for c in p["categories"])]
    if search:
        q = search.lower()
        filtered = [
            p for p in filtered
            if q in p["title"].lower()
            or q in p["tagline"].lower()
            or q in p["description"].lower()
            or any(q in t.lower() for t in p["tech_stack"])
        ]

    st.caption(f"Showing {len(filtered)} of {len(PROJECTS)} projects")

    if not filtered:
        st.info("No projects match the current filters.")
        return

    # 2-column card grid
    for i in range(0, len(filtered), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j >= len(filtered):
                break
            p = filtered[i + j]
            gh = _github_url(p["folder"])
            with col:
                st.markdown(
                    f"""
                    <div class="project-card">
                        <div>
                            <span class="project-number">{p["number"]}</span>
                            <h3>{p["title"]}</h3>
                        </div>
                        <p class="tagline">{p["tagline"]}</p>
                        {_category_badges(p["categories"])}
                        <div style="margin-top:0.5rem">{_tech_pills(p["tech_stack"])}</div>
                        <div class="dataset-info">
                            {p["dataset"]} &middot; {p["dataset_size"]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                btn_cols = st.columns([1, 1, 1])
                with btn_cols[0]:
                    if st.button("View Details", key=f"det_{p['number']}"):
                        _navigate_to_detail(p["number"])
                        st.rerun()
                with btn_cols[1]:
                    st.link_button("GitHub", gh)
                with btn_cols[2]:
                    if p["streamlit_url"]:
                        st.link_button("Launch App", p["streamlit_url"])
                    else:
                        st.button(
                            "Launch App",
                            key=f"launch_{p['number']}",
                            disabled=True,
                            help="Streamlit Cloud deployment coming soon",
                        )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PROJECT DETAILS
# ═════════════════════════════════════════════════════════════════════════════
def page_details() -> None:
    st.markdown('<p class="section-header">Project Details</p>', unsafe_allow_html=True)

    # Project selector
    titles = [f"{p['number']:02d} — {p['title']}" for p in PROJECTS]
    default_idx = 0
    if "detail_project" in st.session_state:
        target = st.session_state["detail_project"]
        for idx, p in enumerate(PROJECTS):
            if p["number"] == target:
                default_idx = idx
                break

    selected = st.sidebar.selectbox("Select project", titles, index=default_idx)
    proj_idx = titles.index(selected)
    p = PROJECTS[proj_idx]
    st.session_state["detail_project"] = p["number"]

    # Header
    st.markdown(
        f"""
        <div style="margin-bottom:1.2rem">
            <span class="project-number" style="font-size:1.1rem;width:2.4rem;
            height:2.4rem;line-height:2.4rem">{p["number"]}</span>
            <span style="font-size:1.5rem;font-weight:700;color:#1E3A5F;
            vertical-align:middle">{p["title"]}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Categories
    st.markdown(_category_badges(p["categories"]), unsafe_allow_html=True)
    st.markdown("")

    # Description
    st.markdown(p["description"])
    st.markdown("---")

    # Two-column layout
    left, right = st.columns([3, 2])

    with left:
        # Methodology
        st.markdown("#### Methodology")
        method_html = '<ul class="method-list">'
        for step in p["methodology"]:
            method_html += f"<li>{step}</li>"
        method_html += "</ul>"
        st.markdown(method_html, unsafe_allow_html=True)

        st.markdown("")

        # Tech stack table
        st.markdown("#### Technology Stack")
        table_html = '<table class="info-table">'
        table_html += f"<tr><td>Dataset</td><td>{p['dataset']}</td></tr>"
        table_html += f"<tr><td>Dataset Size</td><td>{p['dataset_size']}</td></tr>"
        table_html += f"<tr><td>Key Metric</td><td>{p['key_metric']}</td></tr>"
        table_html += (
            f"<tr><td>Libraries</td><td>{', '.join(p['tech_stack'])}</td></tr>"
        )
        table_html += "</table>"
        st.markdown(table_html, unsafe_allow_html=True)

    with right:
        # App pages
        st.markdown("#### App Pages")
        pages_html = ""
        for pg in p["pages"]:
            pages_html += f'<span class="page-chip">{pg}</span>'
        st.markdown(pages_html, unsafe_allow_html=True)

        st.markdown("")

        # Action buttons
        st.markdown("#### Links")
        gh = _github_url(p["folder"])
        st.link_button("View on GitHub", gh, use_container_width=True)

        if p["streamlit_url"]:
            st.link_button(
                "Launch Live App", p["streamlit_url"], use_container_width=True
            )
        else:
            st.button(
                "Launch Live App",
                disabled=True,
                use_container_width=True,
                help="Streamlit Cloud deployment coming soon",
            )

        # Local run instructions
        with st.expander("Run locally"):
            st.code(
                f"cd {p['folder']}\n"
                f"pip install -r requirements.txt\n"
                f"streamlit run app.py",
                language="bash",
            )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SKILLS & TECH
# ═════════════════════════════════════════════════════════════════════════════
def page_skills() -> None:
    st.markdown('<p class="section-header">Skills & Technology</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">Techniques, tools, and domain expertise across the portfolio</p>',
        unsafe_allow_html=True,
    )

    # ── Heatmap: techniques × projects ──────────────────────────────────────
    st.markdown("#### ML Techniques × Projects")
    techniques = list(SKILLS_MATRIX.keys())
    project_labels = [f"P{p['number']:02d}" for p in PROJECTS]

    z = []
    for tech in techniques:
        row = [1 if p["number"] in SKILLS_MATRIX[tech] else 0 for p in PROJECTS]
        z.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=project_labels,
            y=techniques,
            colorscale=[[0, "#f0f2f6"], [1, "#667eea"]],
            showscale=False,
            hovertemplate="Technique: %{y}<br>Project: %{x}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(400, len(techniques) * 28),
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(side="top"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Sunburst chart of ML categories ─────────────────────────────────────
    st.markdown("#### ML Category Breakdown")

    sunburst_data = {"labels": [], "parents": [], "values": [], "colors": []}
    sunburst_data["labels"].append("ML Portfolio")
    sunburst_data["parents"].append("")
    sunburst_data["values"].append(0)
    sunburst_data["colors"].append(COLORS["primary"])

    for cat_name, cat_info in CATEGORIES.items():
        matching = [p for p in PROJECTS if cat_name in p["categories"]]
        if not matching:
            continue
        sunburst_data["labels"].append(cat_name)
        sunburst_data["parents"].append("ML Portfolio")
        sunburst_data["values"].append(len(matching))
        sunburst_data["colors"].append(cat_info["color"])

        for p in matching:
            label = f"P{p['number']:02d}: {p['title']}"
            sunburst_data["labels"].append(label)
            sunburst_data["parents"].append(cat_name)
            sunburst_data["values"].append(1)
            sunburst_data["colors"].append(cat_info["color"])

    fig = go.Figure(
        go.Sunburst(
            labels=sunburst_data["labels"],
            parents=sunburst_data["parents"],
            values=sunburst_data["values"],
            marker=dict(colors=sunburst_data["colors"]),
            branchvalues="total",
            hovertemplate="<b>%{label}</b><extra></extra>",
        )
    )
    fig.update_layout(height=500, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Tools & libraries grid ──────────────────────────────────────────────
    st.markdown("#### Tools & Libraries")
    tool_cats = list(TOOLS.keys())
    tool_cols = st.columns(min(len(tool_cats), 4))
    for i, cat in enumerate(tool_cats):
        col = tool_cols[i % len(tool_cols)]
        items = ", ".join(TOOLS[cat])
        col.markdown(
            f"""
            <div class="tool-card">
                <h4>{cat}</h4>
                <p>{items}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Domain expertise ────────────────────────────────────────────────────
    st.markdown("#### Domain Expertise")
    dom_cols = st.columns(2)
    for i, d in enumerate(DOMAINS):
        col = dom_cols[i % 2]
        proj_list = ", ".join(
            f"P{pn:02d}" for pn in d["projects"]
        )
        col.markdown(
            f"""
            <div class="domain-card">
                <span class="domain-icon">{d["icon"]}</span>
                <span class="domain-name">{d["name"]}</span>
                <div class="domain-projects">Projects: {proj_list}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ABOUT
# ═════════════════════════════════════════════════════════════════════════════
def page_about() -> None:
    st.markdown('<p class="section-header">About This Portfolio</p>', unsafe_allow_html=True)

    st.markdown(
        """
        This portfolio contains **16 end-to-end machine learning and data
        science projects** built entirely on
        [Calgary Open Data](https://data.calgary.ca/). Each project includes
        a fully interactive Streamlit application, reproducible Jupyter
        notebooks, trained models, and documentation.

        The projects span a wide range of ML techniques — regression,
        classification, clustering, time-series forecasting, NLP, anomaly
        detection, survival analysis, and recommendation systems — applied
        to real urban challenges in transportation, public safety,
        environment, housing, energy, and economic development.
        """
    )

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.markdown("#### Quick Links")
        st.markdown(
            f"- [GitHub Repository]({GITHUB_BASE})\n"
            f"- [GitHub Profile](https://github.com/{GITHUB_USER})\n"
            f"- [Calgary Open Data Portal](https://data.calgary.ca/)"
        )

        st.markdown("#### Data Attribution")
        st.markdown(
            "Contains information licensed under the "
            "**Open Government License — City of Calgary**. "
            "Data accessed from [data.calgary.ca](https://data.calgary.ca/)."
        )

    with right:
        st.markdown("#### Installation")
        st.code(
            "git clone https://github.com/guydev42/calgary-data-portfolio.git\n"
            "cd calgary-data-portfolio\n"
            "python -m venv venv\n"
            "source venv/bin/activate   # Windows: venv\\Scripts\\activate\n"
            "pip install -r requirements.txt",
            language="bash",
        )

        st.markdown("#### Run Any Project")
        st.code(
            "cd project_01_building_permit_cost_predictor\n"
            "pip install -r requirements.txt\n"
            "streamlit run app.py",
            language="bash",
        )

    st.markdown("---")

    # Project table
    st.markdown("#### All Projects")
    rows = []
    for p in PROJECTS:
        rows.append(
            {
                "#": p["number"],
                "Project": p["title"],
                "Categories": ", ".join(p["categories"]),
                "Dataset": p["dataset"],
                "Size": p["dataset_size"],
            }
        )

    # Use native Streamlit dataframe (no pandas import needed at top level)
    import pandas as pd

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.caption(
        "Built as part of the "
        f"[Calgary Open Data ML/DS Portfolio]({GITHUB_BASE})"
    )


# ── Page router ─────────────────────────────────────────────────────────────
PAGES = {
    "Home": page_home,
    "Projects Gallery": page_gallery,
    "Project Details": page_details,
    "Skills & Tech": page_skills,
    "About": page_about,
}

PAGES[page]()
