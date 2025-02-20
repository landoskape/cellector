# Configuration file for Sphinx documentation builder
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "Cellector"
copyright = "2024, Andrew T. Landau"
author = "Andrew T. Landau"

import cellector

release = cellector.__version__

# Add any Sphinx extension module names here
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "myst_parser",
]

autodoc_member_order = "bysource"

html_theme_options = {
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/landoskape/cellector",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
    "source_repository": "https://github.com/landoskape/cellector/",
    "source_branch": "main",
    "source_directory": "docs/source",
    "top_of_page_buttons": ["view", "edit"],
    "light_css_variables": {
        "color-brand-primary": "red",
        "color-brand-content": "#CC3333",
        "color-admonition-background": "orange",
    },
    "dark_css_variables": {
        # Primary colors
        "color-brand-primary": "#4a9eff",  # Bright blue for primary elements
        "color-brand-content": "#6cb4ff",  # Lighter blue for content links
        "color-background-primary": "black",  # Very dark gray for background
        "color-background-secondary": "black",  # Dark gray for secondary background
        # Background colors
        "color-admonition-background": "#2a323f",  # Dark blue-gray for admonitions
        "color-code-background": "#1f2937",  # Darker gray for code blocks
        "color-highlight-on-scroll": "#374151",  # Medium gray for scroll highlights
        # Text colors
        "color-foreground-primary": "#f3f4f6",  # Light gray for primary text
        "color-foreground-secondary": "#d1d5db",  # Slightly darker gray for secondary text
        "color-foreground-muted": "#9ca3af",  # Muted gray for less important text
        # Sidebar colors
        "color-sidebar-background": "#black",  # Very dark gray for sidebar
        "color-sidebar-item-background--hover": "#1f2937",  # Darker gray for hover
        "color-sidebar-link": "#93c5fd",  # Light blue for sidebar links
        "color-sidebar-link--active": "#4a9eff",  # Bright blue for active link
        # Border colors
        "color-border": "#374151",  # Medium gray for borders
        "color-border-hover": "#4b5563",  # Slightly lighter gray for border hover
        # API documentation
        "color-api-name": "#93c5fd",  # Light blue for method names
        "color-api-pre-name": "#93c5fd",  # Light blue for namespace/class names
        "color-api-paren": "#d1d5db",  # Light gray for parentheses
        "color-api-keyword": "#4a9eff",  # Bright blue for keywords
        "color-api-highlight-code": "#1f2937",  # Dark gray for highlighted code background
        "color-api-selected-background": "#2a323f",  # Dark blue-gray for selected item background
        "color-api-background": "#1f2937",  # Dark gray for API background
        "color-api-background-hover": "#2a323f",  # Slightly lighter on hover
        "color-api-overall-background": "#111827",  # Very dark gray for overall API background
        "color-highlight-on-target": "#1f2937",  # Slightly blueish dark gray for highlight on target
        # Announcement banner
        "color-announcement-background": "#182234",  # Dark blue-gray for announcements
        "color-announcement-text": "#93c5fd",  # Light blue for announcement text
        # Search
        "color-search-background": "#1f2937",  # Dark gray for search
        "color-search-highlight": "#4a9eff33",  # Semi-transparent blue for search highlights
    },
}
html_theme = "furo"
