import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "GenAI Agents"
copyright = "2024, Your Name"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx_autodoc_typehints", "sphinx_rtd_theme"]
html_theme = "sphinx_rtd_theme"

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
