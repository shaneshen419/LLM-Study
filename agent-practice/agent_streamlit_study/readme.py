import streamlit as st
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pages.readme.readme_page import readme_show_page

if __name__ == "__main__":
    readme_show_page()
