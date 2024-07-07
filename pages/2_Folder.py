import streamlit as st
import tkinter as tk
from tkinter import filedialog


st.set_page_config(
    page_title="Add folder",
    page_icon="📁",
)


def select_folder():
   root = tk.Tk()
   root.withdraw()
   folder_path = filedialog.askdirectory(master=root)
   root.destroy()
   return folder_path

selected_folder_path = st.session_state.get("folder_path", None)
folder_select_button = st.button("Select Folder")
if folder_select_button:
  selected_folder_path = select_folder()
  st.session_state.folder_path = selected_folder_path

if selected_folder_path:
   st.write('''“Selected folder path:”, selected_folder_path''')