
# FILE HEADER
### Tài liệu tham khảo:
# https://docs.streamlit.io/develop/api-reference/widgets/st.button
# https://docs.streamlit.io/develop/api-reference/widgets/st.button#st.button
# https://github.com/makcedward/nlpaug/blob/5238e0be734841b69651d2043df535d78a8cc594/nlpaug/res/word/spelling/spelling_en.txt

### Danh sách nhóm
# - Trần Công Toản (22110284)
# - Đồng Gia Sang (22110219)

### LINK VIDEO: 

import streamlit as st
import pandas as pd
from io import StringIO
from function.constants import *
from function.text_aug import ProcessAugDataframe, DataFrameToCSV


uploaded_file = st.file_uploader("Tải file dữ liệu")

global df

input_container = st.container(border=True)
output_container = st.container(border=True)

def show_preview_data(dataframe):
    input_container.write("Xem dữ liệu mẫu")
    input_container.write(dataframe.head())

def SubmitButtonHandler(**kwargs):
    new_df =  ProcessAugDataframe(**kwargs)
    output_container.write("Dữ liệu sau khi tăng cường")
    output_container.write(new_df)
    csv = DataFrameToCSV(new_df)
    output_container.download_button(
    "Tải về",
    csv,
    "file.csv",
    "text/csv",
    key='download-csv'
    )

if uploaded_file is not None:
    uploaded_file = StringIO(uploaded_file.getvalue().decode("utf-8"))
    df = pd.read_csv(uploaded_file)
    input_container.write(f"Dữ liệu đã cho có {len(df)} dòng và {len(df.columns)} cột")

    show_preview_data(df)
    selected_column = input_container.selectbox("Chọn cột dữ liệu tăng cường", df.columns)
    selected_method = input_container.selectbox("Chọn phương pháp tăng cường", METHODS)

    # Sliders
    aug_p = input_container.slider("Phần trăm từ sẽ tăng cường", 0, 1, 100)
    # Convert to percentage
    aug_p = aug_p / 100
    # Two input fields
    aug_min = input_container.number_input("Số lượng từ tối thiểu", 0, 100, 1)
    aug_max = input_container.number_input("Số lượng từ tối đa", 0, 100, 1)

    # Dict path
    dict_path = input_container.file_uploader("Tải file từ điển", type=[".txt"])
    if dict_path is not None:
        dict_path = StringIO(dict_path.getvalue().decode("utf-8"))
    else:
        dict_path = None


    action = input_container.selectbox("Chọn hành động (với RandomAug)", [ "substitute", "swap", "delete", "crop"])
    

    reversed_tokens = []
    params = {"df": df, "selected_column": selected_column, "method": selected_method, "aug_p": aug_p, "aug_min": aug_min, "aug_max": aug_max}
    if selected_method == SPELLING_AUG:
        params["dict_path"] = dict_path
    if selected_method == RANDOM_AUG or selected_method == REVERSED_AUG or selected_method == CONTEXTUALWORDEMBS_AUG:
        params["action"] = action
    if selected_method == REVERSED_AUG:
        params["reserved_tokens"] = []
    if selected_method == CONTEXTUALWORDEMBS_AUG:
        params["model_path"] = "bert-base-uncased"
        params["model_type"] = "bert"
    
    # Show button
    submit_button = input_container.button("Tăng cường dữ liệu", on_click=SubmitButtonHandler, kwargs=params)


