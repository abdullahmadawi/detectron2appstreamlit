import streamlit as st
from objectDetection import *

def func_1(x):
    detector = Detector(model_type=x)
    image_file = st.file_uploader("Upload image",type=['png','jpeg','jpg'])
    if image_file is not None:
        file_details = {"FileName":image_file.name,"FileType":image_file.type}
        st.write(file_details)
        img = Image.open(image_file)
        st.image(img,caption='Uploaded Image.')
        with open(image_file.name,mode='w') as f:
            f.write(image_file.getbuffer())
        st.success("Saved File")
        detector.onimage(image_file.name)
        img_ = Image.open("result.jpg")
        st.image(img_)

def main():
    with st.expander("Information"):
        st.markdown( '<p style="font-size: 30px;"><strong>Welcome to my Object Detection App!</strong></p>', unsafe_allow_html= True)
        st.markdown('<p style = "font-size : 20px; color : white;">This app was built using Streamlit, Detectron2 and OpenCv to demonstrate <strong>Object Detection</strong> in both videos (pre-recorded) and images.</p>', unsafe_allow_html=True)
    option = st.selectbox('What type of file?',('Images'))
    st.title('Object detection for images.')
    func_1('objectDetection')


if __name__ == '__main__':
    main()