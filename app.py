import torch
from predictor import Predictor
import streamlit as st 
from PIL import Image 
from torchvision import transforms

model = resnet50=torch.load('./resnet50.pt')
transform = transform_resNet = transforms.Compose([
        transforms.Resize([64,64]),
        transforms.RandomCrop(52),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

predictor = Predictor(model, transform)


st.set_option("deprecation.showfileUploaderEncoding", False)

st.sidebar.title("나비야~ 너는 누구니?")
st.sidebar.write("나비 종류 감별기")

st.sidebar.write("")

img_source = st.sidebar.radio("이미지 소스를 선택해 주세요.", ("이미지를 업로드", "카메라로 촬영"))

if img_source == "이미지를 업로드":
    img_file = st.sidebar.file_uploader("이미지를 선택해 주세요.", type = ["png", "jpg", "jpeg"])
elif img_source == "카메라로 촬영":
    img_file = st.camera_input("카메라로 촬영")

if(img_file is not None):
    with st.spinner("측정 중..."):
        img = Image.open(img_file)
        st.image(img, caption = "대상 이미지", width = 480)
        st.write("")
        
        results = predictor.predict(img)
        st.subheader("안녕하세요!")
        st.write(f"저는 {results} 랍니다~")
        st.write("만나서 반가워요 ^_^")
        
        st.sidebar.write("")
        st.sidebar.write("")