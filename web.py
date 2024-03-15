import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ugan import GeneratorNet
import torch
from ugan import img2tensor
from ugan import tensor2img
from PIL import Image, ImageOps
from torchvision import transforms
import io
from detection_infer import detect

model_path ='checkpoint_epoch_350.pth'
checkpoint = torch.load(model_path)


# Assuming 'model' is your model instance
netG = GeneratorNet().cuda()
model_dict = netG.state_dict()
new_state_dict = {}
for k, v in checkpoint['netG_state_dict'].items():
    if k in model_dict:
        new_state_dict[k] = v
        
        
for k, v in checkpoint['netG_state_dict'].items():
    name = k[7:]  # remove 'module.' prefix
    if name in model_dict:
        new_state_dict[name] = v        
# Load the newly created state dict

new_model = GeneratorNet().cuda()
for name, param in new_model.named_parameters():
    if name in model_dict:
        model_dict[name].copy_(param)
        
        
        
new_model.load_state_dict(new_state_dict)



# Main Streamlit app
def main():
    st.title('under water Image enhancement and trash detection App')
    st.write('Upload an image and we will process it!')

    # Upload image
    uploaded_image = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

    if uploaded_image is not None:
        with torch.no_grad():
            # Display the uploaded image
            st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
            img = Image.open(uploaded_image)
            img = ImageOps.exif_transpose(img)  # Corrects orientation
            img = img.resize((512, 512), resample=Image.BILINEAR)
            img_tensor = img2tensor(img)
            output_tensor = new_model.forward(img_tensor)
            output_img = tensor2img(output_tensor)
            cv2.imwrite('output.jpg', output_img)
            processed_image = Image.open('output.jpg')
            st.image(processed_image, caption='Processed Image', use_column_width=True)
            st.text("Running the model for trash detection")
            output_image, class_names = detect(processed_image)
            # Display the output
            st.text("Output Image:")
            # Display "Output Image"
            st.image(output_image)
            if len(class_names)==0:
                st.success("The water is clear!!!")
            else:
                st.error(f"Waste Detected!!!\nThe image has {class_names}")


if __name__ == '__main__':
    main()







