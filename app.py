import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from process import pred_transform


st.set_page_config(page_title="DMSP Particle Precipitate Flux Prediction App",
                  layout="wide")


model=load_model('models/model_lstmF.h5')

with open('models/scaler.pkl','rb') as f:
    scaler=pickle.load(f)

st.markdown("<h1 style='text-align: center;'>DMSP Particle Precipitate Flux Prediction App</h1>", unsafe_allow_html=True)
def main():
    with st.form('predictin_form'):
        st.subheader("Enter the below Features:")

        sin_SC_AACGM_LTIME=st.number_input("sinSC_AACGM_LTIME value: ",-1.00,1.00,format="%.2f")
        PC_6hr=st.number_input("PC_6hr Value: ",-3.00,9.50,format="%.2f")
        SymH_3hr=st.number_input("SymH_3hr Value: ",-163.00,67.00,format="%.2f")
        sC_AACGM_LAT=st.number_input("sC_AACGM_LAT Value: ",45.00,90.00,format="%.2f")
        Bx=st.number_input("Bx Value: ",-17.00,20.00,format="%.2f")
        By=st.number_input("By Value: ",-25.00,20.00,format="%.2f")
        
        
        submit=st.form_submit_button("Predict")
    if submit:
        sin_SC_AACGM_LTIME=sin_SC_AACGM_LTIME
        PC_6hr=PC_6hr
        SymH_3hr=SymH_3hr
        sC_AACGM_LAT=sC_AACGM_LAT
        Bx=Bx
        By=By

        value=[sin_SC_AACGM_LTIME,PC_6hr,SymH_3hr,sC_AACGM_LAT,Bx,By]

        scaled_value=scaler.transform(np.array([value]))
        
        #for LSTM
        final=scaled_value.reshape(scaled_value.shape[0],1,scaled_value.shape[1])

        pred=model.predict(final)[0][0]

        st.write(f"Total energy flux will be {round(pred_transform(pred),2)}")

if __name__=='__main__':
    main()