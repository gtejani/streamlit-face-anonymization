# Streamlit App for Blurring Faces 

## Process
- Face Detection
    - Using OpenCV DNN module
- Face Blurring - 2 ways:
    1. Gaussion blurring
    2. Pixelated blurring

For streamlit app
```bash
streamlit run st_face_anonymize.py
```
- Upload an image
- Results show the original uploaded image and its blurred version

Run the following script on Windows to blur faces,
```bash
run_script.cmd
```
(which implements the folowing code)
```bash
python face_anonymize.py --image /path/to/input_dir --save_dir /path/to/save_dir
```



