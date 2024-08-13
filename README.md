# Streamlit App for Blurring Faces 

### Process
- Face Detection
    - Using OpenCV DNN module
- Face Blurring - 2 ways:
    1. Gaussion blurring
    2. Pixelated blurring

### Running streamlit app 
```bash
streamlit run st_face_anonymize.py
```
- Upload an image
- Results show the original uploaded image and its blurred version

### Running streamlit app through Docker Container
- To build the container
```bash
docker build -t streamlit-face-blur .  
```
- To run the container
```bash
docker run -p 8501:8501 streamlit-face-blur 
```

### Running locally 
#### On Windows terminal
```bash
run_script.cmd
```

#### On Mac terminal
```bash
test_run.sh
```
(which implements the folowing code)
```bash
python face_anonymize.py --image /path/to/input_dir --save_dir /path/to/save_dir
```



