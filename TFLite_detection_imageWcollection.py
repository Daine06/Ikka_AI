import cv2 
import os 
import time 
import uuid
import numpy as np
import glob
import importlib.util
import firebase_admin
from firebase_admin import credentials, db
from firebase_admin import storage 


IMAGES_PATH = os.path.join('D:','AI','Object_Detection','captured_images')
if not os.path.exists(IMAGES_PATH):
    if os.name=='nt':
        os.mkdir (IMAGES_PATH)






cap = cv2.VideoCapture(0)

image_counter = 1

while True:
    
    ret, frame = cap.read()

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('c'):
        image_path = os.path.join(IMAGES_PATH ,f'image{image_counter}.jpg'.format(str(uuid.uuid1)))
        cv2.imwrite(image_path, frame)
        print(f'Saved image: {image_path}')
        image_counter += 1

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

threshold = 0.5
MODEL_NAME=os.path.join('D:','AI','Object_Detection','MyModel')
GRAPH_NAME = os.path.join(MODEL_NAME,'detect.tflite')
LABELMAP_NAME = os.path.join(MODEL_NAME,'labelmap.txt')

min_conf_threshold = float(threshold)

save_results =  os.path.join('D:','AI','Object_Detection','saved_img')


#script check
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter
   
#Get pics
PATH_TO_IMAGES = os.path.join(IMAGES_PATH)
images = glob.glob(PATH_TO_IMAGES + '/*.jpg') + glob.glob(PATH_TO_IMAGES + '/*.png') + glob.glob(PATH_TO_IMAGES + '/*.bmp')
#Save pics
RESULTS_DIR = save_results + '_results'
RESULTS_PATH = os.path.join(RESULTS_DIR)
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

PATH_TO_CKPT = os.path.join(GRAPH_NAME)
PATH_TO_LABELS = os.path.join(LABELMAP_NAME)

with open(PATH_TO_LABELS,'r') as f:
    labels = [line.strip() for line in f.readlines()]


if labels[0] == '???':
    del(labels[0])


interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5


outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: 
    boxes_idx, classes_idx, scores_idx = 0, 1, 2


for image_path in images:

    # Load image and resize to expected shape [1xHxWx3]
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape 
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

   
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

  
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] 
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] 
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] 

    detections = []

   
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

       
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

            detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])

    cv2.imshow('Object Detection', image)
    if cv2.waitKey(0) == ord('q'):
        break

    detection_folder = os.path.join(RESULTS_PATH,'infected_images')
    os.makedirs(detection_folder, exist_ok=True)

    no_detection_folder = os.path.join(RESULTS_PATH,'no_detection_images')
    os.makedirs(no_detection_folder, exist_ok=True)
    #if you want to create folders for detected and no detected images, use this code
    if detections:
        detection_image_path = os.path.join(detection_folder, os.path.basename(image_path))
        cv2.imwrite(detection_image_path, image)
        print(f'Saved image with detected classes: {detection_image_path}')
    if not detections:
        no_detection_image_path = os.path.join(no_detection_folder, os.path.basename(image_path))
        cv2.imwrite(no_detection_image_path, image)
        print(f'Saved image without detected classes: {no_detection_image_path}') 

    #If you want only images but no folders, use this code
    #image_fn = os.path.basename(image_path)
    #image_savepath = os.path.join(RESULTS_PATH,image_fn)
    #base_fn, ext = os.path.splitext(image_fn)
    #txt_result_fn = base_fn +'.txt'
    #txt_savepath = os.path.join(RESULTS_PATH,txt_result_fn)

    #cv2.imwrite(image_savepath, image)

    #with open(txt_savepath,'w') as f:
        #for detection in detections:
            #f.write('%s %.4f %d %d %d %d\n' % (detection[0], detection[1], detection[2], detection[3], detection[4], detection[5]))
        
cv2.destroyAllWindows()

certi = os.path.join('D:','AI','Object_Detection', 'ikka-2a296-firebase-adminsdk-cbiwq-da5d969a50.json')

cred = credentials.Certificate(certi)
firebase_admin.initialize_app(cred, {
    'storageBucket': 'ikka-2a296.appspot.com'
})

local_folder_path = RESULTS_DIR
firebase_folder_path = "AI"


bucket = storage.bucket()
for filename in os.listdir(local_folder_path):
    if filename.endswith(".jpg"):  
        local_file_path = os.path.join(local_folder_path,filename)
        firebase_file_path = f"{firebase_folder_path}/{filename}"

        blob = bucket.blob(firebase_file_path)
        blob.upload_from_filename(local_file_path)

        print(f"Uploaded {filename} to Firebase Storage.")


   
        

