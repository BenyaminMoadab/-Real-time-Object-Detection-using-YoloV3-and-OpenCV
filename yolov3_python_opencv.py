# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 16:19:51 2021

@author: Asus
"""
import numpy  as np
import cv2

#Input Camera & (0) is first camera (Webcam)
cap = cv2.VideoCapture(0)

#Read the name of the data classes 
coco_file= "F:\\Programming course\\yolov3_python_opencv-master(ME)\\coco.names.txt"

#Create a class with empty arrays
coco_classes= []

# Read cfg files with a variable called "net_config"
net_config = "F:\\Programming course\\yolov3_python_opencv-master(ME)\\cfg\\yolov3.cfg"
# Read weights files with a variable called "net_weights"
net_weights = "F:\\Programming course\\yolov3_python_opencv-master(ME)\\cfg\\yolov3.weights"
# Create Parametr for Size Of Blob in "cv2.dnn.blobFromImage" function
blob_size = 320
#Class Probelity(Threshold)
confidence_threshold = 0.5
#Remove over 30% bandwidth overlap (because there are more bandwidths)
#if we decrease this value the accuracy go up
nms_threshold = 0.3



#Open "coco_file" & "rt" mean read text & "rb" mean read binary
with open(coco_file, 'rt') as f:
    
    #Fill "coco_classes" arrays with the file called from "coco_file" 
    #.rstrip("\n")   =   Delete extra (empty) strings at the end of the "coco_file"  ) ("\n")means format of empty string  
    #.split("\n")    =   Separate the words in each line from the coco_file file (except for 2-part words) 
    coco_classes = f.read().rstrip("\n").split("\n")
#Display coco_classes output
#print(coco_classes)
#Display output length of coco_classes file 
#print(len(coco_classes))



#create object for create Network in "net" variable , "dnn" is module , "readNetFromDarkNEt" is Network Method 
# "net_config" is Setting Network , "net_weights" is wieghts network
net = cv2.dnn.readNetFromDarknet(net_config , net_weights)
#  "setPreferableBackend" is backend processing fuction in "net" objcet (Selecting the most suitable processing system to ""build the network""(for example:Opencv))
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# "setPrefetableTarget" is Target fuction in "net" objcet (Select the desired processor to ""run the network""(for example: CPU))
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


#create function with "findobjects" name & input function is = output & img
#output means "output = net.forward(out_layer_names)"
#img meas "frame = cap.read()"
def findobjects(output, img):
    #give hight & weight & chanell of img
    img_h, img_w, img_c = img.shape
    #create emprty list for give the best "Bounding Box"
    bboxes = []
    #index of class for find the Best "Bounding Box"
    class_ids = []
    #value of the best index
    confidences = []
    
    for member in output:
        for detect_vector in member:
            #Separate arrays with an index of 5 to 80 from total arrays in each of the 300 layers
            scores = detect_vector[5:]
            #Return the most index from the top output list 
            class_id = np.argmax(scores)
            #Find the highest probability value of the best index
            confidence = scores[class_id]
            
            if confidence > confidence_threshold:
                #for convert Picture to real Size & convert to integer
                #In summary: Get the absolute width and height of the Boundin box available in "decet_vector"
                w,h = int(detect_vector[2] * img_w), int(detect_vector[3] * img_h)
                x,y = int((detect_vector[0] * img_w) - w/2), int((detect_vector[1] * img_h) - h/2)
                #Attach x,y,w,h values to the list (ie: list of lists where each value contains a list of 4 members) 
                bboxes.append([x,y,w,h])
                #Attach "class_ids" values to class_ids list
                class_ids.append(class_id)
                #Attach "confidences" values to confidences list
                confidences.append(float(confidence))
    #NMS(non max suprission)function selecet the best boundig box in another banding box
    #for exampel we have 3 banouding box with best value but we selecet 1 of them and delete another
    indices = cv2.dnn.NMSBoxes(bboxes, confidences, confidence_threshold, nms_threshold)
  # print(indices)
    for i in indices:
        i = i[0]
        bbox = bboxes[i]
        x,y,w,h = bbox[0], bbox[1], bbox[2], bbox[3]
        #img means input image  / ((x,y) means Source / (x+w, y+h) means  Destination/ 2 means Line thickness of boundin box
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        #img means input image 
        #f'{coco_classes[class_ids[i]].upper()} {int(confidences[i] * 100)}%' means name of write in tittleing
        cv2.putText(img, f'{coco_classes[class_ids[i]].upper()} {int(confidences[i] * 100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                
                
            
            




#Make an infinite loop
while True:
    #Read frame by frame (video) and save in success, frame variable
    success, frame = cap.read()
    
    
    #create object called "blob" for Convert InputImage(frame) to Blob with "cv2.dnn.blobFromImage" function
    # "cv2.dnn.blobFromImage" is for singel Frame(Image) & "cv2.dnn.blobFromImages" s for Multi Frame(Iamges)
    # "scalefactor" = Divide all the pixels of the input image into a number for normalization,Then they measure the size of each pixel between 0 and 1 (so their color spectrum turns gray) (Because the Input images are RGB, we divide by 255 (or multiply by 1/255))
    # "size" = size of Image(type of Tuple)
    # "mean"  = Abbreviation of "mean subtraction", Subtract a numerical value to reduce image light fluctuations from each R,G,B layer ((0,0,0) means that the number has not been reduced)
    # "swapRB" = Because OpenCV reads the input value in reverse that is BGR , so we have to replace the B&R (because when the value is to be reduced for light fluctuations, it is reduced correctly).
    # "crop" = If equal "True" Then crop the input of the webcam image(or photo) to 320 And if it is "False", it compresses it to a size of 320 and don't crop int
    # "ddepth"= Indicates the type of pixel data (integer,flot, ...) (default = flot_32) (If left blank in the value, it will take the default value)
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255, size=(blob_size, blob_size), mean=(0,0,0), swapRB=True, crop= False)
    
    
    # First For is Similar "blobFromImages" function (This for inserts one image into the loop each time (but now we have 1 image input and it becomes 3*320*320  => 3 = RGB , 320 = ImageSize )
    # in Secend For: We separate the members of the input image with the "enumerate" function and place it in the variable "B" and give the index to each with the variable "k" 
    # "str(k)" means show  "k" in the form of "String" fo Title of Windows & b for windows Content 
 #   for image in blob:
 #       for k, b in enumerate(image):
 #           cv2.imshow(str(k), b)
            
            
    #Insert the "blobe" object as the network input
    net.setInput(blob)
    
    
    #Execute pre-learned architecture(Darknet) on a given input (blob) and display objects as a matrix
    # "forward" is mathod in dnn architecture and Net Class, This method takes the name of the "output layers" and executes the network and shows us the output
    # "getUnconnectedOutLayersNames" is mathod in dnn architecture and Net Class, This Method takes the name of "output Lyers"
 #  print(net.getUnconnectedOutLayersNames())
    out_layer_names = net.getUnconnectedOutLayersNames()
    output = net.forward(out_layer_names)
    
    #show length of output
 #  print(len(output))
    
    #show Type of output
 #  print(type(output))
    
    #show first member of output
 #  print(output[0])
    
    #show shape of fist member output (output of print = (300,85) ==> orginal image size=320*320 in "first predict layer" we devide orginal image in 32, so output is 10*10*3 (3 is for RGB ), Each of these(10*10*3) is an array of 85 )
    #output of first predit layer : 10*10*3 =>(320*320 / 32) * 3
 #  print(output[0].shape)
    #show shape of secend member output
    #output of secend predit layer : 20*20*3 =>(320*320 / 16) * 3
 #  print(output[1].shape)
    #show shape of third member output
    #output of third predit layer : 40*40*3 =>(320*320 / 8) * 3
 #  print(output[2].shape)
    
    findobjects(output, frame)
    
    #Create a window called "Webcam" to display the output of the frame variable
    cv2.imshow("Webcam", frame)
    #If you press the keyboard input "q" in 1 millisecond Exit the program
    if cv2.waitKey(1) == ord('q'):
        break 
