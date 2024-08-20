import os
import cv2
import numpy as np
from tqdm.notebook import tqdm
from moviepy.editor import VideoFileClip
from contextlib import redirect_stdout

def predict_single_action(model, video_file_path, SEQUENCE_LENGTH):
    '''
    This function will perform single action recognition prediction on a video using the LRCN model.
    Args:
    video_file_path:  The path of the video stored in the disk on which the action recognition is to be performed.
    SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
    '''

    # Initialize the VideoCapture object to read from the video file.
    video_reader = cv2.VideoCapture(video_file_path)

    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Declare a list to store video frames we will extract.
    frames_list = []
    
    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''

    # Get the number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH),1)

    # Iterating the number of times equal to the fixed length of sequence.
    for frame_counter in range(SEQUENCE_LENGTH):

        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        # Read a frame.
        success, frame = video_reader.read() 

        # Check if frame is not read properly then break the loop.
        if not success:
            break

        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (224, 224))
        
        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
        normalized_frame = resized_frame / 255
        
        # Appending the pre-processed frame into the frames list
        frames_list.append(normalized_frame)

    # Passing the  pre-processed frames to the model and get the predicted probabilities.
    predicted_labels_probabilities = model.predict(np.expand_dims(frames_list, axis = 0), verbose=0)[0]

    # Get the index of class with highest probability.
    predicted_label = np.argmax(predicted_labels_probabilities)

    # # Get the class name using the retrieved index.
    # predicted_class_name = CLASSES_LIST[predicted_label]
    
    # Release the VideoCapture object. 
    video_reader.release()

    return predicted_label, predicted_labels_probabilities[predicted_label]

def detect_violence(model, video_file_path):

    CLASSES_LIST = ["NoViolence", "Violence"]
    SEQUENCES = [20, 30, 40]
    detections = {}
    
    for seq_len in tqdm(SEQUENCES, desc="Detecting", unit="%", leave=True):
        
        detections[seq_len] = predict_single_action(model, video_file_path, seq_len)

    # pred_labels = [pred[0] for pred in detections.values()]
    # print(pred_labels)

    
    # Extract the first elements from the tuples in the dictionary values
    pred_labels = [tpl[0] for tpl in detections.values()]
    
    # Count occurrences of each element in the given list
    count_dict = {element: pred_labels.count(element) for element in set(pred_labels)}
    
    # Find the majority element
    majority_element = max(count_dict, key=count_dict.get)
    
    # Filter dictionary based on the majority element
    filtered_dict = {key: value for key, value in detections.items() if value[0] == majority_element}
    
    pred_probs = [tpl[1] for tpl in filtered_dict.values()]

    return CLASSES_LIST[majority_element], max(pred_probs)

def detect_weapons(model, confidence, video_file_path, show_conf=False):

    # Predict weapons on video
    # with open(os.devnull, 'w') as devnull:
    #     with redirect_stdout(devnull):

    DETECT = False
    results = list(
        model.predict(
            source=video_file_path, conf=confidence, save=True, show_conf=show_conf, save_crop=False, verbose=False,stream=True)) 
    
    weapon_detect = set()
    frames_with_weapons = []

    def folowing_frames(my_list, consecutive_count=3):

        differences = np.diff(my_list)
        consecutive_found = np.where(differences == 1)[0]

        return len(consecutive_found) >= consecutive_count - 1
        
    for frame, result in enumerate(results):
        
        boxes = result.boxes
        names = result.names
            
        if len(boxes) != 0:
            DETECT = True
        
        if DETECT:
            for box in boxes:
                weapon = names[box.cls[0].item()]
                weapon_detect.add(weapon)
                frames_with_weapons.append(frame)

    
    # Save the predicted video in web format
    saved_dir = results.__getitem__(0).save_dir
    vid_path  = results.__getitem__(0).path
    vid_dir, full_vid_name = os.path.split(vid_path)
    vid_name, extension = os.path.splitext(full_vid_name)
    
    
    def convert_avi_to_webm(input_file, output_file):
        # Load the AVI video clip
        clip = VideoFileClip(input_file)
    
        # Write the video clip to WebM format
        clip.write_videofile(output_file) #codec="libvpx", audio_codec="libvorbis" [for webm]
    
    
    input_avi = os.path.join(saved_dir, vid_name+'.avi')
    output_webm = os.path.join(saved_dir, vid_name+'.mp4')
    convert_avi_to_webm(input_avi, output_webm)

    return weapon_detect, output_webm, folowing_frames(frames_with_weapons)

def detect(violence_model, weapons_model, conf, video_file_path):

    violence_Y_N, confidence = detect_violence(violence_model, video_file_path)
    weapons_found, out_bboxs, folowing_frames = detect_weapons(weapons_model, conf, video_file_path)

    return violence_Y_N, confidence, weapons_found, out_bboxs, folowing_frames

        