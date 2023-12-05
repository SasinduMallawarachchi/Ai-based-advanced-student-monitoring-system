import face_recognition
import cv2
import numpy as np
import os

def input_and_segment(input_vid):
    output_folder = "student_segment"

    cap = cv2.VideoCapture(input_vid)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    student_frames_demention = []

    for x in range (5):

        ret, frame = cap.read()

        # Perform face detection using face_recognition
        image_ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        face_locations = face_recognition.face_locations(image_ndarray)

        # Draw rectangles on the original OpenCV image
        student_frames_demention = face_locations

    #
    # print(len(student_frames_demention))
    student_frame_set_list = []

    for x in range(len(student_frames_demention)):
        student_frame_set_list.append(np.zeros((1,500, 500,3), dtype=np.uint8))

    frame_count = 1
    #total_frames = 100

    for x in range(total_frames-10):
        # Capture frame-by-frame
        student_id = 0
        ret, frame = cap.read()

        for top, right, bottom, left in student_frames_demention:
            n = 3
            m = 3

            left_m = int((left - ((right - left) / 2) * n))
            right_m = int((right + ((right - left) / 2) * n))

            top_m = int((top - ((bottom - top) / 2) * n))
            bottom_m = int((bottom + ((bottom - top) / 2) * n))

            try:
                student_frame = frame[top_m:bottom_m , left_m:right_m ]
                student_frame = cv2.resize(student_frame, (500,500), interpolation=cv2.INTER_AREA)

                cv2.imshow(('student - '+str(student_id)), student_frame)
                student_frame = student_frame.reshape((1,500,500,3))

                student_frame_set_list[student_id] = np.concatenate((student_frame_set_list[student_id], student_frame), axis=0)
                student_id = student_id + 1

            except:
                print('pass')

            # cv2.rectangle(frame, (left_m, top_m), (right_m, bottom_m), (255, 0, 0), 2)
            cv2.waitKey(1)

        # print('frame_count - ', frame_count, ' / ', total_frames)
        frame_count = frame_count + 1

        # Display the image with rectangles

        cv2.imshow('Face Detection', frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #print(student_frame_set.shape)

    # print(student_frame_set_list)

    # Release the video capture stream
    cap.release()
    cv2.destroyAllWindows()



    for student in range(len(student_frames_demention)):

        fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')   # Codec
        fps = 30.0  # Frames per second
        frame_size = (500, 500)  # Frame size (width, height)

        output_vid_name = "student " + str(student)+'.mp4'

        output_video_path = os.path.join(output_folder, output_vid_name)

        out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)


        black_frame_remove = True
        for frame in student_frame_set_list[student]:

            if black_frame_remove == True:
                print('black_remove')
                cv2.waitKey(30)
                black_frame_remove = False
            else:
                out.write(frame)
                cv2.imshow('Segment',frame)
                cv2.waitKey(30)

        out.release()
        cap.release()
        cv2.destroyAllWindows()