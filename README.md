In modified_pose_tracking.py -

1. Added a new function called merged in the code (line 20 - 32)
2. Added -o in the args out video root for easier command line to run the script
3. Modified in lines 111 - 119 for 4 video streams and also created 4 folders for 4 videos' frames which can be used easily for merge function
4. Added new required libraries
5. The merge function returns img which can be used in every iteration for further process in the main function.
6. This modified version saves the frames of each video in their specific folder and also the merged images in "Merged" folder.

    command line to run - 
    python modified_pose_tracking.py -o task_output_video/final_output.mp4




mergeAll.py -
In a specfic folder if the 4 videos are available then we can simply run this python file and it will create 4 folders where it will save frames from each video in their designated folder and will also save the merged frames in Merged folder.

    command line to run - 
    python mergeAll.py





frame_count.py -
If we run this script from a folder which contains mp4 videos then it will print the frame count for each video. It's important to know about the video of with highest number of frames as it can be used as the limit for iterations.

    command line to run - 
    python frame_count.py
