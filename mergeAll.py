from PIL import Image
import cv2
import os

# merge function
def merge(img1,img2,img3,img4,i):
    img1 = Image.open('G:/merging photos/videos/1/'+str(i)+'.jpg')
    img2 = Image.open('G:/merging photos/videos/2/'+str(i)+'.jpg')
    img3 = Image.open('G:/merging photos/videos/3/'+str(i)+'.jpg')
    img4 = Image.open('G:/merging photos/videos/4/'+str(i)+'.jpg')
    new_image = Image.new('RGB',(3840,2160))
    new_image.paste(img1,(0,0))
    new_image.paste(img2,(1920,0))  
    new_image.paste(img3,(0,1080))  
    new_image.paste(img4,(1920,1080))
    img = new_image.resize((1920,1080))
    img.save('G:/merging photos//videos/Merged/'+str(i)+'.jpg','JPEG')

# Opens the Video file
i=1
cap1= cv2.VideoCapture('G:/Merging Photos/videos/1.mp4')
cap2= cv2.VideoCapture('G:/Merging Photos/videos/2.mp4')
cap3= cv2.VideoCapture('G:/Merging Photos/videos/3.mp4')
cap4= cv2.VideoCapture('G:/Merging Photos/videos/4.mp4')
os.mkdir('G:/Merging Photos/videos/1')
os.mkdir('G:/Merging Photos/videos/2')
os.mkdir('G:/Merging Photos/videos/3')
os.mkdir('G:/Merging Photos/videos/4')
os.mkdir('G:/Merging Photos/videos/Merged')

while(cap1.isOpened() or cap2.isOpened() or cap3.isOpened() or cap4.isOpened()):
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()
    ret4, frame4 = cap4.read()
    if (ret1 == True and ret2 == True and ret3 == True and ret4 == True):
        cv2.imwrite('G:/Merging Photos/videos/1/'+str(i)+'.jpg',frame1)
        cv2.imwrite('G:/Merging Photos/videos/2/'+str(i)+'.jpg',frame2)
        cv2.imwrite('G:/Merging Photos/videos/3/'+str(i)+'.jpg',frame3)
        cv2.imwrite('G:/Merging Photos/videos/4/'+str(i)+'.jpg',frame4)
        merge(frame1,frame2,frame3,frame4,i)
    
    elif (ret1 == True and ret2 == True and ret3 == True and ret4 == False):
        cv2.imwrite('G:/Merging Photos/videos/1/'+str(i)+'.jpg',frame1)
        cv2.imwrite('G:/Merging Photos/videos/2/'+str(i)+'.jpg',frame2)
        cv2.imwrite('G:/Merging Photos/videos/3/'+str(i)+'.jpg',frame3)
        frame4 = Image.new('RGB',(1920,1080),(0,0,0))
        frame4.save('G:/merging photos//videos/4/'+str(i)+'.jpg','JPEG')
        merge(frame1,frame2,frame3,frame4,i)
    
    elif (ret1 == True and ret2 == False and ret3 == True and ret4 == False):
        cv2.imwrite('G:/Merging Photos/videos/1/'+str(i)+'.jpg',frame1)
        cv2.imwrite('G:/Merging Photos/videos/3/'+str(i)+'.jpg',frame3)
        frame2 = Image.new('RGB',(1920,1080),(0,0,0))
        frame2.save('G:/merging photos//videos/2/'+str(i)+'.jpg','JPEG')
        frame4 = Image.new('RGB',(1920,1080),(0,0,0))
        frame4.save('G:/merging photos//videos/4/'+str(i)+'.jpg','JPEG')
        merge(frame1,frame2,frame3,frame4,i)
    
    elif(ret1 == True):
        cv2.imwrite('G:/Merging Photos/videos/1/'+str(i)+'.jpg',frame1)
        frame2 = Image.new('RGB',(1920,1080),(0,0,0))
        frame2.save('G:/merging photos//videos/2/'+str(i)+'.jpg','JPEG')
        frame3 = Image.new('RGB',(1920,1080),(0,0,0))
        frame3.save('G:/merging photos//videos/3/'+str(i)+'.jpg','JPEG')
        frame4 = Image.new('RGB',(1920,1080),(0,0,0))
        frame4.save('G:/merging photos//videos/4/'+str(i)+'.jpg','JPEG')
        merge(frame1,frame2,frame3,frame4,i)
        
    else:
        break
    
    i+=1