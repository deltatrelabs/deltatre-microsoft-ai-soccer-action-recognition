import cv2
import numpy as np
  

class FocusedFlow:
 

    def __init__(self, feature_params, lk_params):
        self.feature_params = feature_params
        self.lk_params = lk_params

        
    def __boxes_to_mask(self, image, boxes, offset):
        if len(boxes) == 0:
            return None
        stencil = np.zeros(image.shape).astype(np.uint8)
        for box in boxes:
            b = box.astype(int)
            offset_x = int(offset[0] * (b[2] - b[0]) / 2)
            offset_y = int(offset[0] * (b[3] - b[1]) / 2)
            cv2.rectangle(
                stencil, 
                (b[0] + offset_y, b[1] + offset_x), 
                (b[2] + offset_y, b[3] + offset_x), 
                (255,255,255), 
                thickness=cv2.FILLED
            )
        return stencil
        
        
    def __init_keypoints(self, video, boxes, offset):
        cur = next((i for i in range(len(boxes)) if len(boxes[i]) > 0), -1)
        if cur == -1:
            return None
        frame = video[cur]
        mask = self.__boxes_to_mask(frame, boxes[cur], offset)
        p0 = cv2.goodFeaturesToTrack(frame, mask=mask, **self.feature_params)
        video = video[cur + 1:]
        boxes = boxes[cur + 1:]
        return video, boxes, p0, frame
    
    
    def __simple_init(self, video, boxes):
        frame = video[0]
        p0 = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)
        video = video[1:]
        boxes = boxes[1:]
        return video, boxes, p0, frame
        
    
    def run(self, video, boxes, offset=(0,0)):
        video = [ cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in video]
        video2 = video.copy()
        setup = self.__init_keypoints(video, boxes, offset)
        if not setup:
            video = video2
            setup = self.__simple_init(video, boxes)
        video, boxes, p0, old_frame = setup
        flow = []
        while video:
            frame = video[0]
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, frame, p0, None, **self.lk_params)
            if (np.sum(st) == 0): 
                setup = self.__init_keypoints(video, boxes, offset)
                if setup:
                    video, boxes, p0, old_frame = setup
                    flow.append(None)
                    continue
                else:
                    return flow
            good_new = p1[st==1]
            good_old = p0[st==1]
            flow.append(list(zip(good_old, good_new)))
            old_frame = frame
            p0 = good_new.reshape(-1,1,2)
            video = video[1:]
            boxes = boxes[1:]
        return flow
    
    
    def __draw_flow(self, image, flow, colors):
        if not flow:
            return image
        mask = np.zeros_like(image)
        for i, (old,new) in enumerate(flow):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), colors[i].tolist(), 2)
            image = cv2.circle(image, (a,b),5,colors[i].tolist(),-1)
            image = cv2.add(image,mask)
        return image
    
    
    def draw(self, video, boxes, offset=(0,0)):
        colors = np.random.randint(0,255,(100,3))
        flow = self.run(video, boxes, offset)
        return [ self.__draw_flow(frame, sub_flow, colors) for frame, sub_flow in zip(video[1:], flow)]
    
    
    def __calc_gradient(self, flowi):
        old, new = zip(*flowi)
        return (np.array(new) - np.array(old)).T
    
    
    def get_gradients(self, video, boxes, offset=(0,0)):
        flow = self.run(video, boxes, offset)
        return [self.__calc_gradient(flowi) for flowi in flow if flowi]