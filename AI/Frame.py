

class Frame(State):
    def __init__(self, state_data, crop_factor=None, destination_size=None, vert_cent=0.5):
        State.__init__(self, state_data)
        #         self.state_data = self.process_state(crop_factor, vert_cent, destination_shape)
        self.state_data = self.process_state(crop_factor, vert_cent, (32,32))

    def process_state(self, crop_factor, vert_cent, destination_shape):
        """
        Does all the processing of the frame using the helper functions
        """
        frame = self.crop_frame(self.state_data, crop_factor, vert_cent)
        frame = self.normalise_frame(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        assert len(frame.shape) == 2
        #         print(frame)

        #         print(frame)
        #         io.imshow(frame)
        frame = self.downsample_frame(frame, destination_shape)
        #     Gray scaling was done above.
        #         frame = self.gray_scale(frame)

        return frame


    def gray_scale(self, frame, gray_scale_factor=[0.3, 0.3, 0.3]):
        frame = np.dot(frame, np.asarray(gray_scale_factor))
        return frame

    def normalise_frame(self, frame):
        frame = frame.astype('float32') / 255.0
        return frame

    def downsample_frame(self, frame, destination_shape):
        """
        downsamples the frame. decreases the resolution
        """
        frame = cv2.resize(np.asarray(frame), dsize=destination_shape, interpolation=cv2.INTER_CUBIC)
        return frame

    def crop_frame(self, frame, crop_factor, vert_cent=0.5):
        """
        input is the frame
        output is the cropped frame
        crop_factor is the ratio at which you want to crop the height and width
        cent is the ratio at which the centre of the cropped frame should be
        """
        if crop_factor is None:
            return frame

        height_factor = int((crop_factor[0]*frame.shape[0]) // 2)
        width_factor = int((crop_factor[1]*frame.shape[1]) // 2)
        vert_cent = int(frame.shape[0]*vert_cent)
        width_cent = int(frame.shape[1]*0.5)

        frame = frame[vert_cent - height_factor: vert_cent + height_factor,
                width_cent - width_factor: width_cent + width_factor]
        return frame
