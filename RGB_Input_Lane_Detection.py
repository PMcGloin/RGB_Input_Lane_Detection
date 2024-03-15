import os
import cv2
import numpy
from threading import Thread
from queue import Queue

class RGB_Input_Lane_Detection(object):
    def __init__(self):
        """
        Read input video stream and produce a video stream and video file with detected lane lines.
        Parameters:
            self.INPUT_Folder: location of input videos folder for parsing (folder path, string)
            self.OUTPUT_Folder: location of output video files (folder path, string)
            self.OUTPUT_VIDEO_POSTFIX: postfix of output video (string)
            self.SCALE_OUTPUT_FACTOR: scale factor to apply to combined output (factor, float)
            self.OPEN_CV_CUDA: Enable Open CV CUDA aceleration (If compiled, bool)
            self.GAUSS_KERNEL_SIZE: Square size of Gaussian Blur Kernel (int)
            self.CANNY_THRESHOLD_LOW: Low threshold for Canny hysteresis procedure (int)
            self.CANNY_THRESHOLD_HIGH: High threshold for Canny hysteresis procedure (int)
            self.HOUGH_RHO: Distance resolution of Hough transform accumulator (pixels, int)
            self.HOUGH_THETA: Angle resolution of Hough transform accumulator (radians, float)
            self.HOUGH_THRESHOLD: Lines greater than <var> will be returned from Hough transform (int)
            self.HOUGH_LINE_LENGHT_MIN: Line segments shorter than <var> are rejected from Hough transform (int)
            self.HOUGH_LINE_GAP_MAX: Maximum allowed gap between points on the same line to link them (int)
        """
        self.INPUT_FOLDER = 'input_videos'
        self.OUTPUT_FOLDER = 'output_videos'
        self.OUTPUT_VIDEO_POSTFIX = '_output.mp4'
        self.COMBINE_OUTPUT = True
        self.ADD_PERCENTAGE_LINES = False
        self.SAVE_OUTPUT_VIDEO = False
        self.SCALE_OUTPUT_FACTOR = 0.5
        self.OPEN_CV_CUDA = True
        self.GAUSS_KERNEL_SIZE = 5
        self.CANNY_THRESHOLD_LOW = 50
        self.CANNY_THRESHOLD_HIGH = 200
        self.POLY_BL = [0.00, 1.00]
        self.POLY_TL = [0.30, 0.6]
        self.POLY_BR = [0.9, 1.00]
        self.POLY_TR = [0.6, 0.6]
        self.HOUGH_RHO = 1
        self.HOUGH_THETA = numpy.pi/180
        self.HOUGH_THRESHOLD = 50
        self.HOUGH_LINE_LENGHT_MIN = 150
        self.HOUGH_LINE_GAP_MAX = 500

        if self.SAVE_OUTPUT_VIDEO:
            os.makedirs(self.OUTPUT_FOLDER, exist_ok= True)

        for filename in os.listdir(self.INPUT_FOLDER):
            if filename.endswith('.mp4'):
                self.input_filename = filename
                variables_string = '_GKSize_' + str(self.GAUSS_KERNEL_SIZE) + '_CTLow_' + str(self.CANNY_THRESHOLD_LOW) + '_CTHigh_' + str(self.CANNY_THRESHOLD_HIGH) + '_HR_' + str(self.HOUGH_RHO) + '_HT_' + str(self.HOUGH_RHO) + '_HThres_' + str(self.HOUGH_THRESHOLD) + '_HLL_' + str(self.HOUGH_LINE_LENGHT_MIN) + '_HLG_' + str(self.HOUGH_LINE_GAP_MAX)
                self.output_video_path = self.OUTPUT_FOLDER + '\\' + filename[:-4] + variables_string + self.OUTPUT_VIDEO_POSTFIX
                # Input video object
                self.input_video = cv2.VideoCapture(self.INPUT_FOLDER + '\\' + self.input_filename)
                # Video information setup
                self.input_fps = self.input_video.get(cv2.CAP_PROP_FPS)
                self.input_width = self.input_video.get(cv2.CAP_PROP_FRAME_WIDTH)
                self.input_height = self.input_video.get(cv2.CAP_PROP_FRAME_HEIGHT)
                if self.COMBINE_OUTPUT:
                    self.output_width = self.input_width * 3
                    self.output_height = self.input_height * 3
                    self.output_scaled_width = int(self.output_width * self.SCALE_OUTPUT_FACTOR)
                    self.output_scaled_height = int(self.output_height  * self.SCALE_OUTPUT_FACTOR)
                else:
                    self.output_width = self.input_width
                    self.output_height = self.input_height
                    self.output_scaled_width = int(self.output_width * self.SCALE_OUTPUT_FACTOR)
                    self.output_scaled_height = int(self.output_height  * self.SCALE_OUTPUT_FACTOR)

                if self.SAVE_OUTPUT_VIDEO:
                    # Video capture thread separation
                    capture_thread = Thread(target=self.capture_video)
                    capture_thread.daemon = True
                    capture_thread.start()
                    # Processing queue for video generation on separate thread
                    self.capture_thread_queue = Queue()

                while True:
                    if self.input_video.grab():
                        flag, self.input_frame = self.input_video.retrieve()
                        if not flag:
                            continue
                        else:
                            self.process_frame()
                            cv2.imshow(self.input_filename, self.combined_output_scaled)
                            if cv2.waitKey(10) == 27:
                                break
                    else:
                        cv2.destroyAllWindows()
                        break
                if self.SAVE_OUTPUT_VIDEO:
                    capture_thread.join()
                self.input_video.release()
        
    def capture_video(self):
        """
        Capture processed output frames
        """
        # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(self.output_video_path, fourcc, self.input_fps, (self.output_scaled_width, self.output_scaled_height))
        while True:
            try:
                # Read capture queue, throw queueEmpty exception if wait 1 second for another frame.
                output_video.write(self.capture_thread_queue.get(timeout=1))
                self.capture_thread_queue.task_done()
            except:
                output_video.release()
                break

    def process_frame(self):
        """
        Processing Flow Control
        """
        # Input Frame > Greyscale Frame
        self.convert_input_to_greyscale()
        # Greyscale Frame > Gaussian Blured Greyscale Frame
        self.apply_gaussian_blur()
        # Gaussian Blured Greyscale Frame > Canny Edge Detection Frame
        self.apply_canny_edge_detection()
        # Canny Edge Detection Frame > Region of Interest Mask Applied Frame
        self.apply_region_of_interest_mask()
        # Region of Interest Mask Applied Frame > Generate Probabalistic Hough Transform
        self.generate_hough_transform()
        # Probabalistic Hough Transform > Slope and Intercept of Left and Right Lanes
        self.generate_average_slope_intercept()
        # Slope and Intercept of Left and Right Lanes > Lines (represented as pixel points)
        self.generate_lane_lines()
        # Lines (represented as pixel points) > Draw lines onto the input frame
        self.draw_lane_lines()
        # Compile generated frames into scaled 9*9 output
        self.compile_output()

    def convert_input_to_greyscale(self):
        """
        Convert input frame to greyscale frame
        """
        if self.OPEN_CV_CUDA:
            frame_gpu = cv2.cuda.GpuMat()
            frame_gpu.upload(self.input_frame)
            self.greyscale_gpu = cv2.cuda.cvtColor(frame_gpu, cv2.COLOR_BGR2GRAY)
            self.greyscale = self.greyscale_gpu.download()
        else:
            self.greyscale = cv2.cvtColor(self.input_frame , cv2.COLOR_BGR2GRAY)
    
    def apply_gaussian_blur(self):
        """
        Apply Gaussian blur to greyscale frame
        """
        if self.OPEN_CV_CUDA:
            self.blur_gpu = cv2.cuda.GpuMat()
            sigma = (0.3 * ((self.GAUSS_KERNEL_SIZE-1)*0.5 -1) + 0.8)
            self.blur_gpu = cv2.cuda.createGaussianFilter(cv2.CV_8U, cv2.CV_8U, (self.GAUSS_KERNEL_SIZE, self.GAUSS_KERNEL_SIZE), sigma, sigma).apply(self.greyscale_gpu)
            self.blur = self.blur_gpu.download()
        else:
            self.blur = cv2.GaussianBlur(self.greyscale, (self.GAUSS_KERNEL_SIZE, self.GAUSS_KERNEL_SIZE), 0)
    
    def apply_canny_edge_detection(self):
        """
        Apply Canny edge detection algorithim to Gaussian blurred greyscale frame
        """
        if self.OPEN_CV_CUDA:
            self.edges_gpu = cv2.cuda.GpuMat()
            self.edges_gpu = cv2.cuda.createCannyEdgeDetector(self.CANNY_THRESHOLD_LOW, self.CANNY_THRESHOLD_HIGH).detect(self.blur_gpu)
            self.edges = self.edges_gpu.download()
        else:
            self.edges = cv2.Canny(self.blur, self.CANNY_THRESHOLD_LOW, self.CANNY_THRESHOLD_HIGH)

    def apply_region_of_interest_mask(self):
        """
        Apply region of interest mask to Canny edge detected Gaussian blurred greyscale frame
        """
        # create an array of the same size as of the input image 
        self.mask = numpy.zeros_like(self.edges)
        # if you pass an image with more then one channel
        if len(self.edges.shape) > 2:
            channel_count = self.edges.shape[2]
            ignore_mask_color = (255,) * channel_count
        # our image only has one channel so it will go under "else"
        else:
            # color of the mask polygon (white)
            ignore_mask_color = 255
        # creating a polygon to focus only on the road in the picture we have created this polygon in accordance to how the camera was placed
        rows, cols = self.edges.shape[:2]
        bottom_left = [cols * self.POLY_BL[0], rows * self.POLY_BL[1]]
        top_left	 = [cols * self.POLY_TL[0], rows * self.POLY_TL[1]]
        bottom_right = [cols * self.POLY_BR[0], rows * self.POLY_BR[1]]
        top_right = [cols * self.POLY_TR[0], rows * self.POLY_TR[1]]
        vertices = numpy.array([[bottom_left, top_left, top_right, bottom_right]], dtype=numpy.int32)
        # filling the polygon with white color and generating the final mask
        cv2.fillPoly(self.mask, vertices, ignore_mask_color)
        # performing Bitwise AND on the input image and mask to get only the edges on the road
        if self.OPEN_CV_CUDA:
            self.mask_gpu = cv2.cuda.GpuMat()
            self.mask_gpu.upload(self.mask)
            self.masked_image_gpu = cv2.cuda.bitwise_and(self.edges_gpu, self.mask_gpu)
            self.masked_image = self.masked_image_gpu.download()
        else:
            self.masked_image = cv2.bitwise_and(self.edges, self.mask)

    def generate_hough_transform(self):
        """
        Generate probabalistic Hough transform to region of interest masked Canny edge detected Gaussian blurred greyscale frame
        """
        # function returns an array containing dimensions of straight lines  appearing in the input image
        self.hough = cv2.HoughLinesP(self.masked_image, rho = self.HOUGH_RHO, theta = self.HOUGH_THETA, threshold = self.HOUGH_THRESHOLD, minLineLength = self.HOUGH_LINE_LENGHT_MIN, maxLineGap = self.HOUGH_LINE_GAP_MAX)

    def generate_average_slope_intercept(self):
        """
        Generates the slope and intercept of the left and right lanes of each image utilising line segments generated from the probabalistic Hough transform of the region of interest masked Canny edge detected Gaussian blurred greyscale frame
        """
        # (slope, intercept)
        left_lines = []
        # (lenght)
        left_weights = []
        # (slope, intercept)
        right_lines = []
        # (length)
        right_weights = []
        # if self.hough:
        if self.hough is not None:
            for line in self.hough:
                for x1, y1, x2, y2 in line:
                    if x1 == x2:
                        continue
                    # calculating slope of a line
                    slope = (y2 - y1) / (x2 - x1)
                    # calculating intercept of a line
                    intercept = y1 - (slope * x1)
                    # calculating length of a line
                    length = numpy.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
                    # slope of left lane is negative and for right lane slope is positive
                    if slope < 0:
                        left_lines.append((slope, intercept))
                        left_weights.append((length))
                    else:
                        right_lines.append((slope, intercept))
                        right_weights.append((length))
            self.left_lane = numpy.dot(left_weights, left_lines) / numpy.sum(left_weights) if len(left_weights) > 0 else None
            self.right_lane = numpy.dot(right_weights, right_lines) / numpy.sum(right_weights) if len(right_weights) > 0 else None
    
    def generate_lane_lines(self):
        """
        Generate lines as pixel points.
        """
        y1 = self.input_frame.shape[0]
        y2 = y1 * 0.6
        self.left_line = self.generate_pixel_points(y1, y2, self.left_lane)
        self.right_line = self.generate_pixel_points(y1, y2, self.right_lane)
    
    def draw_lane_lines(self, color=[255, 0, 0], thickness=12):
        """
        Draw lines onto the input image
        """
        self.line_image = numpy.zeros_like(self.input_frame)
        if self.left_line is not None:
            cv2.line(self.line_image, *self.left_line, color, thickness)
        if self.right_line is not None:
            cv2.line(self.line_image, *self.right_line, color, thickness)
        if self.OPEN_CV_CUDA:
            self.output_frame = cv2.cuda.addWeighted(self.input_frame, 1.0, self.line_image, 1.0, 0.0)
        else:
            self.output_frame = cv2.addWeighted(self.input_frame, 1.0, self.line_image, 1.0, 0.0) 

    def generate_pixel_points(self, y1, y2, line):
        """
        Generate pixel points for line utilising generated slope and intercept.
            Parameters:
                y1: y-value of the line's starting point.
                y2: y-value of the line's end point.
                line: The slope and intercept of the line.
        """
        if line is None:
            return None
        slope, intercept = line
        if slope == numpy.float64(0):
            x1 = 0
            x2 = 0
        else:
            x1 = int((y1 - intercept)/slope)
            x2 = int((y2 - intercept)/slope)
        y1 = int(y1)
        y2 = int(y2)
        return ((x1, y1), (x2, y2))
    def add_percentage_lines(self, input_frame):
        percent_lines = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        for percent in percent_lines:
            cv2.line(input_frame, (0, int(percent * (input_frame.shape[0]/100))),(input_frame.shape[1], int(percent * (input_frame.shape[0]/100))), (255,0,0), 1)
            cv2.line(input_frame, (int(percent * (input_frame.shape[1]/100)), 0),(int(percent * (input_frame.shape[1]/100)), input_frame.shape[0]), (255,0,0), 1)
            
    def compile_output(self):
        """
        Compiles frames, masks and an information frame into a single 9*9 output.
        The 9*9 output is then scaled to allow viewing.
        """
        # Convert grey colourspace images to RGB colourspace.
        if self.OPEN_CV_CUDA:
            greyscale_bgr_colour_space = cv2.cuda.cvtColor(self.greyscale_gpu, cv2.COLOR_GRAY2BGR).download()
            blur_bgr_colour_space = cv2.cuda.cvtColor(self.blur_gpu, cv2.COLOR_GRAY2BGR).download()
            edges_bgr_colour_space = cv2.cuda.cvtColor(self.edges_gpu, cv2.COLOR_GRAY2BGR).download()
            roi_bgr_colour_space = cv2.cuda.cvtColor(self.masked_image_gpu, cv2.COLOR_GRAY2BGR).download()
        else:
            greyscale_bgr_colour_space = cv2.cvtColor(self.greyscale , cv2.COLOR_GRAY2BGR)
            blur_bgr_colour_space = cv2.cvtColor(self.blur , cv2.COLOR_GRAY2BGR)
            edges_bgr_colour_space = cv2.cvtColor(self.edges , cv2.COLOR_GRAY2BGR)
            roi_bgr_colour_space = cv2.cvtColor(self.masked_image , cv2.COLOR_GRAY2BGR)
        
        roi_mask_bgr_colour_space = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)
        hough_image = numpy.zeros_like(roi_bgr_colour_space)
        if self.hough is not None:
            a,_,_ = self.hough.shape
            for i in range(a):
                cv2.line(hough_image, (self.hough[i][0][0], self.hough[i][0][1]), (self.hough[i][0][2], self.hough[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
        # Overlay Text Setup
        org = (40, 40)
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        # Add text to images
        input_frame = cv2.putText(self.input_frame, '1: Input Frame; No %d' % (int(self.input_video.get(cv2.CAP_PROP_POS_FRAMES))), org, fontFace, fontScale, color, thickness, cv2.LINE_AA)
        greyscale_bgr_colour_space = cv2.putText(greyscale_bgr_colour_space, '2: Greyscale Input Frame', org, fontFace, fontScale, color, thickness, cv2.LINE_AA)
        blur_bgr_colour_space = cv2.putText(blur_bgr_colour_space, '3: Gaussian Blur Applied; Kernel Size %d' % (self.GAUSS_KERNEL_SIZE), org, fontFace, fontScale, color, thickness, cv2.LINE_AA)
        edges_bgr_colour_space = cv2.putText(edges_bgr_colour_space, '4: Canny Frame; Thresh Low %d, Thresh High %d' % (self.CANNY_THRESHOLD_LOW, self.CANNY_THRESHOLD_HIGH), org, fontFace, fontScale, color, thickness, cv2.LINE_AA)
        # edges_bgr_colour_space = cv2.putText(edges_bgr_colour_space, '4: Canny Frame; Thresh Low %d, Thresh High %d, No %d' % (self.CANNY_THRESHOLD_LOW, self.CANNY_THRESHOLD_HIGH, int(self.input_video.get(cv2.CAP_PROP_POS_FRAMES))), org, fontFace, fontScale, color, thickness, cv2.LINE_AA)
        roi_mask_bgr_colour_space = cv2.putText(roi_mask_bgr_colour_space, '5: Region of Interest Mask', org, fontFace, fontScale, color, thickness, cv2.LINE_AA)
        roi_bgr_colour_space = cv2.putText(roi_bgr_colour_space, '6: Region of Interest Applied', org, fontFace, fontScale, color, thickness, cv2.LINE_AA)
        # roi_bgr_colour_space = cv2.putText(roi_bgr_colour_space, '6: Region of Interest Applied; No %d' % (int(self.input_video.get(cv2.CAP_PROP_POS_FRAMES))), org, fontFace, fontScale, color, thickness, cv2.LINE_AA)
        hough_image = cv2.putText(hough_image, '7: Hough Transform; Rho %d, Theta %f' % (self.HOUGH_RHO, self.HOUGH_THETA), org, fontFace, fontScale, color, thickness, cv2.LINE_AA)
        hough_image = cv2.putText(hough_image, 'Thresh %d, Line Len Min %d, Line Gap Max %d' % (self.HOUGH_THRESHOLD, self.HOUGH_LINE_LENGHT_MIN, self.HOUGH_LINE_GAP_MAX), (40, 80), fontFace, fontScale, color, thickness, cv2.LINE_AA)
        line_image = cv2.putText(self.line_image, '8: Generated Lane Lines', org, fontFace, fontScale, color, thickness, cv2.LINE_AA)
        result_image = cv2.putText(self.output_frame, '9: Output Frame', org, fontFace, fontScale, color, thickness, cv2.LINE_AA)
        # Output Layout
        if self.COMBINE_OUTPUT:
            # <Input_Frame>             <Greyscale_Input_Frame>     <Gaussian_Blur_Greyscale_Frame>
            horizontal_stack_top = numpy.hstack((input_frame, greyscale_bgr_colour_space, blur_bgr_colour_space))
            # <Edge_Detection_Frame>    <Region_of_Interest_Mask>   <Region_of_Interest_Applied>
            horizontal_stack_middle = numpy.hstack((edges_bgr_colour_space, roi_mask_bgr_colour_space, roi_bgr_colour_space))
            # <Hough_Lines>             <Generated_Lane_Lines>      <Output_Frame>
            horizontal_stack_bottom = numpy.hstack((hough_image, line_image, result_image))
            # 9*9 combined frame
            self.combined_output = numpy.vstack((horizontal_stack_top, horizontal_stack_middle, horizontal_stack_bottom))
        else:
            self.combined_output = roi_bgr_colour_space
            if self.ADD_PERCENTAGE_LINES:
                self.add_percentage_lines(self.combined_output)

        # Scale output
        if self.SCALE_OUTPUT_FACTOR == 1:
            self.combined_output_scaled = self.combined_output
        elif self.OPEN_CV_CUDA:
            combined_output_gpu = cv2.cuda.GpuMat()
            combined_output_gpu.upload(self.combined_output)
            self.combined_output_scaled = cv2.cuda.resize(combined_output_gpu, (self.output_scaled_width, self.output_scaled_height), fx=self.SCALE_OUTPUT_FACTOR, fy=self.SCALE_OUTPUT_FACTOR).download()
        else:
            self.combined_output_scaled = cv2.resize(self.combined_output, (self.output_scaled_width, self.output_scaled_height), fx=self.SCALE_OUTPUT_FACTOR, fy=self.SCALE_OUTPUT_FACTOR)
        if self.SAVE_OUTPUT_VIDEO:    
            self.capture_thread_queue.put(self.combined_output_scaled, block=False)

RGB_Input_Lane_Detection()