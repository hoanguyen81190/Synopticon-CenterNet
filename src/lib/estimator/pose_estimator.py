# -*- coding: utf-8 -*-
#Estimate head pose according to the facial landmarks"""
import cv2
import numpy as np
import os

actor_height = 157

class PoseEstimator:
    """Estimate head pose according to the facial landmarks"""
    """
            (0.0, 0.0, 0.0),             # Nose tip
            #(0.0, -330.0, -65.0),        # Chin
            (-2.42, 2.42, -2.42),     # Left eye center
            (2.42, 2.42, -2.42),      # Right eye center
            #(-150.0, -150.0, -125.0),    # Mouth left corner
            #(150.0, -150.0, -125.0)      # Mouth right corner
            (-4.48, 1.21, -12.1), 	# Left ear    
            (4.48, 1.21, -12.1),		# Right ear
            #(0.0, 170.0, -140.0),       # middle point between left and right eye
            #(-300.0, -180.0, -70.0),    # Left shoulder
            #(300.0, -180.0, -70.0),     # Right shoulders
            
            (0.0, 0.0, 0.0),             # Nose tip

            (-2.15, 1.70, -1.35),     # Left eye center
            (2.15, 1.70, -1.35),      # Right eye center

            (-4.30, 0.85, -5.40), 	# Left ear    
            (4.30, 0.85, -5.40),		# Right ear

            #(-300.0, -180.0, -70.0),    # Left shoulder
            #(300.0, -180.0, -70.0),     # Right shoulders
            
            
            JP's model:
            (0.0, 4.7, -1.7),             # Nose tip

            (-3.5, 0, 1.5),     # Left eye center
            (3.5, 0, 1.5),      # Right eye center

            (-8.5, 4.0, 11.8), 	# Left ear    
            (8.5, 4.0, 11.8),		# Right ear
            
            Hoa's model'
            (0.0, 4.0, -1.3),             # Nose tip

            (-3.0, 0, 0.5),     # Left eye center
            (3.0, 0, 0.5),      # Right eye center

            (-6.5, 3.0, 6), 	# Left ear    
            (6.5, 3.0, 6),		# Right ear
            """

    def __init__(self, img_size=(480, 640)):
        self.size = img_size

        #x:right, y:up, z:forward
        #Unit = cm
        
        # 3D model points.
        self.model_points = np.array([
            (0.0, 1.7, -1.35),             # Nose tip

            (-2.15, 0, 1.35),     # Left eye center
            (2.15, 0, 1.35),      # Right eye center

            (-4.30, 0.85, 5.40), 	# Left ear    
            (4.30, 0.85, 5.40),		# Right ear
        ]) 
        
        self.body_points = np.array([
            (-0.1295, 0.0, 0.0),    #left shoulder
            (0.1295, 0.0, 0.0),     #right shoulder
            (-0.0955, 0.288, 0.0),    #left hip
            (0.0955, 0.288, 0.0),   #right hip
            ])*actor_height

        self.model_points_68 = self._get_full_model_points()

        # Camera internals
        self.focal_length = self.size[1]
        self.camera_center = (self.size[1] / 2, self.size[0] / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.camera_center[0]],
             [0, self.focal_length, self.camera_center[1]],
             [0, 0, 1]], dtype="double")
        
        self.camera_matrix_body = np.array(
            [[self.focal_length, 0, self.camera_center[0]],
             [0, self.focal_length, self.camera_center[1]],
             [0, 0, 1]], dtype="double")

        # Assuming no lens distortion
        self.dist_coeefs = np.zeros((4, 1))

        # Rotation vector and translation vector
        self.r_vec = np.array([[0.01891013], [0.08560084], [-3.14392813]])
        self.t_vec = np.array(
            [[-14.97821226], [-10.62040383], [-2053.03596872]])
        # self.r_vec = None
        # self.t_vec = None

    def _get_full_model_points(self, filename=os.path.join(os.getcwd(), 'lib', 'estimator', 'assets/model.txt')):
        """Get all 68 3D model points from file"""
        raw_value = []
        with open(filename) as file:
            for line in file:
                raw_value.append(line)
        model_points = np.array(raw_value, dtype=np.float32)
        model_points = np.reshape(model_points, (3, -1)).T

        # Transform the model into a front view.
        model_points[:, 2] *= -1

        return model_points

    def show_3d_model(self):
        from matplotlib import pyplot
        from mpl_toolkits.mplot3d import Axes3D
        fig = pyplot.figure()
        ax = Axes3D(fig)

        x = self.model_points_68[:, 0]
        y = self.model_points_68[:, 1]
        z = self.model_points_68[:, 2]

        ax.scatter(x, y, z)
        ax.axis('square')
        pyplot.xlabel('x')
        pyplot.ylabel('y')
        pyplot.show()

    def solve_pose(self, image_points, body=False):
        """
        Solve pose from image points
        Return (rotation_vector, translation_vector) as pose.
        """
        #assert image_points.shape[0] == self.model_points_68.shape[0], "3D points and 2D points should be of same number."
        #print("points", self.model_points.shape, image_points.shape)
        #(_, rotation_vector, translation_vector) = cv2.solvePnP(
        #    self.model_points, image_points, self.camera_matrix, self.dist_coeefs, None, None, False, cv2.SOLVEPNP_UPNP)
        
        model_points = self.model_points
        if body:
            model_points = self.body_points
            
        #print(image_points.shape, model_points.shape)
        
        (_, rotation_vector, translation_vector, _) = cv2.solvePnPRansac(
            model_points, image_points, self.camera_matrix, self.dist_coeefs, None, None, False, 200, 12, 0.6, None, cv2.SOLVEPNP_UPNP)

        # (success, rotation_vector, translation_vector) = cv2.solvePnP(
        #     self.model_points,
        #     image_points,
        #     self.camera_matrix,
        #     self.dist_coeefs,
        #     rvec=self.r_vec,
        #     tvec=self.t_vec,
        #     useExtrinsicGuess=True)
        return rotation_vector, translation_vector

    def solve_pose_by_68_points(self, image_points):
        """
        Solve pose from all the 68 image points
        Return (rotation_vector, translation_vector) as pose.
        """

        if self.r_vec is None:
            (_, rotation_vector, translation_vector) = cv2.solvePnP(
                self.model_points_68, image_points, self.camera_matrix, self.dist_coeefs)
            self.r_vec = rotation_vector
            self.t_vec = translation_vector

        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points_68,
            image_points,
            self.camera_matrix,
            self.dist_coeefs,
            rvec=self.r_vec,
            tvec=self.t_vec,
            useExtrinsicGuess=True)

        return (rotation_vector, translation_vector)

    def draw_annotation_box(self, image, rotation_vector, translation_vector, color=(255, 255, 255), line_width=2):
        """Draw a 3D box as annotation of pose"""
        #print("rotation", rotation_vector)
        #print("translation", translation_vector)
        point_3d = []
        rear_size = 75
        rear_depth = 0
        point_3d.append((-rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, -rear_size, rear_depth))

        front_size = 100
        front_depth = 100
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d.append((-front_size, front_size, front_depth))
        point_3d.append((front_size, front_size, front_depth))
        point_3d.append((front_size, -front_size, front_depth))
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

        # Map to 2d image points
        (point_2d, _) = cv2.projectPoints(point_3d,
                                          rotation_vector,
                                          translation_vector,
                                          self.camera_matrix,
                                          self.dist_coeefs)
        point_2d = np.int32(point_2d.reshape(-1, 2))

        # Draw all the lines
        cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[1]), tuple(
            point_2d[6]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[2]), tuple(
            point_2d[7]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[3]), tuple(
            point_2d[8]), color, line_width, cv2.LINE_AA)

    def draw_axis(self, img, R, t): #x is red, y is green, z is blue
        points = np.float32(
            [[30, 0, 0], [0, 30, 0], [0, 0, 30], [0, 0, 0]])#.reshape(-1, 3)

        axisPoints, _ = cv2.projectPoints(
            points, R, t, self.camera_matrix, self.dist_coeefs)
        
        #print(axisPoints[0], axisPoints[1], axisPoints[2], axisPoints[3])

        img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(
            axisPoints[0].ravel()), (255, 0, 0), 3)
        img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(
            axisPoints[1].ravel()), (0, 255, 0), 3)
        img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(
            axisPoints[2].ravel()), (0, 0, 255), 3)

    def draw_axes(self, img, R, t):
        img	= cv2.drawFrameAxes(img, self.camera_matrix, self.dist_coeefs, R, t, 30)

    def evaluation(self, img, points, R, t, body=False):
        model_points = self.model_points
        if body:
            model_points = self.body_points
        pred_points = cv2.projectPoints(model_points, R, t, self.camera_matrix, self.dist_coeefs)[0]
 
        for p in pred_points:
            cv2.circle(img, (int(p[0][0]), int(p[0][1])), 3, (0,0,255), -1)
        mses = ((points.reshape([int(points.shape[0]/2), 2]) - pred_points.reshape([int(points.shape[0]/2), 2]))**2).mean()
        #print("MSE: mses", mses)
         

    def get_pose_marks(self, marks):
        """Get marks ready for pose estimation from 68 marks"""
        pose_marks = []
        pose_marks.append(marks[30])    # Nose tip
        pose_marks.append(marks[8])     # Chin
        pose_marks.append(marks[36])    # Left eye left corner
        pose_marks.append(marks[45])    # Right eye right corner
        pose_marks.append(marks[48])    # Mouth left corner
        pose_marks.append(marks[54])    # Mouth right corner
        return pose_marks