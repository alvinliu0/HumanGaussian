import os
import cv2
import math
import json
import tqdm
import numpy as np
import dearpygui.dearpygui as dpg

from kiui.cam import OrbitCamera

import torch
import torch.nn.functional as F

import nvdiffrast.torch as dr
from gs_renderer import Renderer, MiniCam


def dot(x, y):
    if isinstance(x, np.ndarray):
        return np.sum(x * y, -1, keepdims=True)
    else:
        return torch.sum(x * y, -1, keepdim=True)


def length(x, eps=1e-20):
    if isinstance(x, np.ndarray):
        return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    else:
        return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x, eps=1e-20):
    return x / length(x, eps)


def joint_mapper_smplx_to_openpose18(joints):
    indices = (
        np.array(
            [
                56,  # nose
                13,  # neck
                18,  # right_shoulder
                20,  # right_elbow
                22,  # right_wrist
                17,  # left_shoulder
                19,  # left_elbow
                21,  # left_wrist
                3,  # right_hip
                6,  # right_knee
                9,  # right_ankle
                2,  # left_hip
                5,  # left_knee
                8,  # left_ankle
                57,  # right_eye
                58,  # left_eye
                59,  # right_ear
                60,  # left_ear
            ],
            dtype=np.int64,
        )
        - 1
    )
    return joints[indices]


class Skeleton:
    def __init__(self, opt):
        # init pose [18, 3], in [-1, 1]^3
        self.points3D = np.array(
            [
                [-0.00313026, 0.16587697, 0.05414092],
                [-0.00857283, 0.1093518, -0.00522604],
                [-0.06817748, 0.10397182, -0.00657925],
                [-0.11421658, 0.04033477, 0.00040599],
                [-0.15643744, -0.02915882, 0.03309248],
                [0.05288884, 0.10729481, -0.00067854],
                [0.10355149, 0.04464601, -0.00735265],
                [0.15390812, -0.02282556, 0.03085238],
                [0.03897187, -0.0403506, 0.00220192],
                [0.04027461, -0.15746351, -0.00187036],
                [0.04605377, -0.26837209, -0.0018945],
                [-0.0507806, -0.04887162, 0.0022531],
                [-0.04873568, -0.16551849, -0.00128197],
                [-0.04840493, -0.27510208, -0.00128831],
                [-0.03098677, 0.19395538, 0.01987491],
                [0.01657042, 0.19560097, 0.02724142],
                [-0.05411603, 0.17336673, -0.01328044],
                [0.03733583, 0.16922003, -0.00946565],
            ],
            dtype=np.float32,
        )

        self.name = [
            "nose",
            "neck",
            "right_shoulder",
            "right_elbow",
            "right_wrist",
            "left_shoulder",
            "left_elbow",
            "left_wrist",
            "right_hip",
            "right_knee",
            "right_ankle",
            "left_hip",
            "left_knee",
            "left_ankle",
            "right_eye",
            "left_eye",
            "right_ear",
            "left_ear",
        ]

        # homogeneous
        self.points3D = np.concatenate(
            [self.points3D, np.ones_like(self.points3D[:, :1])], axis=1
        )  # [18, 4]

        # lines [17, 2]
        self.lines = np.array(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 4],
                [1, 5],
                [5, 6],
                [6, 7],
                [1, 8],
                [8, 9],
                [9, 10],
                [1, 11],
                [11, 12],
                [12, 13],
                [0, 14],
                [14, 16],
                [0, 15],
                [15, 17],
            ],
            dtype=np.int32,
        )

        # keypoint color [18, 3]
        # color as in controlnet_aux (https://github.com/patrickvonplaten/controlnet_aux/blob/master/src/controlnet_aux/open_pose/util.py#L94C5-L96C73)
        self.colors = [
            [255, 0, 0],
            [255, 85, 0],
            [255, 170, 0],
            [255, 255, 0],
            [170, 255, 0],
            [85, 255, 0],
            [0, 255, 0],
            [0, 255, 85],
            [0, 255, 170],
            [0, 255, 255],
            [0, 170, 255],
            [0, 85, 255],
            [0, 0, 255],
            [85, 0, 255],
            [170, 0, 255],
            [255, 0, 255],
            [255, 0, 170],
            [255, 0, 85],
        ]

        # smplx mesh if available
        self.smplx_model = None
        self.vertices = None
        self.faces = None
        self.ori_center = None
        self.ori_scale = None

        self.body_pose = np.zeros((21, 3), dtype=np.float32)
        # let's default to A-pose
        self.body_pose[15, 2] = -0.7853982
        self.body_pose[16, 2] = 0.7853982
        self.body_pose[0, 1] = 0.2
        self.body_pose[0, 2] = 0.1
        self.body_pose[1, 1] = -0.2
        self.body_pose[1, 2] = -0.1
        """ SMPLX body_pose definition
        0: 'left_hip',#'L_Hip', XYZ -> (-X)(-Y)Z, 后外高 -> 前里高 (3) XYZ
        1: 'right_hip',#'R_Hip', (4) XYZ -> (-X)(-Y)Z, 后里低 -> 前外低 (4) XYZ
        2: 'spine1',#'Spine1', (-X)Y(-Z) -> (0) XYZ
        3: 'left_knee',#'L_Knee', 同左UpperLeg
        4: 'right_knee',#'R_Knee',同右UpperLeg
        5: 'spine2',
        6: 'left_ankle',
        7: 'right_ankle',#'R_Ankle',同右UpperLeg
        8: 'spine3',#'Spine3', (-X)Y(-Z) 同脊椎
        9: 'left_foot',#'L_Foot',同左UpperLeg
        10: 'right_foot',#'R_Foot',同右UpperLeg
        11: 'neck',#'Neck', (-X)Y(-Z) 同脊椎
        12: 'left_collar',#'L_Collar', XYZ -> ZXY (VRM), 前拧, 后, 高 -> 高, 前拧, 后 (1) YZX
        13: 'right_collar',#'R_Collar', XYZ -> (-Z)(-X)Y , 前拧, 前, 低 -> 高, 后拧, 前 (2) YZX
        14: 'head',#'Head', (-X)Y(-Z) 同脊椎
        15: 'left_shoulder',#'L_Shoulder', 同左肩膀
        16: 'right_shoulder',#'R_Shoulder', 同右肩膀
        17: 'left_elbow',#'L_Elbow', 同左肩膀
        18: 'right_elbow',#'R_Elbow', 同右肩膀
        19: 'left_wrist',#'L_Wrist', 同左肩膀
        20: 'right_wrist',#'R_Wrist', 同右肩膀
        """

        self.left_hand_pose = np.zeros((15, 3), dtype=np.float32)
        self.right_hand_pose = np.zeros((15, 3), dtype=np.float32)
        """ hand_pose definition
        index, middle, pinky, ring, thumb; each with 3 joints.
        """

        # gaussian model
        self.gs = Renderer(sh_degree=0, white_background=False)
        self.gs.gaussians.load_ply(opt.ply)

        # motion data
        self.motion_seq = np.load(opt.motion)["poses"][:, 1:22]

        # gaussian center to smplx faces mapping
        self.mapping_dist = None
        self.mapping_face = None
        self.mapping_uvw = None

    @property
    def center(self):
        return self.points3D[:, :3].mean(0)

    @property
    def center_upper(self):
        return self.points3D[0, :3]

    @property
    def torso_bbox(self):
        # valid_points = self.points3D[[0, 1, 8, 11], :3]
        valid_points = self.points3D[:, :3]
        # assure 3D thickness
        min_point = valid_points.min(0) - 0.1
        max_point = valid_points.max(0) + 0.1
        remedy_thickness = np.maximum(0, 0.8 - (max_point - min_point)) / 2
        min_point -= remedy_thickness
        max_point += remedy_thickness
        return min_point, max_point

    def sample_points(self, noise=0.05, N=1000):
        # just sample N points around each line
        pc = []
        for i in range(17):
            A = self.points3D[[self.lines[i][0]], :3]  # [1, 3]
            B = self.points3D[[self.lines[i][1]], :3]
            x = np.linspace(0, 1, N)[:, None]  # [N, 1]
            points = A * (1 - x) + B * x
            # add noise
            points += np.random.randn(N, 3) * noise
            pc.append(points)
        pc = np.concatenate(pc, axis=0)  # [17 * N, 3]
        return pc

    def write_json(self, path):
        with open(path, "w") as f:
            d = {}
            for i in range(18):
                d[self.name[i]] = self.points3D[i, :3].tolist()
            json.dump(d, f)

    def load_json(self, path):
        if os.path.exists(path):
            with open(path, "r") as f:
                d = json.load(f)

        # load keypoints
        for i in range(18):
            self.points3D[i, :3] = np.array(d[self.name[i]])

    def load_smplx(self, path, betas=None, expression=None, gender="neutral"):
        import smplx

        if self.smplx_model is None:
            self.smplx_model = smplx.create(
                path,
                model_type="smplx",
                gender=gender,
                use_face_contour=False,
                num_betas=10,
                num_expression_coeffs=10,
                ext="npz",
                use_pca=False,  # explicitly control hand pose
                flat_hand_mean=True,  # use a flatten hand default pose
            )

        # betas = torch.randn([1, self.smplx_model.num_betas], dtype=torch.float32)
        # expression = torch.randn([1, self.smplx_model.num_expression_coeffs], dtype=torch.float32)

        smplx_output = self.smplx_model(
            body_pose=torch.tensor(self.body_pose, dtype=torch.float32).unsqueeze(0),
            left_hand_pose=torch.tensor(
                self.left_hand_pose, dtype=torch.float32
            ).unsqueeze(0),
            right_hand_pose=torch.tensor(
                self.right_hand_pose, dtype=torch.float32
            ).unsqueeze(0),
            betas=betas,
            expression=expression,
            return_verts=True,
        )

        self.vertices = smplx_output.vertices.detach().cpu().numpy()[0]  # [10475, 3]
        self.faces = self.smplx_model.faces  # [20908, 3]

        # tmp: save deformed smplx mesh
        # import trimesh
        # _mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces)
        # _mesh.export('smplx.obj')

        joints = smplx_output.joints.detach().cpu().numpy()[0]  # [127, 3]
        joints = joint_mapper_smplx_to_openpose18(joints)

        self.points3D = np.concatenate(
            [joints, np.ones_like(joints[:, :1])], axis=1
        )  # [18, 4]

        # rescale and recenter
        if self.ori_center is None:
            vmin = self.vertices.min(0)
            vmax = self.vertices.max(0)
            self.ori_center = (vmax + vmin) / 2
            self.ori_scale = 0.6 / np.max(vmax - vmin)

        self.vertices = (self.vertices - self.ori_center) * self.ori_scale
        self.points3D[:, :3] = (self.points3D[:, :3] - self.ori_center) * self.ori_scale

        self.scale(-10)  # rescale

        # update gaussian location
        if self.mapping_face is None:
            import cubvh

            points = self.gs.gaussians.get_xyz.detach()

            BVH = cubvh.cuBVH(self.vertices, self.faces)
            mapping_dist, mapping_face, mapping_uvw = BVH.signed_distance(
                points, return_uvw=True, mode="raystab"
            )

            self.mapping_dist = mapping_dist.detach().cpu().numpy()
            self.mapping_face = mapping_face.detach().cpu().numpy().astype(np.int32)
            self.mapping_uvw = mapping_uvw.detach().cpu().numpy().astype(np.float32)

            faces = self.faces[self.mapping_face]
            v0 = self.vertices[faces[:, 0]]
            v1 = self.vertices[faces[:, 1]]
            v2 = self.vertices[faces[:, 2]]
            # face normals
            fnormals = np.cross(v1 - v0, v2 - v0)
            fnormals = fnormals / (
                np.linalg.norm(fnormals, axis=1, keepdims=True) + 1e-20
            )

            # it seems some point (3000 out of 526294) cannot be accurately remapped...
            cpoints = (
                v0 * self.mapping_uvw[:, [0]]
                + v1 * self.mapping_uvw[:, [1]]
                + v2 * self.mapping_uvw[:, [2]]
            )
            points = cpoints + self.mapping_dist[:, None] * fnormals

            gt_points = self.gs.gaussians.get_xyz.detach().cpu().numpy()

            # print(points, gt_points)
            err = np.sqrt(np.sum((points - gt_points) ** 2, axis=-1))
            print(err.max(), err.mean(), err.min(), (err > 0.01).sum())

            # cull these erronous points...
            mask = ~(err > 0.01)
            self.gs.gaussians._xyz = self.gs.gaussians._xyz[mask]
            self.gs.gaussians._features_dc = self.gs.gaussians._features_dc[mask]
            self.gs.gaussians._features_rest = self.gs.gaussians._features_rest[mask]
            self.gs.gaussians._opacity = self.gs.gaussians._opacity[mask]
            self.gs.gaussians._scaling = self.gs.gaussians._scaling[mask]
            self.gs.gaussians._rotation = self.gs.gaussians._rotation[mask]
            self.mapping_dist = self.mapping_dist[mask]
            self.mapping_face = self.mapping_face[mask]
            self.mapping_uvw = self.mapping_uvw[mask]

        else:
            faces = self.faces[self.mapping_face]
            v0 = self.vertices[faces[:, 0]]
            v1 = self.vertices[faces[:, 1]]
            v2 = self.vertices[faces[:, 2]]
            # face normals
            fnormals = np.cross(v1 - v0, v2 - v0)
            fnormals = fnormals / (
                np.linalg.norm(fnormals, axis=1, keepdims=True) + 1e-20
            )

            # closest points on mesh
            cpoints = (
                v0 * self.mapping_uvw[:, [0]]
                + v1 * self.mapping_uvw[:, [1]]
                + v2 * self.mapping_uvw[:, [2]]
            )

            # new position
            points = cpoints + self.mapping_dist[:, None] * fnormals
            self.gs.gaussians._xyz = torch.tensor(points, dtype=torch.float32).cuda()

    def scale(self, delta):
        self.points3D[:, :3] *= 1.1 ** (-delta)
        if self.vertices is not None:
            self.vertices *= 1.1 ** (-delta)

    def pan(self, rot, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        delta = 0.0005 * rot.as_matrix()[:3, :3] @ np.array([dx, -dy, dz])
        self.points3D[:, :3] += delta
        if self.vertices is not None:
            self.vertices += delta

    def draw(self, mvp, H, W, enable_occlusion=False):
        # mvp: [4, 4]
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        points = self.points3D @ mvp.T  # [18, 4]
        points = points[:, :3] / points[:, 3:]  # NDC in [-1, 1]

        xs = (points[:, 0] + 1) / 2 * H  # [18]
        ys = (points[:, 1] + 1) / 2 * W  # [18]
        mask = (xs >= 0) & (xs < H) & (ys >= 0) & (ys < W)

        # hide certain keypoints based on empirical occlusion
        if enable_occlusion:
            # decide view by the position of nose between two ears
            if points[0, 2] > points[-1, 2] and points[0, 2] < points[-2, 2]:
                # left view
                mask[-2] = False  # no right ear
                if xs[-4] > xs[-3]:
                    mask[-4] = False  # no right eye if it's "righter" than left eye
            elif points[0, 2] < points[-1, 2] and points[0, 2] > points[-2, 2]:
                # right view
                mask[-1] = False
                if xs[-3] < xs[-4]:
                    mask[-3] = False
            elif points[0, 2] > points[-1, 2] and points[0, 2] > points[-2, 2]:
                # back view
                mask[0] = False  # no nose
                mask[-3] = False  # no eyes
                mask[-4] = False

        # 18 points
        for i in range(18):
            if not mask[i]:
                continue
            cv2.circle(
                canvas, (int(xs[i]), int(ys[i])), 4, self.colors[i], thickness=-1
            )

        # 17 lines
        for i in range(17):
            cur_canvas = canvas.copy()
            if not mask[self.lines[i]].all():
                continue
            X = xs[self.lines[i]]
            Y = ys[self.lines[i]]
            mY = np.mean(Y)
            mX = np.mean(X)
            length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
            polygon = cv2.ellipse2Poly(
                (int(mX), int(mY)), (int(length / 2), 4), int(angle), 0, 360, 1
            )

            cv2.fillConvexPoly(cur_canvas, polygon, self.colors[i])

            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

        canvas = canvas.astype(np.float32) / 255
        return canvas, np.stack([xs, ys], axis=1)

    def render_gs(self, cam, H, W):
        cur_cam = MiniCam(cam.pose, H, W, cam.fovy, cam.fovx, cam.near, cam.far)

        out = self.gs.render(cur_cam)

        image = out["image"].permute(1, 2, 0).contiguous()  # [H, W, 3] in [0, 1]

        return image.detach().cpu().numpy()


class GUI:
    def __init__(self, opt):
        self.opt = opt
        self.gui = opt.gui

        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        self.skel = Skeleton(opt)
        self.glctx = dr.RasterizeCudaContext()

        self.render_buffer = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # camera moved, should reset accumulation
        self.mode = "gs"

        self.playing = False
        self.seq_id = 0

        self.save_image_path = "pose.png"
        self.save_json_path = "pose.json"
        self.mouse_loc = np.array([0, 0])
        self.points2D = None  # [18, 2]
        self.point_idx = 0
        self.drag_sensitivity = 0.0001
        self.pan_scale_skel = False
        self.enable_occlusion = True

        if self.gui:
            dpg.create_context()
            self.register_dpg()
            self.step()

    def __del__(self):
        if self.gui:
            dpg.destroy_context()

    def step(self):
        if self.need_update:
            # mvp
            mv = self.cam.view  # [4, 4]
            proj = self.cam.perspective  # [4, 4]
            mvp = proj @ mv

            if self.mode == "skel":
                # render our openpose image, somehow
                self.render_buffer, self.points2D = self.skel.draw(
                    mvp, self.H, self.W, enable_occlusion=self.enable_occlusion
                )

            # if with smplx, overlay normal of mesh
            elif self.mode == "mesh":
                self.render_buffer = self.render_mesh_normal(
                    mvp, self.H, self.W, self.skel.vertices, self.skel.faces
                )

            # overlay gaussian splattings...
            elif self.mode == "gs":
                self.render_buffer = self.skel.render_gs(self.cam, self.H, self.W)

            self.need_update = False

            if self.gui:
                dpg.set_value("_texture", self.render_buffer)

        if self.playing:
            self.skel.body_pose = np.array(self.skel.motion_seq[self.seq_id % len(self.skel.motion_seq)])
            self.seq_id += 1
            self.skel.load_smplx(self.opt.smplx_path)
            self.need_update = True

    def render_mesh_normal(self, mvp, H, W, vertices, faces):
        mvp = torch.from_numpy(mvp.astype(np.float32)).cuda()
        vertices = torch.from_numpy(vertices.astype(np.float32)).cuda()
        faces = torch.from_numpy(faces.astype(np.int32)).cuda()

        vertices_clip = (
            torch.matmul(
                F.pad(vertices, pad=(0, 1), mode="constant", value=1.0),
                torch.transpose(mvp, 0, 1),
            )
            .float()
            .unsqueeze(0)
        )  # [1, N, 4]
        rast, _ = dr.rasterize(self.glctx, vertices_clip, faces, (H, W))

        i0, i1, i2 = faces[:, 0].long(), faces[:, 1].long(), faces[:, 2].long()
        v0, v1, v2 = vertices[i0, :], vertices[i1, :], vertices[i2, :]

        face_normals = torch.cross(v1 - v0, v2 - v0)

        # Splat face normals to vertices
        vn = torch.zeros_like(vertices)
        vn.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

        # Normalize, replace zero (degenerated) normals with some default value
        vn = torch.where(
            dot(vn, vn) > 1e-20,
            vn,
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=vn.device),
        )
        vn = safe_normalize(vn)

        normal, _ = dr.interpolate(vn.unsqueeze(0).contiguous(), rast, faces)
        normal = safe_normalize(normal)

        normal_image = (normal[0] + 1) / 2
        normal_image = torch.where(
            rast[..., 3:] > 0, normal_image, torch.tensor(1).to(normal_image.device)
        )  # remove background
        buffer = normal_image.detach().cpu().numpy()

        return buffer

    def register_dpg(self):
        ### register texture

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.render_buffer,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window

        # the rendered image, as the primary window
        with dpg.window(
            label="Viewer",
            tag="_primary_window",
            width=self.W,
            height=self.H,
            pos=[0, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
        ):
            dpg.add_image("_texture")

        # dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(
            label="Control",
            tag="_control_window",
            width=600,
            height=self.H,
            pos=[self.W, 0],
            no_move=True,
            no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # save image
            def callback_save_image(sender, app_data):
                image = (self.render_buffer * 255).astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(self.save_image_path, image)
                print(f"[INFO] write image to {self.save_image_path}")

            def callback_set_save_image_path(sender, app_data):
                self.save_image_path = app_data

            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="save image",
                    tag="_button_save_image",
                    callback=callback_save_image,
                )
                dpg.bind_item_theme("_button_save_image", theme_button)

                dpg.add_input_text(
                    label="",
                    default_value=self.save_image_path,
                    callback=callback_set_save_image_path,
                )

            # save json
            def callback_save_json(sender, app_data):
                self.skel.write_json(self.save_json_path)
                print(f"[INFO] write json to {self.save_json_path}")

            def callback_set_save_json_path(sender, app_data):
                self.save_json_path = app_data

            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="save json",
                    tag="_button_save_json",
                    callback=callback_save_json,
                )
                dpg.bind_item_theme("_button_save_json", theme_button)

                dpg.add_input_text(
                    label="",
                    default_value=self.save_json_path,
                    callback=callback_set_save_json_path,
                )

            # pan/scale mode
            def callback_set_pan_scale_mode(sender, app_data):
                self.pan_scale_skel = not self.pan_scale_skel

            dpg.add_checkbox(
                label="pan/scale skeleton",
                default_value=self.pan_scale_skel,
                callback=callback_set_pan_scale_mode,
            )

            # backview mode
            def callback_set_occlusion_mode(sender, app_data):
                self.enable_occlusion = not self.enable_occlusion
                self.need_update = True

            dpg.add_checkbox(
                label="use occlusion",
                default_value=self.enable_occlusion,
                callback=callback_set_occlusion_mode,
            )

            # fov slider
            def callback_set_fovy(sender, app_data):
                self.cam.fovy = app_data
                self.need_update = True

            dpg.add_slider_int(
                label="FoV (vertical)",
                min_value=1,
                max_value=120,
                format="%d deg",
                default_value=self.cam.fovy,
                callback=callback_set_fovy,
            )

            # mode combo
            def callback_change_mode(sender, app_data):
                self.mode = app_data
                self.need_update = True

            dpg.add_combo(
                ("gs", "mesh", "skel"),
                label="mode",
                default_value=self.mode,
                callback=callback_change_mode,
            )

            # play the sequence
            def callback_play(sender, app_data):
                if self.playing:
                    self.playing = False
                    dpg.configure_item("_button_play", label="start")
                else:
                    self.playing = True
                    dpg.configure_item("_button_play", label="stop")

            dpg.add_button(label="start", tag="_button_play", callback=callback_play)
            dpg.bind_item_theme("_button_play", theme_button)

            # SMPLX pose editing
            with dpg.collapsing_header(label="SMPLX body_pose", default_open=False):

                def callback_update_body_pose(sender, app_data, user_data):
                    self.skel.body_pose[user_data] = app_data[:3]
                    self.skel.load_smplx(self.opt.smplx_path)
                    self.need_update = True

                for i in range(self.skel.body_pose.shape[0]):
                    dpg.add_input_floatx(
                        default_value=self.skel.body_pose[i],
                        size=3,
                        width=200,
                        format="%.3f",
                        on_enter=False,
                        callback=callback_update_body_pose,
                        user_data=i,
                    )

            with dpg.collapsing_header(
                label="SMPLX left_hand_pose", default_open=False
            ):

                def callback_update_left_hand_pose(sender, app_data, user_data):
                    self.skel.left_hand_pose[user_data] = app_data[:3]
                    self.skel.load_smplx(self.opt.smplx_path)
                    self.need_update = True

                for i in range(self.skel.left_hand_pose.shape[0]):
                    dpg.add_input_floatx(
                        default_value=self.skel.left_hand_pose[i],
                        size=3,
                        width=200,
                        format="%.3f",
                        on_enter=False,
                        callback=callback_update_left_hand_pose,
                        user_data=i,
                    )

            with dpg.collapsing_header(
                label="SMPLX right_hand_pose", default_open=False
            ):

                def callback_update_right_hand_pose(sender, app_data, user_data):
                    self.skel.right_hand_pose[user_data] = app_data[:3]
                    self.skel.load_smplx(self.opt.smplx_path)
                    self.need_update = True

                for i in range(self.skel.right_hand_pose.shape[0]):
                    dpg.add_input_floatx(
                        default_value=self.skel.right_hand_pose[i],
                        size=3,
                        width=200,
                        format="%.3f",
                        on_enter=False,
                        callback=callback_update_right_hand_pose,
                        user_data=i,
                    )

        ### register camera handler

        def callback_camera_drag_rotate(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            if self.pan_scale_skel:
                self.skel.scale(delta)
            else:
                self.cam.scale(delta)

            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            if self.pan_scale_skel:
                self.skel.pan(self.cam.rot, dx, dy)
            else:
                self.cam.pan(dx, dy)

            self.need_update = True

        def callback_set_mouse_loc(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            # just the pixel coordinate in image
            self.mouse_loc = np.array(app_data)

        def callback_skel_select(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            # determine the selected keypoint from mouse_loc
            if self.points2D is None:
                return  # not prepared

            dist = np.linalg.norm(self.points2D - self.mouse_loc, axis=1)  # [18]
            self.point_idx = np.argmin(dist)

        def callback_skel_drag(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            # 2D to 3D delta
            dx = app_data[1]
            dy = app_data[2]

            self.skel.points3D[self.point_idx, :3] += (
                self.drag_sensitivity
                * self.cam.rot.as_matrix()[:3, :3]
                @ np.array([dx, -dy, 0])
            )

            self.need_update = True

        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

            # for skeleton editing
            dpg.add_mouse_move_handler(callback=callback_set_mouse_loc)
            dpg.add_mouse_click_handler(
                button=dpg.mvMouseButton_Right, callback=callback_skel_select
            )
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Right, callback=callback_skel_drag
            )

        dpg.create_viewport(
            title="pose viewer", resizable=False, width=self.W + 600, height=self.H
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)
        dpg.focus_item("_primary_window")

        dpg.setup_dearpygui()

        # dpg.show_metrics()

        dpg.show_viewport()

    def render(self):
        assert self.gui
        while dpg.is_dearpygui_running():
            self.step()
            dpg.render_dearpygui_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=str, help="path to gaussians ply")
    parser.add_argument("--motion", type=str, help="path to mition file")
    parser.add_argument(
        "--smplx_path",
        type=str,
        default="smplx_models",
        help="path to models folder (contains smplx/)",
    )
    parser.add_argument(
        "--save", type=str, default="videos", help="path to render and save video"
    )

    parser.add_argument("--rotate", action="store_true", help="rotate during rendering")
    parser.add_argument(
        "--play", action="store_true", help="play the motion during rendering"
    )

    parser.add_argument("--W", type=int, default=800, help="GUI width")
    parser.add_argument("--H", type=int, default=800, help="GUI height")
    parser.add_argument("--gui", action="store_true", help="enable GUI")

    parser.add_argument(
        "--radius", type=float, default=2, help="default GUI camera radius from center"
    )
    parser.add_argument(
        "--fovy", type=float, default=50, help="default GUI camera fovy"
    )

    opt = parser.parse_args()

    name = (
        os.path.splitext(os.path.basename(opt.ply))[0]
        + "_"
        + os.path.splitext(os.path.basename(opt.motion))[0]
    )

    gui = GUI(opt)

    print(f"[INFO] load smplx from {opt.smplx_path}")
    gui.skel.load_smplx(opt.smplx_path)
    gui.need_update = True

    if not opt.gui:
        os.makedirs(opt.save, exist_ok=True)

        import imageio

        images = []

        elevation = 0
        azimuth = np.arange(0, 360, 1, dtype=np.int32)
        rotation_len = len(azimuth)

        gui.playing = opt.play
        motion_len = len(gui.skel.motion_seq)

        total_len = min(motion_len, rotation_len)

        for i in tqdm.trange(total_len):
            if opt.rotate:
                gui.cam.from_angle(elevation, azimuth[i % rotation_len])
                gui.need_update = True

            gui.step()

            if opt.gui:
                dpg.render_dearpygui_frame()

            image = (gui.render_buffer * 255).astype(np.uint8)
            images.append(image)

        images = np.stack(images, axis=0)
        # ~6 seconds, 180 frames at 30 fps
        imageio.mimwrite(os.path.join(opt.save, f"{name}.mp4"), images, fps=30)

    else:
        gui.render()
