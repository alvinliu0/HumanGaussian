from dataclasses import dataclass, field
import torch
import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.scene import Scene, GaussianModel
from gaussiansplatting.arguments import ModelParams, PipelineParams, get_combined_args,OptimizationParams
from gaussiansplatting.scene.cameras import Camera, MiniCam
from argparse import ArgumentParser, Namespace
import os
from pathlib import Path
from plyfile import PlyData, PlyElement
from gaussiansplatting.utils.sh_utils import SH2RGB
from gaussiansplatting.scene.gaussian_model import BasicPointCloud
import numpy as np
# from shap_e.diffusion.sample import sample_latents
# from shap_e.diffusion.gaussian_diffusion import diffusion_from_config as diffusion_from_config_shape
# from shap_e.models.download import load_model, load_config
# from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
# from shap_e.util.notebooks import decode_latent_mesh
import io  
from PIL import Image  
# import open3d as o3d

from threestudio.utils.poser import Skeleton
import torch.nn.functional as F

def load_ply(path,save_path):
    C0 = 0.28209479177387814
    def SH2RGB(sh):
        return sh * C0 + 0.5
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
    color = SH2RGB(features_dc[:,:,0])

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)
    point_cloud.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_point_cloud(save_path, point_cloud)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


@threestudio.register("gaussiandreamer-system")
class GaussianDreamer(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        radius: float = 4
        texture_structure_joint: bool = False
        controlnet: bool = False
        smplx_path: str = "/path/to/smplx/model"
        pts_num: int = 100000

        disable_hand_densification: bool = False
        hand_radius: float = 0.05
        densify_prune_start_step: int = 300
        densify_prune_end_step: int = 2100
        densify_prune_interval: int = 300
        size_threshold: int = 20
        size_threshold_fix_step: int = 1500
        half_scheduler_max_step: int = 1500
        max_grad: float = 0.0002
        gender: str = 'neutral'
        prune_only_start_step: int = 2400
        prune_only_end_step: int = 3300
        prune_only_interval: int = 300
        prune_size_threshold: float = 0.008

        apose: bool = True
        bg_white: bool = False

    cfg: Config
    def configure(self) -> None:
        self.radius = self.cfg.radius
        self.gaussian = GaussianModel(sh_degree = 0)
        self.background_tensor = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda") if self.cfg.bg_white else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

        self.parser = ArgumentParser(description="Training script parameters")
        self.pipe = PipelineParams(self.parser)

        self.texture_structure_joint = self.cfg.texture_structure_joint
        self.controlnet = self.cfg.controlnet

        if self.texture_structure_joint:
            # skel
            self.skel = Skeleton(humansd_style=True, apose=self.cfg.apose)
            # self.skel.load_json('17point')
            self.skel.load_smplx(self.cfg.smplx_path, gender=self.cfg.gender)
            self.skel.scale(-10)
        else:
            # skel
            self.skel = Skeleton(apose=self.cfg.apose)
            # self.skel.load_json('8head')
            self.skel.load_smplx(self.cfg.smplx_path, gender=self.cfg.gender)
            self.skel.scale(-10)
        
        self.cameras_extent = 4.0
    
    def save_gif_to_file(self,images, output_file):  
        with io.BytesIO() as writer:  
            images[0].save(  
                writer, format="GIF", save_all=True, append_images=images[1:], duration=100, loop=0  
            )  
            writer.seek(0)  
            with open(output_file, 'wb') as file:  
                file.write(writer.read())
    
    def shape(self):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        xm = load_model('transmitter', device=device)
        model = load_model('text300M', device=device)
        model.load_state_dict(torch.load('./load/shapE_finetuned_with_330kdata.pth', map_location=device)['model_state_dict'])
        diffusion = diffusion_from_config_shape(load_config('diffusion'))

        batch_size = 1
        guidance_scale = 15.0
        prompt = str(self.cfg.prompt_processor.prompt)
        print('prompt',prompt)

        latents = sample_latents(
            batch_size=batch_size,
            model=model,
            diffusion=diffusion,
            guidance_scale=guidance_scale,
            model_kwargs=dict(texts=[prompt] * batch_size),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=64,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )
        render_mode = 'nerf' # you can change this to 'stf'
        size = 256 # this is the size of the renders; higher values take longer to render.

        cameras = create_pan_cameras(size, device)

        self.shapeimages = decode_latent_images(xm, latents[0], cameras, rendering_mode=render_mode)

        pc = decode_latent_mesh(xm, latents[0]).tri_mesh()

        skip = 4
        coords = pc.verts
        rgb = np.concatenate([pc.vertex_channels['R'][:,None],pc.vertex_channels['G'][:,None],pc.vertex_channels['B'][:,None]],axis=1) 

        coords = coords[::skip]
        rgb = rgb[::skip]

        self.num_pts = coords.shape[0]
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(coords)
        point_cloud.colors = o3d.utility.Vector3dVector(rgb)
        self.point_cloud = point_cloud

        return coords,rgb,0.4
    
    def add_points(self,coords,rgb):
        pcd_by3d = o3d.geometry.PointCloud()
        pcd_by3d.points = o3d.utility.Vector3dVector(np.array(coords))
        
        bbox = pcd_by3d.get_axis_aligned_bounding_box()
        np.random.seed(0)

        num_points = 1000000  
        points = np.random.uniform(low=np.asarray(bbox.min_bound), high=np.asarray(bbox.max_bound), size=(num_points, 3))

        kdtree = o3d.geometry.KDTreeFlann(pcd_by3d)

        points_inside = []
        color_inside= []
        for point in points:
            _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
            nearest_point = np.asarray(pcd_by3d.points)[idx[0]]
            if np.linalg.norm(point - nearest_point) < 0.01:  # 这个阈值可能需要调整
                points_inside.append(point)
                color_inside.append(rgb[idx[0]]+0.2*np.random.random(3))

        all_coords = np.array(points_inside)
        all_rgb = np.array(color_inside)
        all_coords = np.concatenate([all_coords,coords],axis=0)
        all_rgb = np.concatenate([all_rgb,rgb],axis=0)
        return all_coords,all_rgb
    
    def pcb(self):
        # Since this data set has no colmap data, we start with random points

        # coords,rgb,scale = self.shape()
        # bound= self.radius*scale
        # all_coords,all_rgb = self.add_points(coords,rgb)
        # pcd = BasicPointCloud(points=all_coords *bound, colors=all_rgb, normals=np.zeros((self.num_pts, 3)))

        points = self.skel.sample_smplx_points(N=self.cfg.pts_num)
        colors = np.ones_like(points) * 0.5
        pcd = BasicPointCloud(points, colors, None)

        return pcd
    
    def forward(self, batch: Dict[str, Any],renderbackground = None) -> Dict[str, Any]:

        if renderbackground is None:
            renderbackground = self.background_tensor
            
        images = []
        depths = []
        pose_images = []
        self.viewspace_point_list = []

        for id in range(batch['c2w'].shape[0]):
       
            viewpoint_cam  = Camera(c2w = batch['c2w'][id],FoVy = batch['fovy'][id],height = batch['height'],width = batch['width'])

            render_pkg = render(viewpoint_cam, self.gaussian, self.pipe, renderbackground)
            image, viewspace_point_tensor, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["radii"]
            self.viewspace_point_list.append(viewspace_point_tensor)

            # manually accumulate max radii across batch
            if id == 0:
                self.radii = radii
            else:
                self.radii = torch.max(radii, self.radii)
                
            depth = render_pkg["depth_3dgs"]

            # import kiui
            # kiui.vis.plot_image(image)

            depth = depth.permute(1, 2, 0)
            image = image.permute(1, 2, 0)
            images.append(image)
            depths.append(depth)

            if self.texture_structure_joint:
                backview = abs(batch['azimuth'][id]) > 120 * np.pi / 180
                mvp = batch['mvp_mtx'][id].detach().cpu().numpy()  # [4, 4]
                pose_image, _ = self.skel.humansd_draw(mvp, 512, 512, backview) # [512, 512, 3], fixed pose image resolution
                # kiui.vis.plot_image(pose_image)
                pose_image = torch.from_numpy(pose_image).to(self.device) # [H, W, 3]
                pose_images.append(pose_image)
            else:
                # render pose image
                backview = abs(batch['azimuth'][id]) > 120 * np.pi / 180
                mvp = batch['mvp_mtx'][id].detach().cpu().numpy()  # [4, 4]
                pose_image, _ = self.skel.draw(mvp, 512, 512, backview) # [512, 512, 3], fixed pose image resolution
                # kiui.vis.plot_image(pose_image)
                pose_image = torch.from_numpy(pose_image).to(self.device) # [H, W, 3]
                pose_images.append(pose_image)


        images = torch.stack(images, 0)
        depths = torch.stack(depths, 0)
        pose_images = torch.stack(pose_images, 0)

        self.visibility_filter = self.radii > 0.0

        # mask near-hand points from visibility_filter, since we don't want to densify them
        if self.cfg.disable_hand_densification:
            points = self.gaussian.get_xyz # [N, 3]
            hand_centers = torch.from_numpy(self.skel.hand_centers).to(points.dtype).to(points.device) # [2, 3]
            distance = torch.norm(points[:, None, :] - hand_centers[None, :, :], dim=-1) # [N, 2]
            hand_mask = distance.min(dim=-1).values < self.cfg.hand_radius # [N]
            self.visibility_filter = self.visibility_filter & (~hand_mask)

        render_pkg["comp_rgb"] = images
        render_pkg["depth"] = depths
        render_pkg['pose'] = pose_images
        render_pkg["opacity"] = depths / (depths.max() + 1e-5)

        return {
            **render_pkg,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
    
    def training_step(self, batch, batch_idx):

        self.gaussian.update_learning_rate(self.true_global_step)
        
        if self.true_global_step > self.cfg.half_scheduler_max_step:
            self.guidance.set_min_max_steps(min_step_percent=0.02, max_step_percent=0.55)

        self.gaussian.update_learning_rate(self.true_global_step)

        out = self(batch) 

        prompt_utils = self.prompt_processor()
        images = out["comp_rgb"]
        depth_images = out['depth']
        depth_min = torch.amin(depth_images, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_images, dim=[1, 2, 3], keepdim=True)
        depth_images = (depth_images - depth_min) / (depth_max - depth_min + 1e-10)# to [0, 1]
        depth_images = depth_images.repeat(1, 1, 1, 3)# to 3-channel
        control_images = out['pose']

        # guidance_eval = (self.true_global_step % 200 == 0)
        guidance_eval = False
        
        if self.texture_structure_joint:
            guidance_out = self.guidance(
                control_images, images, depth_images, prompt_utils, **batch, 
                rgb_as_latents=False, guidance_eval=guidance_eval
            )
        elif self.controlnet:
            guidance_out = self.guidance(
                control_images, images, prompt_utils, **batch, 
                rgb_as_latents=False, guidance_eval=guidance_eval
            )
        else:
            guidance_out = self.guidance(
                images, prompt_utils, **batch, 
                rgb_as_latents=False, guidance_eval=guidance_eval
            )

        loss = 0.0

        loss = loss + guidance_out['loss_sds'] *self.C(self.cfg.loss['lambda_sds'])
        
        loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log("train/loss_opaque", loss_opaque)
        loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)
        if guidance_eval:
            self.guidance_evaluation_save(
                out["comp_rgb"].detach()[: guidance_out["eval"]["bs"]],
                guidance_out["eval"],
            )
        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))
        return {"loss": loss}



    def on_before_optimizer_step(self, optimizer):

        # return

        with torch.no_grad():
            
            if self.true_global_step < self.cfg.densify_prune_end_step: # 15000
                viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
                for idx in range(len(self.viewspace_point_list)):
                    viewspace_point_tensor_grad = viewspace_point_tensor_grad + self.viewspace_point_list[idx].grad
                # Keep track of max radii in image-space for pruning
                self.gaussian.max_radii2D[self.visibility_filter] = torch.max(self.gaussian.max_radii2D[self.visibility_filter], self.radii[self.visibility_filter])
                
                self.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.visibility_filter)

                if self.true_global_step > self.cfg.densify_prune_start_step and self.true_global_step % self.cfg.densify_prune_interval == 0: # 500 100
                    size_threshold = self.cfg.size_threshold if self.true_global_step > self.cfg.size_threshold_fix_step else None # 3000
                    self.gaussian.densify_and_prune(self.cfg.max_grad , 0.05, self.cameras_extent, size_threshold) 

            # prune-only phase according to Gaussian size, rather than the stochastic gradient to eliminate floating artifacts.
            if self.true_global_step > self.cfg.prune_only_start_step and self.true_global_step < self.cfg.prune_only_end_step:
                viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
                for idx in range(len(self.viewspace_point_list)):
                    viewspace_point_tensor_grad = viewspace_point_tensor_grad + self.viewspace_point_list[idx].grad
                # Keep track of max radii in image-space for pruning
                self.gaussian.max_radii2D[self.visibility_filter] = torch.max(self.gaussian.max_radii2D[self.visibility_filter], self.radii[self.visibility_filter])
                
                self.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.visibility_filter)

                if self.true_global_step % self.cfg.prune_only_interval == 0:
                    self.gaussian.prune_only(size_thresh=self.cfg.prune_size_threshold)

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "rgb" in batch
                else []
            )
            + [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            ),
            name="validation_step",
            step=self.true_global_step,
        )
        # save_path = self.get_save_path(f"it{self.true_global_step}-val.ply")
        # self.gaussian.save_ply(save_path)
        # load_ply(save_path,self.get_save_path(f"it{self.true_global_step}-val-color.ply"))

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        only_rgb = True
        bg_color = [1, 1, 1] if self.cfg.bg_white else [0, 0, 0]

        testbackground_tensor = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        out = self(batch,testbackground_tensor)
        if only_rgb:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": batch["rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb" in batch
                    else []
                )
                + [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                ),
                name="test_step",
                step=self.true_global_step,
            )
        else:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": batch["rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb" in batch
                    else []
                )
                + [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "grayscale",
                            "img": out["depth"][0],
                            "kwargs": {},
                        }
                    ]
                    if "depth" in out
                    else []
                )
                + [
                    {
                        "type": "grayscale",
                        "img": out["opacity"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ],
                name="test_step",
                step=self.true_global_step,
            )


    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
        save_path = self.get_save_path(f"last.ply")
        self.gaussian.save_ply(save_path)
        # self.pointefig.savefig(self.get_save_path("pointe.png"))
        # o3d.io.write_point_cloud(self.get_save_path("shape.ply"), self.point_cloud)
        # self.save_gif_to_file(self.shapeimages, self.get_save_path("shape.gif"))
        # load_ply(save_path,self.get_save_path(f"it{self.true_global_step}-test-color.ply"))
        

    def configure_optimizers(self):
        opt = OptimizationParams(self.parser)

        point_cloud = self.pcb()
        self.gaussian.create_from_pcd(point_cloud, self.cameras_extent)

        self.gaussian.training_setup(opt)
        
        ret = {
            "optimizer": self.gaussian.optimizer,
        }

        return ret
    
    def guidance_evaluation_save(self, comp_rgb, guidance_eval_out):
        B, size = comp_rgb.shape[:2]
        resize = lambda x: F.interpolate(
            x.permute(0, 3, 1, 2), (size, size), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        filename = f"it{self.true_global_step}-train.png"

        def merge12(x):
            return x.reshape(-1, *x.shape[2:])

        self.save_image_grid(
            filename,
            [
                {
                    "type": "rgb",
                    "img": merge12(comp_rgb),
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_noisy"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_1step"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_1orig"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_final"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["midas_depth_imgs_noisy"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["midas_depth_imgs_1step"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["midas_depth_imgs_1orig"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["midas_depth_imgs_final"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            ),
            name="train_step",
            step=self.true_global_step,
            texts=guidance_eval_out["texts"],
        )