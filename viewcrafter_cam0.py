import sys
sys.path.append('./extern/dust3r')
from dust3r.inference import inference, load_model
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import to_numpy
import torch
import pytorch3d
import numpy as np
import os
import copy
import cv2
import glob
from PIL import Image
from torchvision.utils import save_image
import torch.nn.functional as F
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from utils.diffusion_utils import instantiate_from_config,load_model_checkpoint,image_guided_synthesis
from utils.pvd_utils import *


class ViewCrafterCam0:
    def __init__(self, opts, gradio = False):
        self.opts = opts
        self.device = opts.device
        self.setup_dust3r()
        self.setup_diffusion()
        if not gradio:
            if os.path.isfile(self.opts.image_dir):
                self.images, self.img_ori = self.load_initial_images(image_dir=self.opts.image_dir)
                self.run_dust3r(input_images=self.images)
            elif os.path.isdir(self.opts.image_dir):
                self.images, self.img_ori = self.load_initial_dir(image_dir=self.opts.image_dir)
                self.run_dust3r(input_images=self.images, clean_pc = True)
            else:
                print(f"{self.opts.image_dir} doesn't exist")

    def setup_dust3r(self):
        self.dust3r = load_model(self.opts.model_path, self.device)

    def setup_diffusion(self):
        seed_everything(self.opts.seed)
        config = OmegaConf.load(self.opts.config)
        model_config = config.pop("model", OmegaConf.create())
        model_config['params']['unet_config']['params']['use_checkpoint'] = False
        model = instantiate_from_config(model_config)
        model = model.to(self.device)
        model.cond_stage_model.device = self.device
        model.perframe_ae = self.opts.perframe_ae
        assert os.path.exists(self.opts.ckpt_path), "Error: checkpoint Not Found!"
        model = load_model_checkpoint(model, self.opts.ckpt_path)
        model.eval()
        self.diffusion = model
        h, w = self.opts.height // 8, self.opts.width // 8
        channels = model.model.diffusion_model.out_channels
        n_frames = self.opts.video_length
        self.noise_shape = [self.opts.bs, channels, n_frames, h, w]

    def load_initial_images(self, image_dir):
        images = load_images([image_dir], size=512,force_1024 = True)
        img_ori = (images[0]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2.
        if len(images) == 1:
            images = [images[0], copy.deepcopy(images[0])]
            images[1]['idx'] = 1
        return images, img_ori

    def load_initial_dir(self, image_dir):
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.PNG', '.JPG', '.JPEG', '.BMP', '.TIFF', '.TIF'}
        candidates = glob.glob(os.path.join(image_dir, "*"))
        image_files = []
        for p in candidates:
            base = p.split('/')[-1]
            name, ext = os.path.splitext(base)
            if ext not in exts:
                continue
            try:
                _ = int(name)
            except Exception:
                continue
            image_files.append(p)
        if len(image_files) < 2:
            raise ValueError("Input views should not less than 2 valid numeric-named images.")
        image_files = sorted(image_files, key=lambda x: int(os.path.splitext(x.split('/')[-1])[0]))
        images = load_images(image_files, size=512, force_1024=True)
        img_gts = []
        for i in range(len(image_files)):
            img_gts.append((images[i]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2.)
        return images, img_gts

    def run_dust3r(self, input_images,clean_pc = False):
        pairs = make_pairs(input_images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, self.dust3r, self.device, batch_size=self.opts.batch_size)
        mode = GlobalAlignerMode.PointCloudOptimizer
        scene = global_aligner(output, device=self.device, mode=mode)
        if mode == GlobalAlignerMode.PointCloudOptimizer:
            _ = scene.compute_global_alignment(init='mst', niter=self.opts.niter, schedule=self.opts.schedule, lr=self.opts.lr)
        if clean_pc:
            self.scene = scene.clean_pointcloud()
        else:
            self.scene = scene

    def run_render(self, pcd, imgs, masks, H, W, camera_traj, num_views):
        render_setup = setup_renderer(camera_traj, image_size=(H,W))
        renderer = render_setup['renderer']
        images, _ = self.render_pcd(pcd, imgs, masks, num_views, renderer, self.device)
        return images

    def render_pcd(self, pts3d, imgs, masks, views, renderer, device):
        imgs = to_numpy(imgs)
        pts3d = to_numpy(pts3d)
        if masks is None:
            pts = torch.from_numpy(np.concatenate([p for p in pts3d])).view(-1, 3).to(device)
            col = torch.from_numpy(np.concatenate([p for p in imgs])).view(-1, 3).to(device)
        else:
            pts = torch.from_numpy(np.concatenate([p[m] for p, m in zip(pts3d, masks)])).to(device)
            col = torch.from_numpy(np.concatenate([p[m] for p, m in zip(imgs, masks)])).to(device)
        point_cloud = pytorch3d.structures.Pointclouds(points=[pts], features=[col]).extend(views)
        images = renderer(point_cloud)
        return images, None

    def run_diffusion(self, renderings):
        prompts = [self.opts.prompt]
        videos = (renderings * 2. - 1.).permute(3,0,1,2).unsqueeze(0).to(self.device)
        condition_index = [0]
        with torch.no_grad(), torch.cuda.amp.autocast():
            batch_samples = image_guided_synthesis(
                self.diffusion, prompts, videos, self.noise_shape, self.opts.n_samples, self.opts.ddim_steps, self.opts.ddim_eta,
                self.opts.unconditional_guidance_scale, self.opts.cfg_img, self.opts.frame_stride, self.opts.text_input,
                self.opts.multiple_cond_cfg, self.opts.timestep_spacing, self.opts.guidance_rescale, condition_index
            )
        return torch.clamp(batch_samples[0][0].permute(1,2,3,0), -1., 1.)

    def _save_png(self, img_tensor_01, path):
        img_uint8 = (img_tensor_01.cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(path, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))

    def _load_real_camera_poses(self, yaml_path):
        """从YAML文件加载真实相机位姿"""
        import yaml
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        real_poses = []
        for cam_name in ['cam0', 'cam1', 'cam2']:
            if cam_name in data:
                # 使用 T_cam_to_world_4x4 (camera to world)
                pose_matrix = torch.tensor(data[cam_name]['T_cam_to_world_4x4'], dtype=torch.float32)
                real_poses.append(pose_matrix)
        
        return torch.stack(real_poses, dim=0)  # [3, 4, 4]
    
    def _estimate_transform_matrix(self, real_poses, estimated_poses):
        """估计真实位姿到估计位姿的变换矩阵"""
        # real_poses: [3, 4, 4] - 真实相机位姿
        # estimated_poses: [3, 4, 4] - DUSt3R估计的相机位姿
        
        # 提取旋转和平移部分
        R_real = real_poses[:, :3, :3]  # [3, 3, 3]
        t_real = real_poses[:, :3, 3]  # [3, 3]
        
        R_est = estimated_poses[:, :3, :3]  # [3, 3, 3]
        t_est = estimated_poses[:, :3, 3]  # [3, 3]
        
        # 方法1: 使用第一个相机作为参考点，计算相对变换
        # 选择cam0作为参考
        T_real_ref = real_poses[0]  # cam0的真实位姿
        T_est_ref = estimated_poses[0]  # cam0的估计位姿
        
        # 计算从真实cam0到估计cam0的变换
        # T_est_ref = T_transform @ T_real_ref
        # 所以 T_transform = T_est_ref @ T_real_ref^(-1)
        T_real_ref_inv = torch.inverse(T_real_ref)
        T_transform = T_est_ref @ T_real_ref_inv
        
        # 验证变换是否正确：检查其他相机
        print("验证变换矩阵:")
        print(f"真实cam0位姿:\n{real_poses[0]}")
        print(f"估计cam0位姿:\n{estimated_poses[0]}")
        print(f"变换矩阵:\n{T_transform}")
        
        # 测试：单位矩阵转换后应该等于估计cam0位姿
        identity = torch.eye(4, dtype=torch.float32, device=real_poses.device)
        transformed_identity = T_transform @ identity
        print(f"单位矩阵转换后:\n{transformed_identity}")
        print(f"与估计cam0的差异:\n{transformed_identity - estimated_poses[0]}")
        
        for i in range(3):
            T_real_i = real_poses[i]
            T_est_i = estimated_poses[i]
            T_transformed_i = T_transform @ T_real_i
            
            # 计算变换误差
            translation_error = torch.norm(T_transformed_i[:3, 3] - T_est_i[:3, 3])
            rotation_error = torch.norm(T_transformed_i[:3, :3] - T_est_i[:3, :3])
            
            print(f"  Camera {i}:")
            print(f"    平移误差: {translation_error:.6f}")
            print(f"    旋转误差: {rotation_error:.6f}")
        
        # 计算缩放因子（基于相机间距离）
        def compute_scale_factor(poses1, poses2):
            scales = []
            for i in range(len(poses1)):
                for j in range(i+1, len(poses1)):
                    dist1 = torch.norm(poses1[i][:3, 3] - poses1[j][:3, 3])
                    dist2 = torch.norm(poses2[i][:3, 3] - poses2[j][:3, 3])
                    if dist1 > 1e-6:
                        scales.append(dist2 / dist1)
            return torch.mean(torch.stack(scales)) if scales else 1.0
        
        scale_factor = compute_scale_factor(real_poses, estimated_poses)
        
        return T_transform, scale_factor
    
    def _transform_external_pose(self, external_pose, transform_matrix):
        """将外部位姿转换到DUSt3R坐标系"""
        # external_pose: [4, 4] - 输入的外部位姿
        # transform_matrix: [4, 4] - 真实到估计的变换矩阵
        # 返回转换后的位姿
        
        transformed_pose = transform_matrix @ external_pose
        return transformed_pose

    def _interp_pose(self, T0, T1, alpha):
        # 简单线性插值旋转的欧拉角与平移（近似）：
        # 对实际项目，可替换为 Slerp 四元数插值。此处保持轻量实现。
        import torch
        R0, t0 = T0[:3,:3], T0[:3,3]
        R1, t1 = T1[:3,:3], T1[:3,3]
        # 旋转矩阵转为欧拉角（ZYX），线性插值再回转矩阵
        def rot_to_euler_zyx(R):
            sy = torch.sqrt(R[0,0]**2 + R[1,0]**2)
            singular = sy < 1e-6
            if not singular:
                z = torch.atan2(R[1,0], R[0,0])
                y = torch.atan2(-R[2,0], sy)
                x = torch.atan2(R[2,1], R[2,2])
            else:
                z = torch.atan2(-R[0,1], R[1,1])
                y = torch.atan2(-R[2,0], sy)
                x = torch.tensor(0., device=R.device)
            return torch.stack([z,y,x])
        def euler_zyx_to_rot(e):
            z,y,x = e[0],e[1],e[2]
            cz, sz = torch.cos(z), torch.sin(z)
            cy, sy = torch.cos(y), torch.sin(y)
            cx, sx = torch.cos(x), torch.sin(x)
            Rz = torch.tensor([[cz,-sz,0.],[sz,cz,0.],[0.,0.,1.]], device=e.device)
            Ry = torch.tensor([[cy,0.,sy],[0.,1.,0.],[-sy,0.,cy]], device=e.device)
            Rx = torch.tensor([[1.,0.,0.],[0.,cx,-sx],[0.,sx,cx]], device=e.device)
            return Rz @ Ry @ Rx
        e0 = rot_to_euler_zyx(R0)
        e1 = rot_to_euler_zyx(R1)
        e = (1-alpha)*e0 + alpha*e1
        R = euler_zyx_to_rot(e)
        t = (1-alpha)*t0 + alpha*t1
        T = torch.eye(4, device=T0.device)
        T[:3,:3] = R
        T[:3,3] = t
        return T

    def nvs_external_pose_cam0(self):
        import json, torch
        c2ws = self.scene.get_im_poses().detach()  # [N,4,4] - DUSt3R估计的位姿
        principal_points = self.scene.get_principal_points().detach()
        focals = self.scene.get_focals().detach()
        shape = self.images[0]['true_shape']
        H, W = int(shape[0][0]), int(shape[0][1])
        pcd = [i.detach() for i in self.scene.get_pts3d(clip_thred=self.opts.dpt_trd)]
        imgs = np.array(self.scene.imgs)

        # 1) 加载真实相机位姿并估计变换矩阵
        yaml_path = self.opts.real_poses_yaml if hasattr(self.opts, 'real_poses_yaml') else '/home/nics/Workspace/ViewCrafter/image_process/cam_extrinsics.yaml'
        if os.path.exists(yaml_path):
            real_poses = self._load_real_camera_poses(yaml_path).to(self.device)
            print("真实相机位姿:")
            for i, pose in enumerate(real_poses):
                print(f"  cam{i}:")
                print(f"    平移: {pose[:3, 3]}")
                print(f"    旋转矩阵: {pose[:3, :3]}")
            
            print("\nDUSt3R估计位姿:")
            for i, pose in enumerate(c2ws):
                print(f"  cam{i}:")
                print(f"    平移: {pose[:3, 3]}")
                print(f"    旋转矩阵: {pose[:3, :3]}")
            
            transform_matrix, scale_factor = self._estimate_transform_matrix(real_poses, c2ws)
            print(f"估计的变换矩阵:\n{transform_matrix}")
            print(f"缩放因子: {scale_factor}")
        else:
            print("警告: 未找到真实相机位姿文件，使用单位变换矩阵")
            transform_matrix = torch.eye(4, dtype=torch.float32, device=self.device)

        # 2) 确定目标位姿 T_target
        if self.opts.ext_pose_path is not None and os.path.exists(self.opts.ext_pose_path):
            with open(self.opts.ext_pose_path, 'r') as f:
                T_list = json.load(f)
            T_target_real = torch.tensor(T_list, dtype=torch.float32, device=self.device)
            if T_target_real.shape != (4,4):
                raise ValueError('ext_pose_path 中的矩阵必须是 4x4')
            # 将真实位姿转换到DUSt3R坐标系
            T_target = self._transform_external_pose(T_target_real, transform_matrix)
            print(f"原始外部位姿:\n{T_target_real}")
            print(f"转换后的位姿:\n{T_target}")
        elif self.opts.interp_alpha is not None:
            if c2ws.shape[0] < 2:
                raise ValueError('插值需要至少 2 个相机位姿')
            alpha = float(self.opts.interp_alpha)
            if not (0.0 < alpha < 1.0):
                raise ValueError('interp_alpha 需在 (0,1) 内')
            T_target = self._interp_pose(c2ws[0], c2ws[1], alpha)
        else:
            raise ValueError('需指定 --ext_pose_path 或 --interp_alpha 才能使用 external_pose 模式')

        # diffusion
        # INSERT_YOUR_CODE
        # 假定c2ws是[N,4,4]，N==3，T_target是[4,4]
        # 我们判断T_target在c2ws[0]-c2ws[1]之间还是c2ws[1]-c2ws[2]之间(根据与各段首的距离)
        def pose_dist(a, b):
            # 欧氏平移距离 + 欧氏旋转距离
            t_dist = torch.norm(a[:3,3]-b[:3,3])
            r_dist = torch.norm(a[:3,:3]-b[:3,:3])
            return t_dist + r_dist

        N = c2ws.shape[0]
        if N != 3:
            raise ValueError("当前代码只适用于3个位姿(c2ws.shape[0]==3)")
        # 计算T_target到c2ws前两个位姿的距离
        d0 = pose_dist(T_target, c2ws[0])
        d1 = pose_dist(T_target, c2ws[1])
        d2 = pose_dist(T_target, c2ws[2])
        # 判断T_target在c2ws[0]-c2ws[1], 还是c2ws[1]-c2ws[2]之间
        # 简单策略：距离c2ws[1]最近就插到[1,2]之间，否则插到[0,1]之间
        if d0 <= d2 :
            insert_idx = 0
        else:
            insert_idx = 1 
        # 在合适位置插入T_target
        # 检查T_target是否与现有位姿太接近
        min_distance = float('inf')
        closest_idx = 0
        for i, pose in enumerate(c2ws):
            # 计算位姿间的距离（平移+旋转）
            translation_dist = torch.norm(T_target[:3, 3] - pose[:3, 3])
            rotation_dist = torch.norm(T_target[:3, :3] - pose[:3, :3])
            total_dist = translation_dist + rotation_dist
            if total_dist < min_distance:
                min_distance = total_dist
                closest_idx = i
        
        print(f"T_target与最近位姿的距离: {min_distance:.6f}")
        
        # 如果距离太小，添加一个小的偏移
        if min_distance < 1e-4:
            print("警告: T_target与现有位姿太接近，添加小偏移")
            # 在T_target上添加一个小的平移偏移
            offset = torch.tensor([0.01, 0.01, 0.01], device=T_target.device)
            T_target[:3, 3] += offset
        
        # 只保留target和附近两帧（共3帧），另一帧舍弃
        # 例如假如 insert_idx=1, 则仅保留 [c2ws[insert_idx], T_target, c2ws[insert_idx+1]]
        c2ws_short = torch.stack([c2ws[insert_idx], T_target, c2ws[insert_idx+1]], dim=0)
        c2ws = c2ws_short
        # print(f"c2ws shape: {c2ws.shape}")
        # print(f"c2ws: {c2ws}")
        camera_traj,num_views = generate_traj_interp(c2ws, H, W, focals, principal_points, self.opts.video_length, self.device)
        
        
        render_results = self.run_render(pcd, imgs, None, H, W, camera_traj, num_views)
        render_results = F.interpolate(render_results.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
        render_results[0] = self.img_ori[0] if isinstance(self.img_ori, list) else self.img_ori
        
        # for i in range(len(self.img_ori)):
        #     render_results[i*(self.opts.video_length - 1)] = self.img_ori[i]
        
        render_results[2*(self.opts.video_length - 1)] = self.img_ori[insert_idx+1]

        save_video(render_results, os.path.join(self.opts.save_dir, f'render.mp4'))

        
        diffusion_results = []
        for i in range(len(self.img_ori) - 1 if isinstance(self.img_ori, list) else 1):
            diffusion_results.append(self.run_diffusion(render_results[i*(self.opts.video_length - 1):self.opts.video_length + i*(self.opts.video_length - 1)]))
        
        diffusion_results = torch.cat(diffusion_results) if len(diffusion_results) > 0 else self.run_diffusion(render_results)

        target_idx = self.opts.video_length - 1
        target_frame = (diffusion_results[target_idx] + 1.0) / 2.0
        os.makedirs(self.opts.save_dir, exist_ok=True)
        self._save_png(target_frame, os.path.join(self.opts.save_dir, 'diffusion_t_target.png'))
        save_video((diffusion_results + 1.0) / 2.0, os.path.join(self.opts.save_dir, f'diffusion.mp4'))
        return target_frame

    def nvs_sparse_view_interp_cam0(self):
        c2ws = self.scene.get_im_poses().detach()
        principal_points = self.scene.get_principal_points().detach()
        focals = self.scene.get_focals().detach()
        shape = self.images[0]['true_shape']
        H, W = int(shape[0][0]), int(shape[0][1])
        pcd = [i.detach() for i in self.scene.get_pts3d(clip_thred=self.opts.dpt_trd)]
        imgs = np.array(self.scene.imgs)
        camera_traj,num_views = generate_traj_interp(c2ws, H, W, focals, principal_points, self.opts.video_length, self.device)
        render_results = self.run_render(pcd, imgs, None, H, W, camera_traj, num_views)
        render_results = F.interpolate(render_results.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
        render_results[0] = self.img_ori[0] if isinstance(self.img_ori, list) else self.img_ori
        diffusion_results = []
        for i in range(len(self.img_ori)-1 if isinstance(self.img_ori, list) else 1):
            diffusion_results.append(self.run_diffusion(render_results[i*(self.opts.video_length - 1):self.opts.video_length+i*(self.opts.video_length - 1)]))
        diffusion_results = torch.cat(diffusion_results) if len(diffusion_results)>0 else self.run_diffusion(render_results)
        cam0_frame = (diffusion_results[0] + 1.0) / 2.0
        os.makedirs(self.opts.save_dir, exist_ok=True)
        self._save_png(cam0_frame, os.path.join(self.opts.save_dir, 'diffusion_cam0.png'))
        return cam0_frame

    def nvs_single_view_cam0(self):
        c2ws = self.scene.get_im_poses().detach()[1:]
        principal_points = self.scene.get_principal_points().detach()[1:]
        focals = self.scene.get_focals().detach()[1:]
        shape = self.images[0]['true_shape']
        H, W = int(shape[0][0]), int(shape[0][1])
        pcd = [i.detach() for i in self.scene.get_pts3d(clip_thred=self.opts.dpt_trd)]
        depth = [i.detach() for i in self.scene.get_depthmaps()]
        depth_avg = depth[-1][H//2,W//2]
        radius = depth_avg*self.opts.center_scale
        c2ws,pcd =  world_point_to_obj(poses=c2ws, points=torch.stack(pcd), k=-1, r=radius, elevation=self.opts.elevation, device=self.device)
        imgs = np.array(self.scene.imgs)
        camera_traj,num_views = generate_traj_specified(c2ws, H, W, focals, principal_points, self.opts.d_theta[0], self.opts.d_phi[0], self.opts.d_r[0],self.opts.video_length, self.device)
        render_results = self.run_render([pcd[-1]], [imgs[-1]], None, H, W, camera_traj, num_views)
        render_results = F.interpolate(render_results.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
        render_results[0] = self.img_ori
        diffusion_results = self.run_diffusion(render_results)
        cam0_frame = (diffusion_results[0] + 1.0) / 2.0
        os.makedirs(self.opts.save_dir, exist_ok=True)
        self._save_png(cam0_frame, os.path.join(self.opts.save_dir, 'diffusion_cam0.png'))
        return cam0_frame

    def nvs_single_view_eval_cam0(self):
        c2ws = self.scene.get_im_poses().detach()
        principal_points = self.scene.get_principal_points().detach()
        focals = self.scene.get_focals().detach()
        shape = self.images[0]['true_shape']
        H, W = int(shape[0][0]), int(shape[0][1])
        pcd = [i.detach() for i in self.scene.get_pts3d(clip_thred=self.opts.dpt_trd)]
        c2ws,pcd =  world_point_to_kth(poses=c2ws, points=torch.stack(pcd), k=0, device=self.device)
        camera_traj,num_views = generate_traj(c2ws, H, W, focals, principal_points, self.device)
        images = np.array(self.scene.imgs)
        render_results = self.run_render([pcd[0]], [images[0]], None, H, W, camera_traj, num_views)
        render_results = F.interpolate(render_results.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
        render_results[0] = self.img_ori[0] if isinstance(self.img_ori, list) else self.img_ori
        diffusion_results = self.run_diffusion(render_results)
        cam0_frame = (diffusion_results[0] + 1.0) / 2.0
        os.makedirs(self.opts.save_dir, exist_ok=True)
        self._save_png(cam0_frame, os.path.join(self.opts.save_dir, 'diffusion_cam0.png'))
        return cam0_frame

