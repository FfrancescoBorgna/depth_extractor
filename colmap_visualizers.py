import open3d as o3d
from read_write_model import read_model, BaseImage
import os
import numpy as np
from torch.utils.data import Dataset
import torchvision
import torch
import matplotlib.pyplot as plt
import json
import depth_estimator as de
import cv2
from tqdm import tqdm


import copy
from hovering.o3d_line_mesh import LineMesh
from scipy.spatial.distance import cdist

EPIC_WIDTH = 456
EPIC_HEIGHT = 256

FRUSTUM_SIZE = 0.6
FRUSTUM_LINE_RADIUS = 0.02

TRAJECTORY_LINE_RADIUS = 0.02
def get_frustum(sz=1.0, 
                line_radius=0.15,
                colmap_image: BaseImage = None, 
                camera_height=None,
                camera_width=None,
                frustum_color=[1, 0, 0]) -> o3d.geometry.TriangleMesh:
    """
    Args:
        sz: float, size (width) of the frustum
        colmap_image: ColmapImage, if not None, the frustum will be transformed
            otherwise the frustum will "lookAt" +z direction
    """
    cen = [0, 0, 0]
    wid = sz
    if camera_height is not None and camera_width is not None:
        hei = wid * camera_height / camera_width
    else:
        hei = wid
    tl = [wid, hei, sz]
    tr = [-wid, hei, sz]
    br = [-wid, -hei, sz]
    bl = [wid, -hei, sz]
    points = np.float32([cen, tl, tr, br, bl])
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1],]
    line_mesh = LineMesh(
        points, lines, colors=frustum_color, radius=line_radius)
    line_mesh.merge_cylinder_segments()
    frustum = line_mesh.cylinder_segments[0]

    if colmap_image is not None:
        w2c = np.eye(4)
        w2c[:3, :3] = colmap_image.qvec2rotmat()
        w2c[:3, -1] = colmap_image.tvec
        c2w = np.linalg.inv(w2c)
        frustum = frustum.transform(c2w)
    return frustum

def get_o3d_FOR(origin=[0, 0, 0],size=10):
    """ 
    Create a FOR that can be added to the open3d point cloud
    """
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=size)
    mesh_frame.translate(origin)
    return(mesh_frame)

def load_meta(root, name="meta.json"):
    """Load meta information per scene and frame (nears, fars, poses etc.)."""
    path = os.path.join(root, name)
    with open(path, "r") as fp:
        ds = json.load(fp)
    for k in ["nears", "fars", "images", "poses"]:
        ds[k] = {int(i): ds[k][i] for i in ds[k]}
        if k == "poses":
            ds[k] = {i: np.array(ds[k][i]) for i in ds[k]}
    ds["intrinsics"] = np.array(ds["intrinsics"])
    return ds

def convert_camera2world(points,R,t):
    # R, t are relative to world to camera
    w2c = np.eye(4)
    w2c[:3,:3] = R
    w2c[:3,3] = t

    c2w = np.linalg.inv(w2c)

    #pass to homogeneous coords
    len_p = points.shape[0]
    points_h = np.concatenate((points,np.ones((len_p,1))),axis=-1)

    #camera to world
    points_world = np.dot(c2w, points_h.T).T
    points_world = points_world[:, :3] / points_world[:, 3][:, None]
    
    return points_world  

def get_pcd_statistics(pcd):
        # Compute the average distance
        distances = pcd.compute_point_cloud_distance()
        avg_distance = np.mean(distances)
        return avg_distance

def compute_dist(p1,p2,th):
    v1 = np.asarray(p1)
    v2 = np.asarray(p2)
    idx_list = np.array([],dtype=int)
    for i in tqdm(range(v1.shape[0]),total=v1.shape[0]):
        dist = cdist(np.expand_dims(v1[i,:],axis=0),v2,'euclidean')
        _,c = np.where(dist < th)
        c_unique = np.unique(c)
        idx_list = np.concatenate((idx_list,c_unique))

    idx_list_unique = np.unique(idx_list)
    return idx_list_unique
def dense_segment(dense_pcd,moving_pcd,th,first_time=True,path="data/Epic_converted/P01_01/pcds/indices.npy"):
        ## Method 1
        # dist = cdist(moving_pcd.points,dense_pcd.points,'euclidean')
        # _,c = np.where(dist < th)
        # c_unique = np.unique(c)
        
        #  Method 2
        if(first_time):
            c_unique = compute_dist(moving_pcd.points,dense_pcd.points,0.01)
            np.save(path, c_unique)
        else:
            c_unique = np.load(path)
            
        dense_moving_pcd = o3d.geometry.PointCloud()
        dense_static_pcd = o3d.geometry.PointCloud()

        static_points = np.delete(np.asarray(dense_pcd.points), c_unique, axis=0)
        static_colors = np.delete(np.asarray(dense_pcd.colors), c_unique, axis=0)

        dense_points = np.asarray(dense_pcd.points)
        dense_moving_pcd.points = o3d.utility.Vector3dVector(dense_points[c_unique])
        dense_static_pcd.points = o3d.utility.Vector3dVector(static_points)
        
        red = o3d.utility.Vector3dVector(np.tile(np.array([1,0,0]),(len(c_unique),1)))
        
        dense_moving_pcd.colors = red
        dense_static_pcd.colors = o3d.utility.Vector3dVector(static_colors)
        return dense_moving_pcd, dense_static_pcd
class EPICDiff(Dataset):
    def __init__(self, vid, root="data/EPIC-Diff", split=None):

        self.root = os.path.join(root, vid)
        self.vid = vid
        self.img_w = 228
        self.img_h = 128
        self.split = split
        self.val_num = 1
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), #TODO
            torchvision.transforms.Resize((self.img_h,self.img_w))
        ])
        self.init_meta()

    def imshow(self, index):
        plt.imshow(self.imread(index))
        plt.axis("off")
        plt.show()

    def imread(self, index):
        return plt.imread(os.path.join(self.root, "frames", self.image_paths[index]))

    def x2im(self, x, type_="np"):
        """Convert numpy or torch tensor to numpy or torch 'image'."""
        w = self.img_w
        h = self.img_h
        if len(x.shape) == 2 and x.shape[1] == 3:
            x = x.reshape(h, w, 3)
        else:
            x = x.reshape(h, w)
        if type(x) == torch.Tensor:
            x = x.detach().cpu()
            if type_ == "np":
                x = x.numpy()
        elif type(x) == np.array:
            if type_ == "pt":
                x = torch.from_numpy(x)
        return x

    def rays_per_image(self, idx, pose=None):
        """Return sample with rays, frame index etc."""
        sample = {}
        if pose is None:
            sample["c2w"] = c2w = torch.FloatTensor(self.poses_dict[idx])
        else:
            sample["c2w"] = c2w = pose

        sample["im_path"] = self.image_paths[idx]

        img = Image.open(os.path.join(self.root, "frames", self.image_paths[idx]))
        img_w, img_h = img.size
        img = self.transform(img)  # (3, h, w)
        _,img_h,img_w = img.size() #TODO Check thissss
        img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB

        directions = get_ray_directions(img_h, img_w, self.K)
        rays_o, rays_d = get_rays(directions, c2w)

        c2c = torch.zeros(3, 4).to(c2w.device)
        c2c[:3, :3] = torch.eye(3, 3).to(c2w.device)
        rays_o_c, rays_d_c = get_rays(directions, c2c)

        rays_t = idx * torch.ones(len(rays_o), 1).long()

        rays = torch.cat(
            [
                rays_o,
                rays_d,
                self.nears[idx] * torch.ones_like(rays_o[:, :1]),
                self.fars[idx] * torch.ones_like(rays_o[:, :1]),
                rays_o_c,
                rays_d_c,
            ],
            1,
        )

        sample["rays"] = rays
        sample["img_wh"] = torch.LongTensor([img_w, img_h])
        sample["ts"] = rays_t
        sample["rgbs"] = img

        return sample

    def init_meta(self):
        """Load meta information, e.g. intrinsics, train, test, val split etc."""
        meta = load_meta(self.root)
        self.img_ids = meta["ids_all"]
        self.img_ids_train = meta["ids_train"]
        self.img_ids_test = meta["ids_test"]
        self.img_ids_val = meta["ids_val"]
        self.poses_dict = meta["poses"]
        self.nears = meta["nears"]
        self.fars = meta["fars"]
        self.image_paths = meta["images"]
        self.K = meta["intrinsics"]

        if self.split == "train":
            # create buffer of all rays and rgb data
            self.rays = []
            self.rgbs = []
            self.ts = []

            for idx in self.img_ids_train:
                sample = self.rays_per_image(idx)
                self.rgbs += [sample["rgbs"]]
                self.rays += [sample["rays"]]
                self.ts += [sample["ts"]]

            self.rays = torch.cat(self.rays, 0)  # ((N_images-1)*h*w, 8)
            self.rgbs = torch.cat(self.rgbs, 0)  # ((N_images-1)*h*w, 3)
            self.ts = torch.cat(self.ts, 0)

    def __len__(self):
        if self.split == "train":
            # rays are stored concatenated
            return len(self.rays)
        if self.split == "val":
            # evaluate only one image, sampled from val img ids
            return 1
        else:
            # choose any image index
            return max(self.img_ids)

    def __getitem__(self, idx, pose=None):

        if self.split == "train":
            # samples selected from prefetched train data
            sample = {
                "rays": self.rays[idx],
                "ts": self.ts[idx, 0].long(),
                "rgbs": self.rgbs[idx],
            }

        elif self.split == "val":
            # for tuning hyperparameters, tensorboard samples
            idx = random.choice(self.img_ids_val)
            sample = self.rays_per_image(idx, pose)

        elif self.split == "test":
            # evaluating according to table in paper, chosen index must be in test ids
            assert idx in self.img_ids_test
            sample = self.rays_per_image(idx, pose)

        else:
            # for arbitrary samples, e.g. summary video when rendering over all images
            sample = self.rays_per_image(idx, pose)

        return sample


class COLMAP(Dataset):
    """Superclass for sequential images dataloaders
    """
    def __init__(self, data_path,input_res=[456,256],output_res=[228,128]):
        super(COLMAP, self).__init__()
        #self.colmap_poses = os.path.join(data_path, kitchen,'colmap')
        self.colmap_path = data_path
        
        self.cameras_Colmap, self.imgs_Colmap, self.pts_Colmap = read_model(self.colmap_path, ext=".bin")
        self.fx_orig = self.cameras_Colmap[1].params[0]
        self.fy_orig = self.cameras_Colmap[1].params[1]
        self.fx = self.fx_orig*output_res[0]/input_res[0]
        self.fy = self.fy_orig*output_res[1]/input_res[1]
        self.cx_orig = self.cameras_Colmap[1].params[2]
        self.cy_orig = self.cameras_Colmap[1].params[3]
        self.cx = self.cx_orig*output_res[0]/input_res[0]
        self.cy = self.cy_orig*output_res[1]/input_res[1]
        


    def __len__(self):
        return len(self.imgs_Colmap)

    def __getitem__(self, index):
        inputs = {}
        full_filename = self.filenames[index]
        for i in self.frame_idxs:
            inputs[("color", i)] = self.get_color(self.filenames[index + i])
        inputs["full_filename"] = full_filename
        inputs["filename"] = full_filename.split('/')[-1] 
        print(full_filename.split('/')[-1].split('_')[0:2])
        sequence = full_filename.split('/')[-1].split('_')[0:2]
        #Join the two string elements of the list with a '_' in the middle
        inputs['sequence'] = '_'.join(sequence)
        inputs["subset"] = 'train'
        inputs["aff_annotation"], inputs["EP100_annotation"], inputs['VISOR_annotation'] = self.EP100_and_VISOR_reader.affordance_hotspot(inputs["filename"], inputs['subset'], inputs['sequence'])
        inputs["exists_affordance"] = self.check_exits_affordance(inputs["aff_annotation"])
        return inputs

    def draw_pcd_from_image(self,idx):
        pts_idxs = self.imgs_Colmap[idx].point3D_ids
        pts_idxs = pts_idxs[pts_idxs>0]
        pcd_np_xyz = np.zeros((len(pts_idxs),3))
        pcd_np_rgb = np.zeros((len(pts_idxs),3))
        for i,k in enumerate(pts_idxs):
            pcd_np_xyz[i,:] = self.pts_Colmap[k].xyz
            #pcd_np_rgb[i,:] = self.pts_Colmap[k].rgb
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_np_xyz)
        pcd.colors = o3d.utility.Vector3dVector(pcd_np_rgb)
        o3d.visualization.draw_geometries([pcd])

    def draw_pcd(self):
        p_xyz = np.zeros((len(self.pts_Colmap.keys()),3))
        p_rgb = np.zeros((len(self.pts_Colmap.keys()),3))
        for c,i in enumerate(self.pts_Colmap.keys()):
            p_xyz[c,:] = self.pts_Colmap[i].xyz
            #p_rgb[c,:] = self.pts_Colmap[i].rgb

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(p_xyz)
        pcd.colors = o3d.utility.Vector3dVector(p_rgb)

        #path_ply = "/Users/francesco/Desktop/Università/Tesi/EgoChiara/DenseReconstruction/P01_01/fused.ply"
        #pcd_ply = pcd = o3d.io.read_point_cloud(path_ply)
        #o3d.visualization.draw_geometries([pcd])
  

class motion_estimator():
    def __init__(self,root_data,path_col,path_diff,vid) -> None:
        self.root_data = root_data
        self.vid = vid
        self.pcd_path = os.path.join(self.root_data,self.vid,"pcds")
        self.c = COLMAP(path_col)#,output_res=[456,256])
        self.epic_diff = EPICDiff(vid,path_diff)
        self.depth_est = de.Inference()
        self.width = 228
        self.height = 128
    
    def new_scale_SfM_depth(self, depth, colmap_depths, colmap_coords):
        SfM_depth, NN_depth = [], []
        for kypt in range(len(colmap_coords)):
            SfM_depth.append(colmap_depths[kypt]) #Interpretation 1 of the depth: La distancia entre el plano de la camara y el plano paralelo que corta el punto en 3D
            # Change order in coords, from XY to YX!!!
            u_interp = colmap_coords[kypt, 1] % 1
            v_interp = colmap_coords[kypt, 0] % 1
            u = int(colmap_coords[kypt, 1])
            v = int(colmap_coords[kypt, 0])
            if u < self.width - 1 and v < self.height - 1:
                interpolated_NN_depth = (1 - u_interp) * (1 - v_interp) * depth[v, u] + u_interp * (1 - v_interp) * depth[v, u + 1] + (1 - u_interp) * v_interp * depth[v + 1, u] + u_interp * v_interp * depth[v + 1, u + 1]
                NN_depth.append(interpolated_NN_depth)
            if u > self.width:
                print('alerta 1 !!!', u)
            if v > self.height:
                print('alerta  2 !!!', v)
        local_scale = np.median(np.array(SfM_depth)) / np.median(np.array(NN_depth))
        return local_scale
    
    def obtain_rgbd(self, depth, scale,camera_origin,extrinsic_cam):
        z = depth * scale
        x = (np.tile(np.arange(self.width), (self.height, 1)) - self.c.cx) * z / self.c.fx
        y = (np.tile(np.arange(self.height), (self.width, 1)).T - self.c.cy) * z / self.c.fy
        #points = np.stack([x, y, z], axis=2) #h, w, 3

        x = x.reshape((self.width*self.height,1)) 
        y = y.reshape((self.width*self.height,1)) 
        z = z.reshape((self.width*self.height,1)) 
        points =np.concatenate([x,y,z], axis=1)

        points_w = convert_camera2world(points,extrinsic_cam,camera_origin)
        return points_w
    def obtain_rgbd2(self, depth, scale,camera_origin,extrinsic_cam):
        offset = 0#1.5 / 2700
        z = (depth+offset) * scale
        x = (np.tile(np.arange(self.width), (self.height, 1)) - self.c.cx) * z / self.c.fx
        y = (np.tile(np.arange(self.height), (self.width, 1)).T - self.c.cy) * z / self.c.fy
        #points = np.stack([x, y, z], axis=2) #h, w, 3

        x = x.reshape((self.width*self.height,1)) 
        y = y.reshape((self.width*self.height,1)) 
        z = z.reshape((self.width*self.height,1)) 
        points =np.concatenate([x,y,z], axis=1)

        w2c = np.eye(4)
        w2c[:3, :3] = extrinsic_cam
        w2c[:3, -1] = camera_origin
        c2w = np.linalg.inv(w2c)
        
        return points, c2w
    
    def diff2Col(self,idx):
        try:
            col_idx = [self.c.imgs_Colmap[i].id for i in self.c.imgs_Colmap.keys() if self.c.imgs_Colmap[i].name == self.epic_diff.image_paths[idx]]
        except:
            col_idx = np.array([-1,0])
        return col_idx[0]
    def extract_pcd_by_indx(self,idx):
        #################################################################
        # Input: idx of the diff video (is different from colmap index!!!)
        # ------------------------------------------------------------------------------
        # Output: rescaled_rgbd = pointcloud in world coord
        #         
        #         colmap_keypoints = points of the reconstruction seen from the camera
        #Get Image Depth
        img_path = os.path.join(self.root_data,self.vid,"frames",self.epic_diff.image_paths[idx])
        img = cv2.imread(img_path)
        img = cv2.resize(img,(self.width,self.height))
        depth = self.depth_est.depth_extractor(img,"niente")

        # Diff to COLMAP index translator
        col_idx = self.diff2Col(idx)
    
        #Get corresponding colmap depth
        v = self.c.imgs_Colmap[col_idx]
        colmap_depths = np.array([(v.qvec2rotmat() @ self.c.pts_Colmap[p3d].xyz + v.tvec)[2] for p3d in v.point3D_ids[v.point3D_ids > -1]]) #WE PASS TO CAMERA COORDINATES
        colmap_coords = np.array([v.xys[np.where(v.point3D_ids == p3d)][0, ::-1] for p3d in v.point3D_ids[v.point3D_ids > -1]]) #Depth of the keypoints in the camera coordinates
        colmap_keypoints = np.array([self.c.pts_Colmap[p3d].xyz for p3d in v.point3D_ids[v.point3D_ids > -1]]) #Absolute coordinates
        colmap_rgb = np.array([self.c.pts_Colmap[p3d].rgb for p3d in v.point3D_ids[v.point3D_ids > -1]]) #Absolute coordinates
        colmap_ids = np.array([p3d for p3d in v.point3D_ids[v.point3D_ids > -1]])
        
        local_scale = self.new_scale_SfM_depth(depth, colmap_depths, colmap_coords)
        rescaled_rgbd= self.obtain_rgbd(depth, local_scale,v.tvec,v.qvec2rotmat())
        moving_rgbd = self.mask_rgbd(idx,rescaled_rgbd)


        return rescaled_rgbd, colmap_keypoints,colmap_rgb,colmap_ids
    def mask_rgbd(self,idx,rgbd):
        mask_path = self.root_data+"/"+self.vid+"/single_masks/static/static_"+str(idx)+".png"
        mask_img = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
        mask_img = cv2.resize(mask_img,(self.width,self.height))
        mask_img = 1-mask_img
        mask_img = mask_img.reshape((self.width*self.height,))
        points = rgbd[mask_img>60,:]
        return points

    def draw_pcd(self,points,color=False):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if color:
            pcd.colors = o3d.utility.Vector3dVector()
        #o3d.io.write_point_cloud("./data.ply", pcd)
        o3d.visualization.draw_geometries([pcd])

    def compare_pcds(self,p1,p2,p2_color=None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(p1)
        red = np.tile(np.array([1,0,0]),(p1.shape[0],1))
        pcd.colors = o3d.utility.Vector3dVector(red)

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(p2)
        #p2_color = np.tile(np.array([0,0,1]),(p2.shape[0],1))
        pcd2.colors = o3d.utility.Vector3dVector(p2_color)

        o3d.visualization.draw_geometries([pcd,pcd2])

    def get_frustum(self,idx):
        c_idx = [self.c.imgs_Colmap[i].id for i in self.c.imgs_Colmap.keys() if self.c.imgs_Colmap[i].name == self.epic_diff.image_paths[idx]]
        #Get corresponding colmap depth
        test_img = self.c.imgs_Colmap[c_idx[0]]
        frustum = get_frustum(
            sz=FRUSTUM_SIZE, line_radius=FRUSTUM_LINE_RADIUS,
            colmap_image=test_img,
            camera_height=EPIC_HEIGHT, camera_width=EPIC_WIDTH)
        return frustum

    
     
    def segment_pcd(self,th):
        list_id = np.array([])
        print("Analysing points...")
        for i, idx in tqdm(enumerate(self.epic_diff.img_ids),total=1000):
            if(self.diff2Col(idx) > 0): # se esiste il corrispondente in colmap
                rescaled_rgbd, colmap_keypoints,_,colmap_ids = self.extract_pcd_by_indx(idx)
                moving_rgbd = self.mask_rgbd(idx,rescaled_rgbd)

                dist = cdist(moving_rgbd,colmap_keypoints,'euclidean')

                _,c = np.where(dist < th)
                c_unique = np.unique(c)
                lower_ids = colmap_ids[c_unique] 
                list_id = np.concatenate((list_id, lower_ids))
            

        points_id = np.array(list_id)
        points_id = np.unique(points_id)
        return points_id

    def segment_run(self,load=False):
        dyn_path = os.path.join(self.pcd_path,"pcd_dyn.ply")
        static_path = os.path.join(self.pcd_path,"pcd_static.ply")
        if(load):
            
            pcd_dyn = o3d.io.read_point_cloud(dyn_path)
            pcd_static = o3d.io.read_point_cloud(static_path)
        else:
            threshold = 2
            #Getting id of moving objects
            points_id = self.segment_pcd(threshold)

            pcd_dyn = o3d.geometry.PointCloud()
            pcd_dyn.points = o3d.utility.Vector3dVector([self.c.pts_Colmap[p3d].xyz for p3d in points_id])
            pcd_dyn.colors = o3d.utility.Vector3dVector(np.tile(np.array([1,0,0]),(len(points_id),1)))
            pcd_static = o3d.geometry.PointCloud()
            pcd_static.points = o3d.utility.Vector3dVector([self.c.pts_Colmap[p3d].xyz for p3d in self.c.pts_Colmap if p3d not in points_id])
            pcd_static.colors = o3d.utility.Vector3dVector(np.tile(np.array([0,0,1]),(len(pcd_static.points),1)))
            #Save file
            o3d.io.write_point_cloud(static_path,pcd_static)
            o3d.io.write_point_cloud(dyn_path,pcd_dyn)
        return pcd_static,pcd_dyn
###############################################################################
## Main segment run
#path  = "/Users/francesco/Desktop/Università/Tesi/EgoChiara/CodiceEPICFIELDS/example_data/P01_01"
#root_data = "/Users/francesco/Desktop/Università/Tesi/EgoChiara/CodiceEPICFIELDS/depth_extractor/data/Epic_converted"
#vid = "P01_01"
#path_diff = "/Users/francesco/Desktop/Università/Tesi/EgoChiara/CodiceEPICFIELDS/depth_extractor/data/Epic_converted"
#me = motion_estimator(root_data,path,path_diff,vid)
#
#pcd_static, pcd_moving = me.segment_run(load=True)
#o3d.visualization.draw_geometries([pcd_static,pcd_moving])
###################################################################################################
# See segmentation results
path_dense = "/Users/francesco/Desktop/Università/Tesi/EgoChiara/DenseReconstruction/P01_01/fused.ply"
path_dyn = "/Users/francesco/Desktop/Università/Tesi/EgoChiara/CodiceEPICFIELDS/depth_extractor/data/Epic_converted/P01_01/pcds/pcd_dyn.ply"
pcd_dense = o3d.io.read_point_cloud(path_dense)
pcd_dyn = o3d.io.read_point_cloud(path_dyn)

dense_moving_pcd,dense_static_pcd = dense_segment(pcd_dense,pcd_dyn,0.02,first_time=True)
o3d.visualization.draw_geometries([dense_static_pcd])
###################################################################################################
## Extract pcd from a single frame
#idx = 285
#pcd_50, col_50_xyz,col_50_rgb,_ = me.extract_pcd_by_indx(idx) #120 285 730 785
#pcd_50 = me.mask_rgbd(idx,pcd_50)
#p2_color = np.tile(np.array([0,0,1]),(col_50_xyz.shape[0],1)) #BLUE Color
#me.compare_pcds(pcd_50,col_50_xyz,p2_color)
#
#
###################################################################################################
## Compare with all the pointcloud
#colmap_pcd_xyz = np.array([me.c.pts_Colmap[p3d].xyz for p3d in me.c.pts_Colmap.keys()])
#
#me.compare_pcds(pcd_50,colmap_pcd_xyz,p2_color)
#
#################################################################################################
## Compare with .ply file (Dense pcd)
#path = "/Users/francesco/Desktop/Università/Tesi/EgoChiara/DenseReconstruction/P01_01/fused.ply"
#pcd2 = o3d.io.read_point_cloud(path)
#
#p1 = pcd_50
#pcd = o3d.geometry.PointCloud()
#pcd.points = o3d.utility.Vector3dVector(p1)
#red = np.tile(np.array([1,0,0]),(p1.shape[0],1))
#pcd.colors = o3d.utility.Vector3dVector(red)
#
#FOR =  get_o3d_FOR()
#frustum = me.get_frustum(idx)
#o3d.visualization.draw_geometries([pcd,pcd2,FOR,frustum])
#
print("Ciao")