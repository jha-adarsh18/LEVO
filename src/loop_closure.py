import numpy as np
import torch
from collections import defaultdict
from geometry import ransac_pnp, refine_pose_gauss_newton
import os

os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())
os.environ['OPENBLAS_NUM_THREADS'] = str(os.cpu_count())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LoopClosureDetector:
    def __init__(self, config=None):
        config = config or {}
        self.descriptor_threshold = config.get('descriptor_threshold', 0.85)
        self.min_temporal_distance = config.get('min_temporal_distance', 30)
        self.min_spatial_distance = config.get('min_spatial_distance', 5.0)
        self.ransac_threshold = config.get('ransac_threshold', 4.0)
        self.min_inliers = config.get('min_inliers', 15)
        self.max_candidates = config.get('max_candidates', 3)
        self.check_every_n = config.get('check_every_n', 30)
        self.descriptor_database = []
        self.keyframe_database = []
        self.frame_count = 0
    
    def add_keyframe(self, keyframe):
        self.keyframe_database.append(keyframe)
        self.descriptor_database.append(np.ascontiguousarray(keyframe.descriptors))
    
    def detect_loop(self, current_kf, K):
        self.frame_count += 1
        if self.frame_count % self.check_every_n != 0:
            return None, None
        if len(self.keyframe_database) < self.min_temporal_distance:
            return None, None
        candidates = self._find_candidates_vectorized(current_kf)
        if len(candidates) == 0:
            return None, None
        for candidate_kf, score in candidates:
            rel_pose, inliers = self._verify_loop(current_kf, candidate_kf, K)
            if rel_pose is not None and inliers >= self.min_inliers:
                return candidate_kf.frame_id, rel_pose
        return None, None
    
    def _find_candidates_vectorized(self, current_kf):
        current_desc = np.ascontiguousarray(current_kf.descriptors)
        temporal_threshold = len(self.keyframe_database) - self.min_temporal_distance
        if temporal_threshold <= 0:
            return []
        candidate_kfs = self.keyframe_database[:temporal_threshold]
        spatial_dists = np.array([np.linalg.norm(current_kf.pose[:3, 3] - ref_kf.pose[:3, 3]) 
                                  for ref_kf in candidate_kfs])
        valid_spatial = spatial_dists >= self.min_spatial_distance
        if not np.any(valid_spatial):
            return []
        valid_kfs = [kf for kf, valid in zip(candidate_kfs, valid_spatial) if valid]
        valid_descs = [self.descriptor_database[i] for i, valid in enumerate(valid_spatial[:len(self.descriptor_database)]) if valid]
        chunk_size = 128
        n_desc = len(current_desc)
        candidates = []
        desc_per_kf = [len(d) for d in valid_descs]
        splits = np.cumsum([0] + desc_per_kf)
        all_ref_descs = np.vstack(valid_descs)
        current_desc_gpu = torch.from_numpy(current_desc).to(device)
        all_ref_descs_gpu = torch.from_numpy(all_ref_descs).to(device)
        all_scores = []
        for start_idx in range(0, n_desc, chunk_size):
            end_idx = min(start_idx + chunk_size, n_desc)
            desc_chunk_gpu = current_desc_gpu[start_idx:end_idx]
            scores_chunk = (desc_chunk_gpu @ all_ref_descs_gpu.T).cpu().numpy()
            all_scores.append(scores_chunk)
        scores_matrix = np.vstack(all_scores)
        for i, ref_kf in enumerate(valid_kfs):
            kf_scores = scores_matrix[:, splits[i]:splits[i+1]]
            max_scores = kf_scores.max(axis=1)
            valid_scores = max_scores[max_scores > self.descriptor_threshold]
            if len(valid_scores) > 0:
                candidates.append((ref_kf, valid_scores.mean()))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:self.max_candidates]
    
    def _verify_loop(self, current_kf, candidate_kf, K):
        current_desc = np.ascontiguousarray(current_kf.descriptors)
        candidate_desc = np.ascontiguousarray(candidate_kf.descriptors)
        current_desc_gpu = torch.from_numpy(current_desc).to(device)
        candidate_desc_gpu = torch.from_numpy(candidate_desc).to(device)
        scores = (current_desc_gpu @ candidate_desc_gpu.T).cpu().numpy()
        matches = scores.argmax(axis=1)
        match_scores = scores.max(axis=1)
        valid = match_scores > self.descriptor_threshold
        if valid.sum() < self.min_inliers:
            return None, 0
        current_pts_3d = current_kf.points_3d[valid]
        candidate_pts_2d = candidate_kf.keypoints_2d[matches[valid]]
        R, t, inliers = ransac_pnp(current_pts_3d, candidate_pts_2d, K, max_iter=100, reproj_thresh=self.ransac_threshold)
        if R is None or inliers.sum() < self.min_inliers:
            return None, 0
        R_refined, t_refined = refine_pose_gauss_newton(R, t, current_pts_3d[inliers], candidate_pts_2d[inliers], K, max_iter=5)
        rel_pose = np.eye(4)
        rel_pose[:3, :3] = R_refined
        rel_pose[:3, 3] = t_refined
        return rel_pose, inliers.sum()
    
    def get_database_size(self):
        return len(self.keyframe_database)

class BagOfWords:
    def __init__(self, vocab_size=1000, config=None):
        config = config or {}
        self.vocab_size = vocab_size
        self.vocabulary = None
        self.inverted_index = defaultdict(list)
        self.idf = None
        self.descriptor_threshold = config.get('descriptor_threshold', 0.85)
        self.min_temporal_distance = config.get('min_temporal_distance', 30)
        self.min_common_words = config.get('min_common_words', 10)
    
    def build_vocabulary(self, descriptors_list):
        all_descriptors = np.vstack(descriptors_list)
        indices = np.random.choice(len(all_descriptors), min(100000, len(all_descriptors)), replace=False)
        sample = all_descriptors[indices]
        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=self.vocab_size, max_iter=100, batch_size=1000, random_state=42)
        kmeans.fit(sample)
        self.vocabulary = np.ascontiguousarray(kmeans.cluster_centers_)
        self._compute_idf(descriptors_list)
    
    def _compute_idf(self, descriptors_list):
        word_counts = np.zeros(self.vocab_size)
        for descriptors in descriptors_list:
            words = self._quantize_descriptors(descriptors)
            word_counts[np.unique(words)] += 1
        self.idf = np.log(len(descriptors_list) / (word_counts + 1))
    
    def _quantize_descriptors(self, descriptors):
        descriptors = np.ascontiguousarray(descriptors)
        descriptors_gpu = torch.from_numpy(descriptors).to(device)
        vocabulary_gpu = torch.from_numpy(self.vocabulary).to(device)
        similarities = (descriptors_gpu @ vocabulary_gpu.T).cpu().numpy()
        return similarities.argmax(axis=1)
    
    def add_keyframe(self, keyframe):
        if self.vocabulary is None:
            return
        words = self._quantize_descriptors(keyframe.descriptors)
        for word in words:
            self.inverted_index[word].append(keyframe.frame_id)
    
    def query_similar_keyframes(self, keyframe, keyframe_database, top_k=5):
        if self.vocabulary is None:
            return []
        words = self._quantize_descriptors(keyframe.descriptors)
        word_counts = np.bincount(words, minlength=self.vocab_size)
        tf_idf_query = word_counts * self.idf
        tf_idf_query = tf_idf_query / (np.linalg.norm(tf_idf_query) + 1e-8)
        candidate_ids = set()
        for word in words:
            candidate_ids.update(self.inverted_index[word])
        temporal_limit = keyframe.frame_id - self.min_temporal_distance
        candidate_ids = [kid for kid in candidate_ids if kid < temporal_limit]
        if len(candidate_ids) == 0:
            return []
        scores = []
        for kid in candidate_ids:
            kf = keyframe_database[kid]
            words_ref = self._quantize_descriptors(kf.descriptors)
            word_counts_ref = np.bincount(words_ref, minlength=self.vocab_size)
            tf_idf_ref = word_counts_ref * self.idf
            tf_idf_ref = tf_idf_ref / (np.linalg.norm(tf_idf_ref) + 1e-8)
            scores.append((kid, np.dot(tf_idf_query, tf_idf_ref)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]