import os
import numpy as np
import torch
import igl
from typing import Sequence, Tuple, Optional


def _to_tensor(x, device, dtype=torch.float32):
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def load_vertices_auto(path: str, mean: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Load a target shape from either:
      - .npy of shape (N,3)  → interpreted as absolute vertices
      - .npy of shape (3N,)  → reshaped to (N,3) and interpreted as absolute
      - .npy 'offset' (same shape as mean): if mean is provided and data appears
        centered around 0, we add mean to get absolute vertices
      - .obj                  → read via igl and return verts (N,3)

    Heuristics:
      - If abs(data).mean() < abs(mean).mean() * 0.25 and mean is given,
        we treat data as an offset and add mean.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".obj":
        v, _ = igl.read_triangle_mesh(path)
        return v.astype(np.float32)

    if ext == ".npy":
        arr = np.load(path)
        if arr.ndim == 1:  # (3N,)
            N3 = arr.shape[0]
            assert N3 % 3 == 0, "Flattened npy must have length multiple of 3."
            arr = arr.reshape(N3 // 3, 3).astype(np.float32)
        elif arr.ndim == 2 and arr.shape[1] == 3:
            arr = arr.astype(np.float32)
        else:
            raise ValueError(f"Unsupported npy shape: {arr.shape}")

        if mean is not None:
            # crude offset detection
            m_abs = float(np.abs(mean).mean()) if mean is not None else 1.0
            a_abs = float(np.abs(arr).mean())
            if m_abs > 1e-8 and a_abs < 0.25 * m_abs:
                # likely an offset → convert to absolute
                arr = mean.astype(np.float32) + arr
        return arr

    raise ValueError(f"Unsupported file extension: {ext}")


class LandmarkFitter:
    """
    Fit 3DMM shape coefficients to a target mesh using sparse 3D landmark constraints.

    Model:
        x = mean + B^T (p * std)    where B has shape (K, 3N), p are normalized coeffs,
                                    and 'std' = sqrt(eigenvalues) from PCA.

    Loss:
        L = L1(pred_landmarks, target_landmarks) + lambda * ||p||_2^2
        (optionally weighted per landmark)

    Args:
        components_path: npy, PCA components of shape (K, 3N)
        pca_var_path:    npy, PCA explained_variance_ (length K) → std = sqrt(var)
        mean_path:       npy, mean vertices of shape (N,3) or (3N,)
        face_path:       npy, faces (F,3) int32
        landmark_indices:list[int], vertex indices for landmarks (M)
        device:          'cpu' or 'cuda:0'
    """

    def __init__(
        self,
        components_path: str,
        pca_var_path: str,
        mean_path: str,
        face_path: str,
        landmark_indices: Sequence[int],
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        # Load PCA
        B = np.load(components_path).astype(np.float32)           # (K, 3N)
        self.basis_shape = _to_tensor(B.T, self.device)           # (3N, K)
        self.n_basis = self.basis_shape.shape[1]

        pca_var = np.load(pca_var_path).astype(np.float32)        # (K,)
        self.pca_std = _to_tensor(np.sqrt(pca_var), self.device)  # (K,)

        mean = np.load(mean_path).astype(np.float32)
        if mean.ndim == 1:
            assert mean.size % 3 == 0, "mean flat length must be multiple of 3"
            N = mean.size // 3
            mean = mean.reshape(N, 3)
        self.mean = _to_tensor(mean, self.device)                 # (N, 3)
        self.num_vertices = self.mean.shape[0]

        face = np.load(face_path).astype(np.int32)
        self.face = face

        self.landmark_indices = list(map(int, landmark_indices))  # (M,)

        # State filled after fit()
        self.p = None
        self.target_lm = None

    # ---------- internal ----------
    def _synthesize_vertices_from_p(self, p: torch.Tensor) -> torch.Tensor:
        """
        Given normalized p (K,), compute delta vertices (N,3):
            beta = p * std
            delta = (B^T @ beta).view(N,3)
        """
        beta = p * self.pca_std                                # (K,)
        delta_flat = self.basis_shape @ beta                   # (3N,)
        delta = delta_flat.view(self.num_vertices, 3)          # (N,3)
        return delta

    # ---------- public ----------
    @torch.no_grad()
    def reconstruct(self, p: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Return full reconstructed vertices (N,3) for either current self.p or provided p.
        """
        if p is None:
            p = self.p
        delta = self._synthesize_vertices_from_p(p)
        return self.mean + delta

    def fit(
        self,
        target_path: str,
        init_p: Optional[np.ndarray] = None,
        lr: float = 5e-3,
        n_iter: int = 500,
        reg_l2: float = 1e-5,
        weights: Optional[Sequence[float]] = None,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Optimize shape coefficients p to match target landmarks.

        Args:
            target_path: path to .obj or .npy (absolute verts or offsets)
            init_p:      optional initial params (K,)
            lr:          Adam learning rate
            n_iter:      iterations
            reg_l2:      lambda * ||p||_2^2
            weights:     optional per-landmark weights (M,)
            verbose:     print every 10 iters

        Returns: (p_numpy, recon_vertices_numpy, loss_history_list)
        """
        # Load target absolute vertices (N,3)
        target_vertices_np = load_vertices_auto(target_path, mean=self.mean.detach().cpu().numpy())
        target_vertices = _to_tensor(target_vertices_np, self.device)  # (N,3)
        target_lm = target_vertices[self.landmark_indices]             # (M,3)
        self.target_lm = target_lm.detach()

        # Params
        if init_p is None:
            self.p = torch.zeros(self.n_basis, device=self.device, requires_grad=True)
        else:
            self.p = _to_tensor(init_p, self.device).clone().requires_grad_(True)

        # Optional weights
        if weights is not None:
            w = _to_tensor(np.asarray(weights, dtype=np.float32), self.device)  # (M,)
            w = w / (w.mean() + 1e-8)
        else:
            w = None

        opt = torch.optim.Adam([self.p], lr=lr, weight_decay=0.0)

        loss_hist = []
        for it in range(n_iter):
            opt.zero_grad()

            delta = self._synthesize_vertices_from_p(self.p)          # (N,3)
            pred_lm = delta[self.landmark_indices] + self.mean[self.landmark_indices]

            if w is None:
                data_loss = torch.nn.functional.l1_loss(pred_lm, target_lm)
            else:
                diff = torch.abs(pred_lm - target_lm)                 # (M,3)
                data_loss = (w[:, None] * diff).mean()

            reg = reg_l2 * torch.sum(self.p**2)
            loss = data_loss + reg

            loss.backward()
            opt.step()
            loss_hist.append(float(loss.item()))

            if verbose and (it % 10 == 0 or it == n_iter - 1):
                print(f"[Iter {it:04d}] total={loss.item():.6f}  data={data_loss.item():.6f}  reg={reg.item():.6f}")

        recon = (self.mean + self._synthesize_vertices_from_p(self.p)).detach().cpu().numpy()
        return self.p.detach().cpu().numpy(), recon, loss_hist

    @torch.no_grad()
    def report_landmark_errors(self, vertices_np: np.ndarray) -> Tuple[float, float]:
        """
        Compute mean/max Euclidean error on landmarks (in same units as vertices).
        """
        pred = torch.from_numpy(vertices_np[self.landmark_indices]).to(self.device).float()
        d = torch.linalg.norm(pred - self.target_lm, dim=1)
        return float(d.mean().item()), float(d.max().item())

    @torch.no_grad()
    def save_mesh(self, save_path: str, vertices_np: np.ndarray, faces_np: Optional[np.ndarray] = None):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        faces = self.face if faces_np is None else faces_np
        igl.writeOBJ(save_path, vertices_np.astype(np.float32), faces.astype(np.int32))
