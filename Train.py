import torch
import torch.nn as nn
import numpy as np
import normflows as nf

from matplotlib import pyplot as plt
from tqdm import tqdm
from Base import DirichletProcessGaussianMixture, DPGM, MixtureBaseDistribution

import torch

def _print_tensor(name, tensor, max_entries=5):
    data = tensor.detach().cpu().numpy().ravel()
    if data.size > max_entries:
        summary = ", ".join(f"{x:.4f}" for x in data[:max_entries]) + ", …"
    else:
        summary = ", ".join(f"{x:.4f}" for x in data)
    print(f"  {name}: [{summary}]")

def print_model_parameters(q0):
    """
    주어진 DirichletProcessMixture (혹은 BaseDistribution) 객체의
    variational 파라미터 및 buffer를 출력합니다.
    """
    # 1) Top‐level stick-breaking params
    if hasattr(q0, 'log_a'):
        print("--- stick-breaking params ---")
        _print_tensor("a", torch.exp(q0.log_a))
        _print_tensor("b", torch.exp(q0.log_b))
        _print_tensor("pi (expected)", q0.pi)
        print()

    # 2) 자식 모듈(components) 순회
    for idx, comp in enumerate(getattr(q0, 'components', [q0])):
        print(f"--- component #{idx} ({comp.__class__.__name__}) ---")
        # parameters
        for name, param in comp.named_parameters(recurse=False):
            val = param
            if name == 'log_scale':
                name = 'scale'
                val = torch.exp(param)
            _print_tensor(name, val)
        # buffers (e.g. df)
        for name, buf in comp.named_buffers(recurse=False):
            _print_tensor(name, buf)
        print()


def plot_samples(dist, num_samples = 10000, show = 1, save_img=False, size=10, two_d = True, min = None, max = None, density = True):
    
    if isinstance(dist, nf.distributions.BaseDistribution):
        samples, _= dist.forward(num_samples=num_samples)
        samples = samples.cpu().detach().numpy()
    elif isinstance(dist, nf.distributions.Target):
        samples = dist.sample(num_samples=num_samples)
        samples = samples.detach().numpy()
    elif isinstance(dist, nf.NormalizingFlow):
        samples, _ = dist.sample(num_samples=num_samples)
        samples = samples.detach().numpy()

    else:
        samples = np.zeros()
        
    if samples.shape[1] == 2 and two_d:             
        grid_size = 200
        xx, yy = torch.meshgrid(torch.linspace(-1*size, size, grid_size), torch.linspace(-1*size, size, grid_size))
        
        # 그래프 플로팅
        
        plt.figure(figsize=(6, 6))
        plt.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=1)
        plt.xlim(-1*size, size)
        plt.ylim(-size, size)
        plt.title(f"Samples from ...")
        plt.xlabel("X1")
        plt.ylabel("X2")
        if save_img:
            plt.savefig(f"figure/Samples from ....png", dpi=300, bbox_inches='tight')
        plt.show()
    else: 
        marginal = show
        # 1,2, ...
        # 해당 변수 선택
        x1 = samples[:, marginal-1]
        
        # 히스토그램 그리기
        plt.figure(figsize=(6, 6))
        if min == None or max == None:
            plt.hist(x1, bins=100, density=True, alpha=0.7, edgecolor='black')
        else: 
            plt.hist(x1, bins=np.linspace(min, max, 50), density=density, alpha=0.7, edgecolor='black')
        plt.title(f'Marginal Distribution of Variable x{marginal}')
        plt.xlabel(f'x{marginal}')
        plt.ylabel('Density')
        plt.grid(True)
        plt.show()
    return


def estimate_tail_indices_t(mean, scale, df, global_mean, target, num_samples, k):
    """
    각 차원별로 독립 Student-t(base)를 뽑아 location-scale 변환 후,
    꼬리 추정 시, scores에서 이미 scaling된 값(s_k, tail)만으로
    원본 X 좌표를 복구하여 target.log_prob에 전달합니다.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mean = torch.as_tensor(mean, dtype=torch.float32, device=device)
    scale = torch.as_tensor(scale, dtype=torch.float32, device=device)
    global_mean = torch.as_tensor(global_mean, dtype=torch.float32, device=device)
    d = mean.numel()

    # 1) base로부터 (N, d) 샘플
    base = torch.distributions.StudentT(df=df)
    Z = base.sample((num_samples, d)).to(device)  # (N, d)
    X = mean + scale * Z                          # location-scale

    nu_hats = torch.empty(d, device=device)
    # 2) 차원별로 tail index 추정
    for j in range(d):
        sign = 1.0 if mean[j] >= global_mean[j] else -1.0

        # scores = X[:, j] * sign  --> 이 자체가 이미 scaled X-coordinate
        scores = X[:, j] * sign
        scores_sorted, _ = torch.sort(scores)
        s_k = scores_sorted[k-1]       # threshold in original scale
        tail = scores_sorted[k:]       # tail values in original scale

        # 복구된 X 좌표: s_k*sign == X_k, tail*sign == X_tail
        # sk_point, tail_points: full d-dim input to target.log_prob
        sk_point   = mean.unsqueeze(0).clone()
        sk_point[0, j] = s_k * sign

        tail_points = mean.unsqueeze(0).repeat(tail.size(0), 1).clone()
        tail_points[:, j] = tail * sign

        # numerator: log f(X_k) - Σ log f(X_tail_i)
        log_sk   = target.log_prob(sk_point)
        log_tail = target.log_prob(tail_points)
        numerator = torch.sum(log_sk - log_tail)

        # denominator: Σ log(tail/s_k)
        denominator = torch.sum(torch.log((tail_points[:,j]-mean[j]) / (sk_point[0,j]-mean[j])))

        nu_hats[j] = numerator / denominator - 1.0

    return nu_hats


def compute_global_mean(base):
    """
    Compute the global mean of a DirichletProcessGaussianMixture
    by taking the weighted average of its component means.
    """
    # assume base.pi holds the normalized weights (T,)
    # and base.means holds the component means (T, *shape)
    weights = base.pi                              # shape: (T,)
    means   = base.means.view(base.T, -1)          # shape: (T, d)
    
    # weighted average in the flattened space
    global_mean_flat = (weights.unsqueeze(1) * means).sum(dim=0)  # (d,)
    
    # reshape back to original event-shape
    return global_mean_flat.view(*base.shape)      # (*shape,)