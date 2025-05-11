import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from normflows.distributions import BaseDistribution
from torch.distributions import StudentT

# 1. DirichletProcessGaussianMixture
# 2. DPGM (Gumbel-softmax)
# 3. DirichletProcessPoductTMixture
# 4. DirichletProcessMultivariateTMixture
# 5. MixtureBase

#####################################################################################
#####################################################################################


class DirichletProcessGaussianMixture(BaseDistribution):

    def __init__(self, shape, T=3, train_eta=True, train_means=True, train_scales=True, 
                 tau=0.1, eta=None, means=None, scales=None):
        '''
        shape: tuple or int (e.g. (2,) for 2D)
        T: number of truncated mixture components
        tau: temperature for Gumbel-Softmax 
        eta, means, scales: optional initial values
        '''
        super().__init__()

        if isinstance(shape, int):
            shape = (shape,)
        elif isinstance(shape, list):
            shape = tuple(shape)

        self.shape = shape
        self.d = np.prod(shape)  
        self.T = T
        self.tau = tau

        # -------------------------
        # eta
        # -------------------------
        init_eta = torch.tensor(eta) if eta is not None else torch.tensor(0.0)
        if train_eta:
            self.log_eta = nn.Parameter(torch.log(init_eta.exp() if eta is not None else torch.tensor(1.0)))
        else:
            self.register_buffer("log_eta", torch.log(init_eta.exp() if eta is not None else torch.tensor(1.0)))

        # -------------------------
        # means
        # -------------------------
        if means is None:
            init_means = torch.randn(T, *shape) * 2.0
        else:
            init_means = torch.tensor(means, dtype=torch.float32)

        if train_means:
            self.means = nn.Parameter(init_means)
        else:
            self.register_buffer("means", init_means)

        # -------------------------
        # scales (not implemented in original code but prepared)
        if scales is None:
            init_log_scale = torch.zeros(*shape)  # log(1) = 0
        else:
            init_log_scale = torch.log(torch.tensor(scales, dtype=torch.float32))

        if train_scales:
            self.log_scale = nn.Parameter(init_log_scale)
        else:
            self.register_buffer("log_scale", init_log_scale)

        # -------------------------
        # 4) create buffers for pi and log_pi
        #    We'll fill them in with update_beta()
        # -------------------------
        self.register_buffer("pi", torch.zeros(T))
        self.register_buffer("log_pi", torch.zeros(T))

        # first initialization: sample beta -> pi
        self.update_beta()

    def update_beta(self):
        """
        Called explicitly if self.log_eta changed (by optimizer, for example).
        This re-samples beta and updates self.pi, self.log_pi accordingly.
        """
        device = self.log_eta.device

        eta = torch.exp(self.log_eta)

        # T-1 uniforms on the same device
        u = torch.rand(self.T - 1, device=device)

        # beta_k = 1 - (1-u)^(1/eta), shape=(T-1,)
        beta = 1.0 - (1.0 - u)**(1.0 / eta)

        # stick-break
        pis = []
        prod_term = torch.tensor(1.0, device=device)
        for i in range(self.T - 1):
            b_i = beta[i]
            pi_i = b_i * prod_term
            pis.append(pi_i)
            prod_term = prod_term * (1.0 - b_i)
        pis.append(prod_term)
        pi_tensor = torch.stack(pis, dim=0)
        '''
        # update the buffers in-place
        with torch.no_grad():
            self.pi.copy_(pi_tensor)
            self.log_pi.copy_(torch.log(pi_tensor + 1e-12))
        '''
        self.pi = pi_tensor
        self.log_pi = torch.log(pi_tensor + 1e-12)

    
    def forward(self, num_samples=1):
        device = self.log_pi.device

        # Sample mode indices based on precomputed self.pi
        mode = torch.multinomial(self.pi, num_samples, replacement=True)  # shape: (num_samples,)
        mode_1h = nn.functional.one_hot(mode, self.T).unsqueeze(-1)  # shape: (num_samples, T, 1)
    
        # Sample from the Gaussian components
        eps_ = torch.randn(num_samples, self.d, dtype=self.means.dtype, device=self.means.device)
    
        # Select the corresponding scale and mean for the sampled mode
        means_expand = self.means.unsqueeze(0)  # shape = (1, T, *shape)
        
        scale_sample = torch.sum(torch.exp(self.log_scale) * mode_1h, dim=1)  # shape: (num_samples, dim)
        loc_sample = torch.sum(means_expand * mode_1h, dim=1) # (num_samples, *shape)
    
        # Generate samples
        z = eps_ * scale_sample + loc_sample  # shape: (num_samples, dim)
    
        # Compute log probability
        eps = (z[:, None, :] - self.means) / torch.exp(self.log_scale)  # shape: (num_samples, T, dim)
        log_p = (
            -0.5 * self.d * np.log(2 * np.pi)
            + self.log_pi  # self.pi 대신 self.log_pi 사용
            - 0.5 * torch.sum(eps ** 2, dim=2)
            - torch.sum(self.log_scale)
        )
        log_p = torch.logsumexp(log_p, dim=1)  # Sum over modes
    
        return z, log_p

    def log_prob(self, z):
        device = self.log_pi.device
        scale = torch.exp(self.log_scale)  # shape=(*self.shape)
         
        eps = (z[:, None, :] - self.means) / torch.exp(self.log_scale)
        log_p = (
            -0.5 * self.d * np.log(2 * np.pi)
            + self.log_pi.unsqueeze(0)
            - 0.5 * torch.sum(torch.pow(eps, 2), 2)
            - torch.sum(torch.log(scale))
        )
        log_p = torch.logsumexp(log_p, 1)

        return log_p

#####################################################################################
#####################################################################################

class DPGM(BaseDistribution):
    """
    Truncated Dirichlet Process + Gumbel-Softmax mixture of Gaussians, 
    with attempt to let grad flow to eta by removing detach/no_grad in update_beta.
    NOTE: actual gradient might still be small or partial, 
          because uniform sampling is not a perfect reparameterization.
    """

    def __init__(
        self,
        shape,
        T=3,
        train_eta=True,
        train_means=True,
        train_scale=True,
        tau=0.1
    ):
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        elif isinstance(shape, list):
            shape = tuple(shape)

        self.shape = shape  
        self.d = np.prod(shape)  
        self.T = T
        self.tau = tau

        # 1) eta => log_eta
        if train_eta:
            # 초기 예: log_eta=1.5 => eta=exp(1.5)=~4.48
            self.log_eta = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("log_eta", torch.tensor(0.0))

        # 2) means => shape=(T,*shape), 예: 스케일 크게
        init_means = torch.randn(T, *shape) * 1.0
        if train_means:
            self.means = nn.Parameter(init_means)
        else:
            self.register_buffer("means", init_means)

        # 3) single diagonal scale
        init_log_scale = torch.zeros(*shape)
        if train_scale:
            self.log_scale = nn.Parameter(init_log_scale)
        else:
            self.register_buffer("log_scale", init_log_scale)

        # 4) pi, log_pi (버퍼 or 일반 속성)
        self.register_buffer("pi", torch.zeros(T))
        self.register_buffer("log_pi", torch.zeros(T))

        # first init => sample pi
        self.update_beta()

    def update_beta(self):
        """
        Remove detach() and no_grad(), 
        so that self.log_eta -> pi might remain in the autograd graph.
        Still, random sampling from Uniform means partial grad for eta won't be fully accurate 
        unless we do a bigger reparam approach.
        """
        device = self.log_eta.device
        # eta in the graph
        eta = torch.exp(self.log_eta)  # no detach

        # random uniform => still a non-differentiable operation wrt eta
        u = torch.rand(self.T - 1, device=device)

        # stick-breaking
        beta = 1.0 - (1.0 - u)**(1.0 / eta)  # shape=(T-1,)
        pis = []
        prod_term = torch.tensor(1.0, device=device)
        for i in range(self.T - 1):
            b_i = beta[i]
            pi_i = b_i * prod_term
            pis.append(pi_i)
            prod_term = prod_term * (1.0 - b_i)
        pis.append(prod_term)
        pi_tensor = torch.stack(pis, dim=0)

        # direct assignment => now pi, log_pi might keep the graph
        # but note that "pi_tensor" includes random ops from uniform...
        self.pi = pi_tensor
        self.log_pi = torch.log(pi_tensor + 1e-12)

    def _gumbel_softmax(self, log_pi, tau, num_samples):
        device = log_pi.device
        g = -torch.log(-torch.log(torch.rand(num_samples, self.T, device=device)))
        logits = log_pi.unsqueeze(0) + g
        c = torch.softmax(logits / tau, dim=1)
        return c

    def forward(self, num_samples=1):
        """
        Sample z using Gumbel-Softmax cluster selection => c
        Then approximate log_p.
        """
        device = self.log_pi.device
        c = self._gumbel_softmax(self.log_pi, self.tau, num_samples)

        scale = torch.exp(self.log_scale)
        eps = torch.randn((num_samples,) + self.shape, device=device)

        means_expand = self.means.unsqueeze(0)  
        c_expand = c.unsqueeze(-1)
        for _ in range(len(self.shape) - 1):
            c_expand = c_expand.unsqueeze(-1)

        weighted_means = (means_expand * c_expand).sum(dim=1)
        z = weighted_means + scale * eps

        # approximate log_p
        z_expand = z.unsqueeze(1)
        means_2 = self.means.unsqueeze(0)
        diff = (z_expand - means_2) / scale
        diff_sq = 0.5 * (diff**2).sum(dim=list(range(2, 2 + len(self.shape))))
        sum_log_scale = torch.sum(torch.log(scale))
        log_gauss = (
            -0.5 * self.d * np.log(2.0 * np.pi)
            - sum_log_scale
            - diff_sq
        )  # shape=(num_samples, T)

        expand_log_pi = self.log_pi.unsqueeze(0)
        mixture_term = expand_log_pi + log_gauss
        log_p_i = (c * mixture_term).sum(dim=1)

        return z, log_p_i
    
    def log_prob(self, z):
        """
        Mixture log p(z) with fixed pi (non-random).
        """
        if z.dim() == 1 + len(self.shape):
            batch_size = z.size(0)
        else:
            raise ValueError(f"z must have shape=(batch_size, {self.shape})")

        device = self.log_pi.device
        scale = torch.exp(self.log_scale)

        z_expand = z.unsqueeze(1)
        means_expand = self.means.unsqueeze(0)
        diff = (z_expand - means_expand) / scale
        diff_sq = 0.5 * torch.sum(diff**2, dim=list(range(2, 2 + len(self.shape))))
        sum_log_scale = torch.sum(torch.log(scale))

        log_gauss = (
            -0.5 * self.d * np.log(2.0 * np.pi)
            - sum_log_scale
            - diff_sq
        )
        pi_expand = self.pi.unsqueeze(0)
        log_mix = torch.logsumexp(torch.log(pi_expand) + log_gauss, dim=1)
        return log_mix

#####################################################################################
#####################################################################################

class DirichletProcessPoductTMixture(BaseDistribution):
    def __init__(
        self,
        shape,
        T=3,
        train_eta=True,
        train_means=True,
        train_scales=True,
        tau=0.1,
        eta=None,
        means=None,
        scales=None,
        df=2,
    ):
        '''
        shape: tuple or int (e.g. (2,) for 2D)
        T: number of truncated mixture components
        '''
        super().__init__()
        
        if isinstance(shape, int):
            shape = (shape,)
        elif isinstance(shape, list):
            shape = tuple(shape)

        self.shape = shape
        self.d = np.prod(shape)
        self.T = T

        #-----# eta #-----#
        init_eta = torch.tensor(eta) if eta is not None else torch.tensor(0.0)
        if train_eta:
            self.log_eta = nn.Parameter(torch.log(init_eta.exp() if eta is not None else torch.tensor(1.0)))
        else:
            self.register_buffer("log_eta", torch.log(init_eta.exp() if eta is not None else torch.tensor(1.0)))
            
        #-----# degrees of freedom per mode & dim #-----#
        # df_init: (T, *shape)
        if isinstance(df, (int, float)):
            df_init = torch.full((T, *shape), float(df))
        else:
            df_init = torch.as_tensor(df, dtype=torch.float32).view((T, *shape))
        self.register_buffer("df", df_init)
        
        #-----# means #-----#
        if means is None:
            init_means = torch.randn(T, *shape) * 1.0
        else:
            init_means = torch.tensor(means, dtype=torch.float32)

        if train_means:
            self.means = nn.Parameter(init_means)
        else:
            self.register_buffer("means", init_means)

        #-----# scales #-----#
        if scales is None:
            init_log_scale = torch.zeros(*shape)  # log(1) = 0
        else:
            init_log_scale = torch.log(torch.tensor(scales, dtype=torch.float32))

        if train_scales:
            self.log_scale = nn.Parameter(init_log_scale)
        else:
            self.register_buffer("log_scale", init_log_scale)

        # -------------------------
        # create buffers for pi and log_pi
        #    We'll fill them in with update_beta()
        # -------------------------
        self.register_buffer("pi", torch.zeros(T))
        self.register_buffer("log_pi", torch.zeros(T))

        # first initialization: sample beta -> pi
        self.update_beta()

    def update_beta(self):
        """
        Called explicitly if self.log_eta changed (by optimizer, for example).
        This re-samples beta and updates self.pi, self.log_pi accordingly.
        """
        device = self.log_eta.device

        eta = torch.exp(self.log_eta)

        # T-1 uniforms on the same device
        u = torch.rand(self.T - 1, device=device)

        # beta_k = 1 - (1-u)^(1/eta), shape=(T-1,)
        beta = 1.0 - (1.0 - u)**(1.0 / eta)

        # stick-break
        pis = []
        prod_term = torch.tensor(1.0, device=device)
        for i in range(self.T - 1):
            b_i = beta[i]
            pi_i = b_i * prod_term
            pis.append(pi_i)
            prod_term = prod_term * (1.0 - b_i)
        pis.append(prod_term)
        pi_tensor = torch.stack(pis, dim=0)
        
        self.pi = pi_tensor
        self.log_pi = torch.log(pi_tensor + 1e-12)

    def forward(self, num_samples=1):
        device = self.log_pi.device

        # 1) 모드 인덱스 샘플링
        mode = torch.multinomial(self.pi, num_samples, replacement=True)      # (N,)
        # 2) 해당 모드의 평균(loc), scale, df 선택
        #   - self.means: (T, *shape)
        #   - self.log_scale: (*shape)  (공통 스케일인 경우)
        #   - self.df: (T, *shape)
        loc       = self.means[mode]                                          # (N, *shape)
        scale_all = torch.exp(self.log_scale)                                 # (*shape)
        # 모드마다 똑같이 반복된 스케일 텐서를 (T, *shape)로 만들고 인덱싱
        scale_t   = scale_all.unsqueeze(0).expand(self.T, *self.shape)       # (T, *shape)
        scale     = scale_t[mode]                                             # (N, *shape)
        df_batch  = self.df[mode]                                              # (N, *shape)

        # 3) Student-t 분포로부터 reparameterized 샘플링
        t_dist = StudentT(df_batch)  # loc=0, scale=1인 Student-t
        eps    = t_dist.rsample()    # (N, *shape)

        # 4) 최종 샘플
        z = loc + scale * eps        # (N, *shape)

        # 5) 간단히 mixture의 log-prob도 반환하고 싶다면 아래처럼…
        log_p = self.log_prob(z)     # (N,)
        return z, log_p

    def log_prob(self, z):
        """
        Mixture log p(z) = log ∑ₖ πₖ ∏ⱼ StudentT(zⱼ | dfₖⱼ, meanₖⱼ, scaleⱼ)
        """
        N = z.shape[0]

        # 1) 차원 확장을 위한 준비
        x           = z.unsqueeze(1)                                         # (N, 1, *shape)
        means_exp   = self.means.unsqueeze(0)                                # (1, T, *shape)
        scale_all   = torch.exp(self.log_scale)                              # (*shape)
        scale_exp   = scale_all.unsqueeze(0).unsqueeze(1).expand(N, self.T, *self.shape)  # (N, T, *shape)
        df_exp      = self.df.unsqueeze(0).expand(N, self.T, *self.shape)     # (N, T, *shape)

        # 2) 표준화 잔차 계산
        y = (x - means_exp) / scale_exp                                      # (N, T, *shape)

        # 3) 각 차원별 log-normalization 상수
        log_norm = (
            torch.lgamma((df_exp + 1) / 2)
            - torch.lgamma(df_exp / 2)
            - 0.5 * torch.log(df_exp * torch.tensor(np.pi, device=df_exp.device))
            - torch.log(scale_exp)
        )  # (N, T, *shape)

        # 4) 각 차원별 log-density
        log_density = log_norm - ((df_exp + 1) / 2) * torch.log1p(y**2 / df_exp)  # (N, T, *shape)

        # 5) 차원 합산 → 컴포넌트별 log-prob, 가중치 합산 → mixture log-sum-exp
        sum_dims    = list(range(2, 2 + len(self.shape)))  
        comp_log_p  = log_density.sum(dim=sum_dims) + self.log_pi.unsqueeze(0)   # (N, T)
        log_p       = torch.logsumexp(comp_log_p, dim=1)                        # (N,)

        return log_p
    

#####################################################################################
#####################################################################################
class MixtureBaseDistribution(BaseDistribution):
    def __init__(self, base1, base2, trainable=True, initial_weights=None):
        super().__init__()
        self.base1 = base1
        self.base2 = base2
        

        if initial_weights is None:
            initial_weights = torch.tensor([0.5, 0.5])
        else:
            initial_weights = torch.tensor(initial_weights)
        if trainable:
            self.logits = nn.Parameter(torch.log(initial_weights).clone().detach())
        else:
            lw = initial_weights.clone().detach()
            self.register_buffer("logits", torch.log(lw))

        self.trainable = trainable

    def forward(self, num_samples=1, eps=1e-4):
        # Compute weights (softmax of logits)
        weights = torch.softmax(self.logits, dim=0)
    
        # 강제로 한 쪽 weight가 너무 작으면 완전히 0으로 만들어줌
        if weights[0] < eps:
            weights = torch.tensor([0.0, 1.0], device=weights.device)
        elif weights[1] < eps:
            weights = torch.tensor([1.0, 0.0], device=weights.device)
    
        # Sample from Bernoulli if needed
        if weights[0] == 1.0:
            choices = torch.ones(num_samples, dtype=torch.long, device=weights.device)
        elif weights[0] == 0.0:
            choices = torch.zeros(num_samples, dtype=torch.long, device=weights.device)
        else:
            choices = torch.bernoulli(weights[0].repeat(num_samples)).long()
    
        # Count how many samples to draw from each base
        n1 = (choices == 1).sum().item()
        n2 = num_samples - n1
    
        # Sample from each base
        if n1 > 0:
            z1, _ = self.base1.forward(n1)
        else:
            dummy_shape = self.base1.forward(1)[0].shape[1:]
            z1 = torch.empty((0, *dummy_shape), device=weights.device)
            #logp1 = torch.empty(0, device=weights.device)
    
        if n2 > 0:
            z2, _ = self.base2.forward(n2)
        else:
            dummy_shape = self.base2.forward(1)[0].shape[1:]
            z2 = torch.empty((0, *dummy_shape), device=weights.device)
            #logp2 = torch.empty(0, device=weights.device)
    
        # Concatenate in the correct order
        z = torch.empty(num_samples, *z1.shape[1:], device=weights.device)
        z[choices == 1] = z1
        z[choices == 0] = z2

        # Compute log probabilities under the full mixture distribution
        logp1 = self.base1.log_prob(z)
        logp2 = self.base2.log_prob(z)
        
        # fallback: 각 샘플별로, 만약 logp2가 –∞이면 base2 기여를 무시하고 logp1+log(weight[0])로 사용
        mixture = torch.stack([logp1 + torch.log(weights[0]),
                               logp2 + torch.log(weights[1])], dim=0)
        # mask: logp2가 –∞인 샘플 (element-wise)
        mask = torch.isneginf(logp2)
        fallback = logp1 + torch.log(weights[0])
        # 최종 혼합 확률: fallback 조건에 맞으면 fallback, 아니면 logsumexp 결과 사용
        logp = torch.where(mask, fallback, torch.logsumexp(mixture, dim=0))
        return z, logp

    def log_prob(self, z):
        # Compute log probabilities from each base
        logp1 = self.base1.log_prob(z)
        logp2 = self.base2.log_prob(z)
        weights = torch.softmax(self.logits, dim=0)
        
        mixture = torch.stack([logp1 + torch.log(weights[0]),
                               logp2 + torch.log(weights[1])], dim=0)
        mask = torch.isneginf(logp2)
        fallback = logp1 + torch.log(weights[0])
        log_prob = torch.where(mask, fallback, torch.logsumexp(mixture, dim=0))
        return log_prob