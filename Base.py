import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from normflows.distributions import BaseDistribution
from torch.distributions import StudentT
from math import lgamma, pi

# 1. DirichletProcessGaussianMixture
# 2. DPGM 
# 3. DirichletProcessPoductTMixture
# 4. DirichletProcessMultivariateTMixture
# 5. DPGTM (Dirichlet Process Gaussian T Mixture)

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
    def __init__(
        self,
        shape,
        T=3,
        alpha=1.0,
        train_alpha=True,
        train_means=True,
        train_scales=True,
        init_means=None,
        init_scales=None,
    ):
        super().__init__()
        # shape handling
        if isinstance(shape, int): shape = (shape,)
        elif isinstance(shape, list): shape = tuple(shape)
        self.shape = shape
        self.d = int(np.prod(shape))
        self.T = T

        # concentration alpha (log-space)
        init_alpha = torch.tensor(alpha, dtype=torch.float32)
        if train_alpha:
            self.log_alpha = nn.Parameter(torch.log(init_alpha))
        else:
            self.register_buffer("log_alpha", torch.log(init_alpha))

        # variational stick-breaking params (log-space for positivity)
        self.log_a = nn.Parameter(torch.zeros(T-1))
        self.log_b = nn.Parameter(torch.log(torch.ones(T-1) * alpha))

        # component means [T, d]
        if init_means is None:
            means = torch.randn(T, self.d) * 2.0
        else:
            means = torch.tensor(init_means, dtype=torch.float32)
        if train_means:
            self.means = nn.Parameter(means)
        else:
            self.register_buffer("means", means)

        # component log_scales [T, d]
        if init_scales is None:
            log_scales = torch.zeros(T, self.d)
        else:
            log_scales = torch.log(torch.tensor(init_scales, dtype=torch.float32))
        if train_scales:
            self.log_scale = nn.Parameter(log_scales)
        else:
            self.register_buffer("log_scale", log_scales)

        # expected weights buffer
        self.register_buffer("pi", torch.zeros(T))
        self.register_buffer("log_pi", torch.zeros(T))
        # initialize expected weights
        pi_mean, log_pi_mean = self._compute_expected_pi()
        self.pi.copy_(pi_mean)
        self.log_pi.copy_(log_pi_mean)

    def _compute_expected_pi(self):
        # compute expected stick-breaking weights under Beta(a,b)
        a = torch.exp(self.log_a)
        b = torch.exp(self.log_b)
        v_mean = a / (a + b)
        pis = []
        remaining = torch.ones((), device=a.device)
        for k in range(self.T - 1):
            pis.append(v_mean[k] * remaining)
            remaining = remaining * (1 - v_mean[k])
        pis.append(remaining)
        pi_mean = torch.stack(pis, dim=0)
        return pi_mean, torch.log(pi_mean + 1e-12)

    def forward(self, num_samples=1):
        device = self.means.device
        # update expected weights from a,b
        pi_mean, log_pi_mean = self._compute_expected_pi()
        self.pi.copy_(pi_mean)
        self.log_pi.copy_(log_pi_mean)

        # sample stick-breaking vars and modes
        a = torch.exp(self.log_a).unsqueeze(0)
        b = torch.exp(self.log_b).unsqueeze(0)
        U = torch.rand(num_samples, self.T - 1, device=device)
        V = (1 - (1 - U)**(1.0 / b))**(1.0 / a)
        V = V.clamp(min=1e-6, max=1-1e-6)
        one_minus_V = 1 - V
        cumprod_om = torch.cumprod(one_minus_V, dim=1)
        prod_prev = torch.cat([torch.ones(num_samples,1,device=device), cumprod_om[:,:-1]], dim=1)
        pi_mat = torch.cat([V * prod_prev[:,:self.T-1], prod_prev[:,-1:]], dim=1)
        pi_mat = pi_mat / pi_mat.sum(dim=1, keepdim=True)

        modes = torch.multinomial(pi_mat, 1).squeeze(1)

        # sample Gaussian for each sample
        eps = torch.randn(num_samples, self.d, device=device)
        loc = self.means[modes]
        scale = torch.exp(self.log_scale)[modes]
        z = loc + eps * scale
        z = z.view(num_samples, *self.shape)

        # compute log probability using expected weights (self.log_pi)
        # evaluate gaussians for all components
        z_flat = z.view(num_samples, self.d)
        means = self.means.unsqueeze(0)  # [1,T,d]
        scales = torch.exp(self.log_scale).unsqueeze(0)  # [1,T,d]
        eps_z = (z_flat.unsqueeze(1) - means) / scales
        log_gauss = (-0.5 * self.d * np.log(2*np.pi)
                     - torch.sum(torch.log(scales), dim=2)
                     - 0.5 * torch.sum(eps_z**2, dim=2))  # [N,T]
        log_comp = log_gauss + self.log_pi.unsqueeze(0)  # [N,T]
        log_prob = torch.logsumexp(log_comp, dim=1)  # [N]

        return z, log_prob

    def log_prob(self, z):
        # always use expected weights
        device = self.means.device
        num_samples = z.shape[0]
        # update expected weights
        pi_mean, log_pi_mean = self._compute_expected_pi()
        self.pi.copy_(pi_mean)
        self.log_pi.copy_(log_pi_mean)

        z_flat = z.view(num_samples, self.d)
        means = self.means.unsqueeze(0)
        scales = torch.exp(self.log_scale).unsqueeze(0)
        eps_z = (z_flat.unsqueeze(1) - means) / scales
        log_gauss = (-0.5 * self.d * np.log(2*np.pi)
                     - torch.sum(torch.log(scales), dim=2)
                     - 0.5 * torch.sum(eps_z**2, dim=2))
        log_comp = log_gauss + log_pi_mean.unsqueeze(0)
        return torch.logsumexp(log_comp, dim=1)

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
    

#####################################################################################
#####################################################################################

class GaussianDistribution(BaseDistribution):
    def __init__(self, shape, mean=None, scale=None):
        super().__init__()
        if isinstance(shape, int): shape = (shape,)
        self.shape = shape
        d = int(np.prod(shape))
        if mean is None:
            mean = torch.zeros(d)
        if scale is None:
            log_scale = torch.zeros(d)
        else:
            log_scale = torch.log(torch.tensor(scale, dtype=torch.float32))
        self.mean = nn.Parameter(mean)
        self.log_scale = nn.Parameter(log_scale)

    def forward(self, num_samples=1):
        eps = torch.randn(num_samples, *self.shape, device=self.mean.device)
        scale = torch.exp(self.log_scale).view(1, *self.shape)
        samples = eps * scale + self.mean.view(1, *self.shape)
        log_p = self.log_prob(samples)
        return samples, log_p

    def log_prob(self, z):
        z_flat = z.view(z.shape[0], -1)
        mu = self.mean.view(1, -1)
        sigma = torch.exp(self.log_scale).view(1, -1)
        lp = -0.5 * ((z_flat - mu)/sigma)**2 - torch.log(sigma) - 0.5 * np.log(2*pi)
        return lp.sum(dim=1)

class TProductDistribution(BaseDistribution):
    def __init__(self, shape, mean=None, scale=None, df=None):
        super().__init__()
        if isinstance(shape, int): shape = (shape,)
        self.shape = shape
        d = int(np.prod(shape))
        if mean is None:
            mean = torch.zeros(d)
        if scale is None:
            log_scale = torch.zeros(d)
        else:
            log_scale = torch.log(torch.tensor(scale, dtype=torch.float32))
        if df is None:
            df_buf = torch.full((d,), 2.0)
        else:
            df_buf = torch.tensor(df, dtype=torch.float32).view(d,)
        self.mean = nn.Parameter(mean)
        self.log_scale = nn.Parameter(log_scale)
        self.register_buffer('df', df_buf)

    def forward(self, num_samples=1):
        df = self.df.view(1, *self.shape)
        eps = torch.distributions.StudentT(df).rsample((num_samples,))
        scale = torch.exp(self.log_scale).view(1, *self.shape)
        samples = eps * scale + self.mean.view(1, *self.shape)
        log_p = self.log_prob(samples)
        return samples, log_p

    def log_prob(self, z):
        z_flat = z.view(z.shape[0], -1)
        mu = self.mean.view(1, -1)
        sigma = torch.exp(self.log_scale).view(1, -1)
        df = self.df.view(1, -1)
        x = (z_flat - mu) / sigma
        coef = lgamma((df+1)/2) - lgamma(df/2) - 0.5*(torch.log(df*pi) + 2*torch.log(sigma))
        lp = coef - ((df+1)/2)*torch.log(1 + x**2/df)
        return lp.sum(dim=1)

class MultivariateTDistribution(BaseDistribution):
    def __init__(self, shape, mean=None, scale=None, df=None):
        super().__init__()
        if isinstance(shape, int): shape = (shape,)
        self.shape = shape
        self.d = int(np.prod(shape))
        if mean is None:
            mean = torch.zeros(self.d)
        if scale is None:
            log_scale = torch.zeros(self.d)
        else:
            log_scale = torch.log(torch.tensor(scale, dtype=torch.float32))
        if df is None:
            df_val = float(self.d + 1)
        else:
            df_val = float(df)
        self.mean = nn.Parameter(mean)
        self.log_scale = nn.Parameter(log_scale)
        self.register_buffer('df', torch.tensor(df_val, dtype=torch.float32))

    def forward(self, num_samples=1):
        eps = torch.distributions.StudentT(self.df).rsample((num_samples, self.d))
        scale = torch.exp(self.log_scale).view(1, -1)
        samples = eps * scale + self.mean.view(1, -1)
        samples = samples.view(num_samples, *self.shape)
        log_p = self.log_prob(samples)
        return samples, log_p

    def log_prob(self, z):
        z_flat = z.view(z.shape[0], self.d)
        mu = self.mean.view(1, -1)
        sigma = torch.exp(self.log_scale).view(1, -1)
        df = self.df
        x = (z_flat - mu) / sigma
        m = torch.sum(x**2, dim=1)
        d = float(self.d)
        lp = lgamma((df + d)/2) - lgamma(df/2)
        lp = lp - 0.5*(d*torch.log(df*pi) + 2*torch.sum(torch.log(sigma)))
        lp = lp - ((df + d)/2)*torch.log1p(m/df)
        return lp

class DirichletProcessMixture(BaseDistribution):
    def __init__(
        self,
        shape,
        components=None,
        T=None,
        alpha=1.0,
        train_alpha=True,
    ):
        super().__init__()
        # record shape for sampling
        if isinstance(shape, int): shape = (shape,)
        self.shape = shape
        # default T=30 if no components provided
        if components is None:
            T = T or 30
            comps = []
            for _ in range(T):
                mean = torch.randn(int(np.prod(self.shape))) * 2.0
                comps.append(GaussianDistribution(self.shape, mean=mean))
            self.components = nn.ModuleList(comps)
            self.T = T
        else:
            self.components = nn.ModuleList(components)
            self.T = len(self.components) if T is None else T
            assert self.T == len(self.components), "T must match number of components"

        # concentration alpha
        init_alpha = torch.tensor(alpha, dtype=torch.float32)
        if train_alpha:
            self.log_alpha = nn.Parameter(torch.log(init_alpha))
        else:
            self.register_buffer("log_alpha", torch.log(init_alpha))

        # variational stick-breaking parameters
        self.log_a = nn.Parameter(torch.zeros(self.T - 1))
        self.log_b = nn.Parameter(torch.log(torch.ones(self.T - 1) * alpha))

        # buffers for expected weights (no grad)
        self.register_buffer("pi", torch.zeros(self.T))
        self.register_buffer("log_pi", torch.zeros(self.T))
        pi_mean, log_pi_mean = self._compute_expected_pi()
        self.pi.detach().copy_(pi_mean)
        self.log_pi.detach().copy_(log_pi_mean)

    def _compute_expected_pi(self):
        a = torch.exp(self.log_a)
        b = torch.exp(self.log_b)
        v_mean = a / (a + b)
        pis = []
        remaining = torch.ones((), device=a.device)
        for k in range(self.T - 1):
            pis.append(v_mean[k] * remaining)
            remaining = remaining * (1 - v_mean[k])
        pis.append(remaining)
        return torch.stack(pis, dim=0), torch.log(torch.stack(pis, dim=0) + 1e-12)

    def forward(self, num_samples=1):
        device = self.log_alpha.device
        pi_mean, log_pi_mean = self._compute_expected_pi()
        self.pi.detach().copy_(pi_mean)
        self.log_pi.detach().copy_(log_pi_mean)

        a = torch.exp(self.log_a).unsqueeze(0)
        b = torch.exp(self.log_b).unsqueeze(0)
        U = torch.rand(num_samples, self.T - 1, device=device)
        V = (1 - (1 - U)**(1.0 / b))**(1.0 / a)
        V = V.clamp(min=1e-6, max=1-1e-6)

        one_minus_V = 1 - V
        cumprod_om = torch.cumprod(one_minus_V, dim=1)
        prod_prev = torch.cat([torch.ones(num_samples,1,device=device), cumprod_om[:,:-1]], dim=1)
        pi_mat = torch.cat([V * prod_prev[:,:self.T-1], prod_prev[:,-1:]], dim=1)
        pi_mat = pi_mat / pi_mat.sum(dim=1, keepdim=True)

        modes = torch.multinomial(pi_mat, 1).squeeze(1)

        # vectorized sampling per component
        z = torch.zeros((num_samples, *self.shape), device=device)
        log_q = torch.zeros(num_samples, device=device)
        for k, comp in enumerate(self.components):
            idx = (modes == k).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            z_k, log_q_k = comp.forward(idx.numel())
            z[idx] = z_k
            log_q[idx] = log_q_k
        return z, log_q

    def log_prob(self, z):
        pi_mean, log_pi_mean = self._compute_expected_pi()
        self.pi.detach().copy_(pi_mean)
        self.log_pi.detach().copy_(log_pi_mean)

        log_probs = [comp.log_prob(z) for comp in self.components]
        log_probs = torch.stack(log_probs, dim=1)
        return torch.logsumexp(log_probs + log_pi_mean.unsqueeze(0), dim=1)