import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Chi2, StudentT
import numpy as np

# normflows 패키지의 Base 클래스와 Target 클래스가 있다고 가정
# (사용자 환경에 맞춰 import를 수정하세요)
from normflows.distributions import BaseDistribution, Target


class AsymmetricStudentT(Target):
    """
    Generalized 'Asymmetric' Student's t-distribution for d dimensions.
    - 각 차원별로 다른 자유도 df[i] 적용
    - 공분산 행렬 cov 를 통해 차원들 간의 상관 구조 표현
    - 샘플링 시, 각 차원별로 다른 chi2(df[i]) 를 사용
    """

    def __init__(
        self,
        df=None,          # float, list, or 1D tensor of length d
        mean=None,        # shape=(d,)
        cov=None         # shape=(d,d)
        ):
        """
        Args:
          df: 각 차원의 자유도. 
              - 스칼라(float, int)면 모든 차원에 동일하게 사용.
              - 리스트/텐서인 경우 d개 길이를 가져야 함.
          mean: 평균 벡터, shape=(d,)
          cov: 공분산 행렬, shape=(d,d)
        """
        super().__init__()
        
        # 기본 df 설정
        if df is None:
            df = 2.0

        # df가 스칼라라면 각 차원별로 복제
        if isinstance(df, (int, float)):
            d = 1  # 기본 차원 1개
            df = torch.tensor([float(df)], dtype=torch.float32)
        else:
            df = torch.as_tensor(df, dtype=torch.float32)
            d = df.shape[0]
        
        # mean과 cov 기본값 설정
        if mean is None:
            mean = torch.zeros(d, dtype=torch.float32)
        else:
            mean = torch.as_tensor(mean, dtype=torch.float32)

        if cov is None:
            cov = torch.eye(d, dtype=torch.float32)
        else:
            cov = torch.as_tensor(cov, dtype=torch.float32)
        
        # 검증
        assert mean.shape == (d,), "mean must be shape (d,)"
        assert cov.shape == (d, d), "cov must be shape (d,d)"
        
        self.df   = df    # shape=(d,)
        self.mean = mean  # shape=(d,)
        self.cov  = cov   # shape=(d,d)
        self.d    = d     # 몇 차원인지
        
        # torch.distributions.MultivariateNormal 사용
        # (0, I) 대신 mean=0, cov='self.cov' 로 설정하여 
        # 나중에 샘플링할 때 mean 보정을 해줄 수도 있지만,
        # 여기서는 "centered Gaussian"을 뽑아, 최종에 self.mean을 더해주는 구조.
        # -> 따라서 MVN의 mean=0, cov=self.cov 로 세팅
        self.mv_normal = MultivariateNormal(
            loc=torch.zeros(d, dtype=torch.float32),
            covariance_matrix=self.cov
        )

    def log_prob(self, z):
        """
        Independent d-dim Student-t product:
        log p(z) = sum_{i=1}^d StudentT(df_i, loc=mean_i, scale=scale_i).log_prob(z[:,i])
        """
        # z: (batch_size, d)
        batch_size, d = z.shape

        # 만약 cov를 diagonal로만 쓰려면:
        scales = torch.sqrt(torch.diagonal(self.cov))  # shape=(d,)

        total_logp = 0.0
        for i in range(d):
            dist_i = StudentT(
                df=self.df[i],
                loc=self.mean[i],
                scale=scales[i]
            )
            # z[:, i] shape = (batch_size,)
            total_logp = total_logp + dist_i.log_prob(z[:, i])

        return total_logp

    def sample(self, num_samples=1):
        """
        Samples from the d-dim 'asymmetric' Student's t distribution.

        - 각 차원별로 서로 다른 chi2(df[i])를 뽑고,
          MultivariateNormal(..., cov=self.cov)에서 (num_samples, d) 개 샘플 -> mv_samples
        - 각 차원 i 마다 t_samples[:, i] = mv_samples[:, i] / sqrt(Chi2_i / df[i])
        - 최종적으로 self.mean을 더해 준다.

        Returns:
          Tensor of shape (num_samples, d)
        """
        # 1) d차원 MVN에서 (num_samples, d) 샘플
        mv_samples = self.mv_normal.sample((num_samples,))  # (num_samples, d)

        # 2) 각 차원별 자유도 df[i]에 맞춰 chi2 샘플 -> shape (num_samples, d)
        #    예) [Chi2(df[0]) 샘플, ..., Chi2(df[d-1]) 샘플]
        chi2_samples = []
        for i in range(self.d):
            # (num_samples,) shape
            c_smp = Chi2(self.df[i]).sample((num_samples,))
            chi2_samples.append(c_smp)
        # (d, num_samples) -> transpose -> (num_samples, d)
        chi2_samples = torch.stack(chi2_samples, dim=1)

        # 3) 각 차원별로 표준화 => T분포화
        #    t_samples[:, i] = mv_samples[:, i] / sqrt( chi2_samples[:, i] / df[i] )
        t_samples = mv_samples.clone()
        for i in range(self.d):
            t_samples[:, i] = mv_samples[:, i] / torch.sqrt(chi2_samples[:, i] / self.df[i])

        # 4) mean 더하기
        return t_samples + self.mean

class MultStudentT(Target):
    """
    Multivariate Student's t-distribution with full covariance.

    log p(z) = log Gamma((nu + d)/2) - log Gamma(nu/2)
             - (d/2) * log(nu * pi) - 0.5 * log|Sigma|
             - ((nu + d)/2) * log(1 + M/nu)
    where M = (z - mean)^T Sigma^{-1} (z - mean).
    """
    def __init__(self, df=None, mean=None, Sigma=None):
        """
        Args:
            df: degrees of freedom (scalar). Defaults to 1.
            mean: tensor-like of shape (d,). Defaults to zero vector.
            Sigma: covariance matrix of shape (d, d). Defaults to identity.
        """
        super().__init__()
        # defaults
        if df is None:
            df = 1.0
        self.df = float(df)
        # mean
        if mean is None:
            self.mean = None
        else:
            self.mean = torch.as_tensor(mean, dtype=torch.float32)
        # covariance
        if Sigma is None:
            if self.mean is None:
                d = 1
                self.Sigma = torch.eye(1, dtype=torch.float32)
            else:
                d = self.mean.shape[0]
                self.Sigma = torch.eye(d, dtype=torch.float32)
        else:
            self.Sigma = torch.as_tensor(Sigma, dtype=torch.float32)
            d = self.Sigma.shape[0]
        # infer d from mean or Sigma
        if self.mean is None:
            self.mean = torch.zeros(d, dtype=torch.float32)
        self.d = d
        # precompute inverse and logdet
        self.Sigma_inv = torch.inverse(self.Sigma)
        sign, logdet = torch.slogdet(self.Sigma)
        assert sign > 0, "Sigma must be positive-definite"
        self.log_det_Sigma = logdet
        # normalization constant
        self.const = (
            torch.lgamma(torch.tensor((self.df + self.d) / 2.0))
            - torch.lgamma(torch.tensor(self.df / 2.0))
            - (self.d / 2.0) * torch.log(torch.tensor(self.df) * torch.tensor(np.pi))
            - 0.5 * self.log_det_Sigma
        )

    def log_prob(self, z):
        """
        Args:
            z: Tensor of shape (batch_size, d)
        Returns:
            Tensor of shape (batch_size,)
        """
        # center
        x = z - self.mean
        # Mahalanobis distance
        M = torch.sum((x @ self.Sigma_inv) * x, dim=1)
        # log-density
        logp = self.const - ((self.df + self.d) / 2.0) * torch.log1p(M / self.df)
        return logp
    
    def sample(self, num_samples=1):
        """
        Samples from the multivariate Student's t-distribution.

        Steps:
        1) Sample X ~ MVN(0, Sigma) of shape (num_samples, d)
        2) Sample W ~ Chi2(df) of shape (num_samples,)
        3) Return mean + X * sqrt(df / W)[:, None]
        """
        # 1) MVN samples
        mvn = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.d, dtype=self.mean.dtype, device=self.mean.device),
            covariance_matrix=self.Sigma
        )
        X = mvn.sample((num_samples,))  # (num_samples, d)
        # 2) Chi-squared samples
        W = torch.distributions.Chi2(self.df).sample((num_samples,)).to(X)
        # 3) scale and shift
        scale = torch.sqrt(self.df / W).unsqueeze(1)  # (num_samples, 1)
        samples = self.mean + X * scale  # (num_samples, d)
        return samples
    
class SymmetricParetoMixture(Target):
    """
    다차원 대칭 Pareto 혼합(Mixture) 분포
    - n_mode개의 모드, 각 모드별 동일 alpha (기본=2.0) 사용
    - 혼합 가중치 weight_k (합=1)
    - 각 모드별 평균 mean[k] (shape=(n_mode, d))
    """

    def __init__(
        self,
        n_mode=2,
        weight=None,  # shape=(n_mode,)
        alpha=None,   # 스칼라 또는 (d,) 형태 (모두 2.0 기본)
        mean=None,    # shape=(n_mode, d), 미지정 시 표준정규로 샘플
        dim=2,        # 차원
    ):
        super().__init__()

        self.n_mode = n_mode
        self.dim = dim

        # 1) weight 설정 (없으면 균등)
        if weight is None:
            weight = torch.ones(n_mode) / float(n_mode)
        else:
            weight = torch.as_tensor(weight, dtype=torch.float32)
            assert weight.shape[0] == n_mode, "weight.shape must be (n_mode,)"
        self.weight = weight

        # 2) alpha 설정(없으면 전부 2.0)
        if alpha is None:
            alpha = 2.0  # 스칼라
        if isinstance(alpha, (int, float)):
            alpha = torch.tensor([float(alpha)] * dim, dtype=torch.float32)  # (dim,)
        else:
            alpha = torch.as_tensor(alpha, dtype=torch.float32)
            assert alpha.shape[0] == dim, "alpha must be scalar or shape=(dim,)"
        # 내부 사용은 (1, dim) 모양
        self.alpha = alpha.view(1, dim)

        # 3) mean 설정(없으면 n_mode x dim 표준정규 샘플)
        if mean is None:
            # 표준정규에서 (n_mode, dim) 뽑기
            mean = torch.randn(n_mode, dim)
        else:
            mean = torch.as_tensor(mean, dtype=torch.float32)
            assert mean.shape == (n_mode, dim), "mean must be (n_mode, dim)"
        self.mean = mean

    @staticmethod
    def _icdf(u, alpha):
        """
        대칭 Pareto 분포의 Inverse CDF (벡터화 버전).
        u: [0,1] 범위 텐서, shape=(..., d)
        alpha: (1, d) 모양 (방향별 alpha)
        
        반환값: shape=(..., d)의 대칭 파레토 샘플
        """
        # u < 0.5:  p1 = -((2u)^(-1/alpha) - 1)
        # else:      p2 =  ((2(1-u))^(-1/alpha) - 1)
        p1 = -((2.0 * u) ** (-1.0 / alpha) - 1.0)
        p2 =  (2.0 * (1.0 - u)) ** (-1.0 / alpha) - 1.0
        return torch.where(u < 0.5, p1, p2)

    @staticmethod
    def _logpdf_nd(X, alpha):
        """
        X: shape=(N,d)
        alpha: shape=(1,d)  -> 각 차원별 alpha_j
        
        log p(x) = sum_{j=1}^d [ log(alpha_j/2) - (alpha_j+1)*log(1+|x_j| ) ]
                 = sum over j.  (N,) shape로 반환.
        """
        import math
        import numpy as np

        N, d = X.shape
        # alpha => (1,d)
        log_alpha = torch.log(alpha) - math.log(2.0)  # (1,d)
        log_alpha_expanded = log_alpha.expand(N, -1)  # (N,d)
        
        sum_log_alpha = torch.sum(log_alpha_expanded, dim=1)  # (N,)

        alpha_plus_1 = alpha + 1.0   # (1,d)
        X_abs = torch.abs(X)         # (N,d)
        sum_log_term = torch.sum(alpha_plus_1 * torch.log1p(X_abs), dim=1)  # (N,)

        return sum_log_alpha - sum_log_term

    def log_prob(self, x):
        """
        혼합분포의 log pdf
        p_mix(x) = sum_k [ weight[k] * pareto_pdf( x - mean[k] ) ]
                  => log( sum_k [ w_k * exp(logpdf_k) ] )
        Args:
          x: shape=(N, d)
        Returns:
          shape=(N,) 의 log_prob
        """
        N = x.shape[0]
        log_probs = []

        for k in range(self.n_mode):
            x_centered = x - self.mean[k]  # (N,d)
            lp = self._logpdf_nd(x_centered, self.alpha)  # (N,)
            # log(w_k) 더해둠
            w_k = self.weight[k]
            log_probs.append(lp + torch.log(w_k + 1e-40))  # log(w_k)

        # (n_mode, N)
        stacked = torch.stack(log_probs, dim=0)
        # log-sum-exp along dim=0 => (N,)
        log_mix = torch.logsumexp(stacked, dim=0)
        return log_mix

    def sample(self, num_samples=1):
        """
        혼합분포 샘플링:
          1) weight에 비례하여 모드 k 인덱스 뽑기(torch.multinomial, replacement=True)
          2) k번 모드의 대칭 파레토( alpha ) 샘플 => (num_samples, d)
          3) mean[k] 더함
        """
        # (num_samples,) in [0, n_mode)
        # replacement=True 이므로 매번 같은 k가 나와도 됨.
        k_idx = torch.multinomial(self.weight, num_samples, replacement=True)
    
        # (num_samples, d)
        u = torch.rand((num_samples, self.dim), dtype=torch.float32)
        pareto_part = self._icdf(u, self.alpha)  # (num_samples, d)
    
        x_out = torch.empty_like(pareto_part)
        for i in range(num_samples):
            k = k_idx[i]
            x_out[i, :] = pareto_part[i, :] + self.mean[k, :]
        return x_out


class MixtureTarget(Target):
    def __init__(self, target1, target2, weight=None, truncation=None):
        """
        Args:
          target1: 첫 번째 Target 인스턴스
          target2: 두 번째 Target 인스턴스
          weight: [w1, w2], 두 Target의 가중치 (합 1). None이면 [0.5, 0.5]
          truncation: 각 차원의 최소 허용값. 예: torch.tensor([0.0, 0.0, 0.0])
        """
        #assert target1.n_dims == target2.n_dims, "두 target의 차원이 일치해야 합니다."
        super().__init__(prop_scale=target1.prop_scale, prop_shift=target1.prop_shift)
        self.target1 = target1
        self.target2 = target2
        self.weight = weight if weight is not None else [0.5, 0.5]
        
        if truncation is not None:
            self.truncation = torch.tensor(truncation, dtype=target1.prop_scale.dtype, device=target1.prop_scale.device)
        else:
            self.truncation = None
            
        #self.n_dims = target1.n_dims  # 필요한 경우 명시적으로 정의

    def log_prob(self, z):
        if self.truncation is not None:
            mask = (z >= self.truncation).all(dim=-1)
        else:
            mask = torch.ones(z.shape[0], dtype=torch.bool, device=z.device)
    
        log_p1 = self.target1.log_prob(z)
        log_p2 = self.target2.log_prob(z)
    
        log_mix = torch.logsumexp(torch.stack([
            torch.log(torch.tensor(self.weight[0], device=z.device)) + log_p1,
            torch.log(torch.tensor(self.weight[1], device=z.device)) + log_p2,
        ], dim=0), dim=0)
    
        # 여기서 inplace 말고 out-of-place로
        log_mix = torch.where(mask, log_mix, torch.tensor(float('-inf'), device=z.device, dtype=log_mix.dtype))
        return log_mix
        
    def sample(self, num_samples=1):
        num_samples_1 = int(num_samples * self.weight[0])
        num_samples_2 = num_samples - num_samples_1

        samples_1 = self.target1.sample(num_samples_1)
        samples_2 = self.target2.sample(num_samples_2)

        samples = torch.cat([samples_1, samples_2], dim=0)

        if self.truncation is not None:
            mask = (samples >= self.truncation).all(dim=-1)
            samples = samples[mask]

        return samples

class ConstantNormal(Target):
    def __init__(self, n_dims, const=2.0):
        """
        Args:
          n_dims: 차원 수
          const: log_prob에 더할 상수 값 (기본값 2.0)
        """
        super().__init__()
        self.n_dims = n_dims
        self.const = const
        self.standard_normal = torch.distributions.Normal(loc=0.0, scale=1.0)

    def log_prob(self, z):
        """
        Args:
          z: [batch_size, n_dims] 텐서

        Returns:
          log_prob: 표준 정규분포 log_prob + 상수
        """
        logp = self.standard_normal.log_prob(z).sum(dim=-1)  # 차원마다 log_prob 합
        return logp + self.const
    