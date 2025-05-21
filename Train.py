import torch
import torch.nn as nn
import numpy as np
import normflows as nf

from matplotlib import pyplot as plt
from tqdm import tqdm
from Base import DirichletProcessMixture, TProductDistribution, GaussianDistribution, DirichletProcessGaussianMixture, DPGM
from scipy.stats import gaussian_kde


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

def plot_samples(dist,
                 num_samples=10000,
                 show=1,
                 save_img=False,
                 size=5,
                 two_d=True,
                 contour=False,
                 grid_size=200,
                 density=True,
                 vmin=None,
                 vmax=None,
                 kde=False):
    """
    Samples from a distribution and plots either a scatter or contour (2D) or histogram/KDE (1D).

    Args:
        dist: nf.distributions.BaseDistribution | nf.distributions.Target | nf.NormalizingFlow
        num_samples (int): number of samples to draw
        show (int): which marginal to plot in 1D mode (1-based index)
        save_img (bool): whether to save the figure under "figure/"
        size (float): axis limit for 2D plots
        two_d (bool): whether to treat 2D samples specially
        contour (bool): if True and two_d and dim==2, plot density contour instead of scatter
        grid_size (int): resolution of grid for contour plot
        density (bool): histogram density flag for 1D
        vmin, vmax (float): min/max values for histogram/KDE
        kde (bool): if True, plot KDE line instead of histogram in 1D
    """
    # draw samples
    if isinstance(dist, nf.distributions.BaseDistribution):
        samples, _ = dist.forward(num_samples=num_samples)
        samples = samples.cpu().detach().numpy()
    elif isinstance(dist, nf.distributions.Target):
        samples = dist.sample(num_samples=num_samples).detach().numpy()
    elif isinstance(dist, nf.NormalizingFlow):
        samples, _ = dist.sample(num_samples=num_samples)
        samples = samples.detach().numpy()
    else:
        raise ValueError("Unsupported distribution type")

    # 2D visualization
    if two_d and samples.shape[1] == 2:
        x, y = samples[:, 0], samples[:, 1]
        plt.figure(figsize=(size, size))
        plt.xlim(vmin, vmax)
        plt.ylim(vmin, vmax)
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.title("Samples from distribution")

        if contour:
            # estimate density on a grid
            xx, yy = torch.meshgrid(
                torch.linspace(vmin, vmax, grid_size),
                torch.linspace(vmin, vmax, grid_size),
                indexing='xy'
            )
            grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)

            # compute log_prob via dist.log_prob if available
            try:
                dist_fn = getattr(dist, 'log_prob')
                with torch.no_grad():
                    lp = dist_fn(torch.tensor(grid_points, dtype=torch.float32))
                zz = torch.exp(lp).cpu().view(grid_size, grid_size).numpy()
                print("Using log_prob for density estimation")
            except Exception:
                # fallback to histogram KDE
                kde2d = gaussian_kde(np.vstack([x, y]))
                zz = kde2d(np.vstack([xx.numpy().flatten(), yy.numpy().flatten()])).reshape(grid_size, grid_size)

            CS = plt.contour(xx.numpy(), yy.numpy(), zz)
            #plt.clabel(CS, inline=1, fontsize=10)
        else:
            plt.scatter(x, y, alpha=0.3, s=1)

        if save_img:
            plt.savefig("figure/samples_2d.png", dpi=300, bbox_inches='tight')
        plt.gca().set_aspect('equal', 'box')
        plt.show()

    # 1D histogram or KDE
    else:
        marginal = show - 1
        if marginal < 0 or marginal >= samples.shape[1]:
            raise IndexError("`show` index out of range for sample dimension")
        x1 = samples[:, marginal]
        plt.figure(figsize=(6, 6))

        if kde:
            # KDE line
            kde_func = gaussian_kde(x1)
            xmin, xmax = (vmin if vmin is not None else x1.min()), (vmax if vmax is not None else x1.max())
            xs = np.linspace(xmin, xmax, 200)
            plt.plot(xs, kde_func(xs), label='KDE')
        else:
            # histogram bars
            if vmin is not None and vmax is not None:
                bins = np.linspace(vmin, vmax, 50)
                plt.hist(x1, bins=bins, density=density, alpha=0.7, edgecolor='black')
            else:
                plt.hist(x1, bins=100, density=density, alpha=0.7, edgecolor='black')

        plt.title(f'Marginal Distribution of Variable x{show}')
        plt.xlabel(f'x{show}')
        plt.ylabel('Density')
        if kde:
            plt.legend()
        plt.grid(True)
        if save_img:
            fname = f"figure/marginal_x{show}{'_kde' if kde else ''}.png"
            plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.show()

def estimate_tail_indices_t(mean, scale, df, global_mean, target, num_samples, k,
                            model = None):
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
        if model is None:
            for flow in model.flows:
                log_sk, _ = flow(log_sk)
                log_tail, _ = flow(log_tail)
            log_sk   = target.log_prob(sk_point)
            log_tail = target.log_prob(tail_points)
        else:
            log_sk   = model.log_prob(sk_point)
            log_tail = model.log_prob(tail_points)
        numerator = torch.sum(log_sk - log_tail)

        # denominator: Σ log(tail/s_k)
        denominator = torch.sum(torch.log((tail_points[:,j]-mean[j]) / (sk_point[0,j]-mean[j])))

        nu_hats[j] = numerator / denominator - 1.0

    return nu_hats

def compute_global_mean(base):
    """
    Compute the global mean of a DirichletProcessMixture
    by taking the weighted average of its component means.
    """
    # 1) mixture weight (T,)
    weights = base.pi  # 이미 normalized 되어 있다고 가정

    # 2) 각 컴포넌트에서 mean을 flatten 해서 모으기: (T, d)
    means = torch.stack([
        comp.mean.view(-1) for comp in base.components
    ], dim=0)

    # 3) weighted average: (d,)
    global_mean_flat = (weights.unsqueeze(1) * means).sum(dim=0)

    # 4) 원래 shape으로 복원
    return global_mean_flat.view(*base.shape)

def TrainModel(model, 
               dimension=2, 
               max_iter = 2000, 
               num_samples = 2**9, 
               show_iter=500, 
               lr = 5e-4, 
               weight_decay=1e-5, 
               freeze_flow=False, 
               freeze_ratio = 0.8,
               vmin = None,
               vmax = None
               ):

    loss_hist = np.array([])
    alpha_hist = []

    if freeze_flow:
        for param in model.flows.parameters():
            param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)    

    if isinstance(model.q0, DirichletProcessMixture):
        for it in tqdm(range(max_iter)):
            optimizer.zero_grad()
            loss = model.reverse_kld(num_samples)

            # Do backprop and optimizer step
            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward(retain_graph=True)
                optimizer.step()
            if it == int(max_iter * freeze_ratio / 2):
                model.eval() # 평가 모드로 전환
                with torch.no_grad():
                    # T-product-t 분포의 df를 추정
                    # 1. q0의 mean, scale을 이용하여 global mean 계산
                    # 2. 각 컴포넌트에 대해 tail index 추정
                    # 3. df가 30보다 작으면 T-product-t 분포로 교체
                    # 4. model.q0.components[t]에 새 컴포넌트를 할당
                    
                    global_mean = compute_global_mean(model.q0)

                    for t in range(model.q0.T):
                        dfs = estimate_tail_indices_t(mean = model.q0.components[t].mean, 
                                                            scale = torch.exp(model.q0.components[t].log_scale), 
                                                            df = 2,
                                                            global_mean = global_mean, 
                                                            target = model.p, 
                                                            num_samples=3000,
                                                            k = 2990,
                                                            model = model)
                        # torch Tensor로 변환 & device 맞추기
                        print(dfs)
                        if any(1 < df < 30 for df in dfs):
                            device = model.q0.components[t].mean.device
                            df_tensor = torch.tensor(dfs, dtype=torch.float32, device=device)
                            # 30을 넘으면 30으로 제한
                            df_tensor = torch.clamp(df_tensor, max=30.0, min=1.0)

                            # 기존 컴포넌트의 mean, scale 꺼내기
                            old_comp = model.q0.components[t]
                            mean  = old_comp.mean.clone().detach().to(device)
                            scale = torch.exp(old_comp.log_scale.clone()).detach().to(device)

                            # 새 T-product-t 분포로 교체
                            new_comp = TProductDistribution(
                                shape = old_comp.shape,
                                mean  = mean,
                                scale = scale,
                                df    = df_tensor
                            )
                            new_comp.to(device)
                            model.q0.components[t] = new_comp

                model.train() # back to training mode
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            
            if it == int(max_iter * freeze_ratio):
                if freeze_flow:
                    for param in model.flows.parameters():
                        param.requires_grad = True
            # Log loss
            loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
            # Plot learned distribution
            if (it + 1) % show_iter == 0:
                if dimension==2: 
                    model.eval()
                    with torch.no_grad():
                        plot_samples(model, num_samples=1000, two_d=True, vmin=vmin, vmax=vmax, contour=True)
                    model.train()
                else:
                    model.eval()  # 평가 모드 전환
                    with torch.no_grad():
                        plot_samples(model, num_samples=1000, two_d=False, vmin=vmin, vmax=vmax)
                    model.train()  # 다시 학습 모드로 전환
    
    elif isinstance(model.q0, (DirichletProcessGaussianMixture, DPGM)):
        print("Training Dirichlet Process Gaussian Mixture Base Model")
        for it in tqdm(range(max_iter)):
            optimizer.zero_grad()
            
            loss = model.reverse_kld(num_samples)

            
                # Do backprop and optimizer step
            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward(retain_graph=True)
                optimizer.step()
                if it < int(max_iter * freeze_ratio /2):
                    model.q0.update_beta()
            if it == int(max_iter * freeze_ratio):
                if freeze_flow:
                    for param in model.flows.parameters():
                        param.requires_grad = True                
            if it == int(max_iter * freeze_ratio /2):
                model.q0.log_eta.requires_grad = False
                if isinstance(model.q0, DirichletProcessGaussianMixture):
                    new_base = DirichletProcessGaussianMixture(shape = model.q0.shape, T=model.q0.T, train_eta=False)
                    
                elif isinstance(model.q0, DPGM):
                    new_base = DPGM(shape = model.q0.shape, T=model.q0.T, train_eta=False)

                new_base.log_eta.data = model.q0.log_eta.data.clone() 
                new_base.means.data = model.q0.means.data.clone()  
                new_base.log_scale.data = model.q0.log_scale.data.clone()
                model.q0 = new_base

                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            
            # Log loss
            loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
            # Plot learned distribution
            if (it + 1) % show_iter == 0:
                if dimension==2: 
                    model.eval()
                    with torch.no_grad():
                        plot_samples(model, num_samples=1000, two_d=True, vmin=vmin, vmax=vmax, contour=True)
                    model.train()
                else:
                    model.eval()  # 평가 모드 전환
                    with torch.no_grad():
                        plot_samples(model, num_samples=1000, two_d=False, vmin=vmin, vmax=vmax)
                    model.train()  # 다시 학습 모드로 전환

    else: #일반적인 경우
        for it in tqdm(range(max_iter)):
            optimizer.zero_grad()
            loss = model.reverse_kld(num_samples)
                # Do backprop and optimizer step
            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward()
                optimizer.step()
            # Log loss
            loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())

            # Plot learned distribution
                # Plot learned distribution
            if (it + 1) % show_iter == 0:
                
                if dimension==2:
                    grid_size = 200
                    xx, yy = torch.meshgrid(torch.linspace(-20, 20, grid_size), torch.linspace(-20, 20, grid_size))
                    zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
                    
                    model.eval()
                    log_prob = model.log_prob(zz)
                    prob = torch.exp(log_prob.to('cpu').view(*xx.shape))
                    prob[torch.isnan(prob)] = 0
                    model.train()

                    plt.figure(figsize=(5, 5))
                    plt.pcolormesh(xx, yy, prob.data.numpy(), cmap='coolwarm')
                    plt.gca().set_aspect('equal', 'box')
                    plt.show()

                else:
                    model.eval()  # 평가 모드 전환
                    # 샘플링
                    samples, _ = model.sample(num_samples=1000)
                    x1 = samples[:, 0].detach().cpu().numpy()

                    # 히스토그램 그리기
                    plt.figure(figsize=(6, 4))
                    plt.hist(x1, bins=50, density=True, alpha=0.7, edgecolor='black')
                    plt.title(f'Iteration {it+1}: Marginal Distribution of x₁')
                    plt.xlabel('x₁')
                    plt.ylabel('Density')
                    plt.grid(True)
                    plt.show()
                    print(loss.to('cpu').data.numpy())
                    model.train()  # 다시 학습 모드로 전환'
                    
    if len(alpha_hist) > 0:
        # alpha_hist는 '리스트 of (d,)-array'
        # 이를 2D NumPy 배열로 변환 => shape=(num_iters, d)
        alpha_hist_arr = np.stack(alpha_hist, axis=0)
    
        plt.figure(figsize=(6, 6))
        # alpha_hist_arr.shape = (num_iters, d)
        # 각 차원마다 선 그래프
        for dim_idx in range(alpha_hist_arr.shape[1]):
            plt.plot(alpha_hist_arr[:, dim_idx], label=f'alpha dim {dim_idx+1}')
    
        plt.title("Alpha values per iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Alpha")
        plt.legend()
        plt.show()
    else:
        print("alpha_hist is empty, no data to plot.")
        
    # Plot loss
    plt.figure(figsize=(6, 6))
    plt.plot(loss_hist, label='loss')
    plt.legend()
    plt.show()