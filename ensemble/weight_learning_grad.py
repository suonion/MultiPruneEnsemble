# -*- coding: utf-8 -*-
# 基于梯度的全局权重学习（对齐 GradTree 的温度化 softmax、可选DRO重加权、早停/降学习率）
# 用途：在验证集上学一组对“剪枝树/方法”的全局权重 w，测试集用该 w 融合概率

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np

try:
    import torch
    TORCH_OK = True
except Exception:
    TORCH_OK = False


# ========== 配置 ==========
@dataclass
class GDWConfig:
    iters: int = 800                 # 训练迭代步数
    lr: float = 0.05                 # 学习率（Adam）
    weight_decay: float = 0.0        # L2 正则（作用在 theta）
    temperature: float = 1.0         # softmax 温度（越小越尖锐）
    print_every: int = 200           # 日志打印间隔
    seed: int = 42                   # 随机种子
    # 正则/稳训选项
    l1_on_w: float = 0.0             # 稀疏化：|w|_1 正则（鼓励少数方法占权）
    ent_min: float = 0.0             # 反熵正则：-ent(w)，>0 时鼓励更尖锐
    # DRO式重加权（对 per-sample CE 做 reweight）
    dro_temperature: Optional[float] = None  # e.g., 0.5 或 1.0；None=关闭DRO重加权
    dro_cap: float = 5.0                      # 对单样本 CE 的裁剪上限（避免过激权重）
    # 早停/降学习率
    early_stop_patience: int = 100    # 连续多少步无提升 early stop；0=关闭
    plateau_patience: int = 40        # 无提升多少步降低学习率；0=关闭
    plateau_factor: float = 0.2       # 降学习率系数（乘法）
    min_lr: float = 1e-4              # 学习率下限


# ========== 小工具 ==========
def _stack_probs(proba_list: List[np.ndarray]) -> np.ndarray:
    """proba_list: [ (N,C), ... ] -> (M,N,C)"""
    return np.stack(proba_list, axis=0)

def fuse_with_weights(proba_list: List[np.ndarray], w: np.ndarray) -> np.ndarray:
    """用学到的权重 w 融合概率；proba_list: [ (N,C), ... ], w: (M,)"""
    P = _stack_probs(proba_list)  # (M,N,C)
    w = np.asarray(w, dtype=float)
    w = w / (np.sum(w) + 1e-12)
    return np.tensordot(w, P, axes=(0, 0))  # (N,C)


# ========== Learner 主体 ==========
class GlobalWeightLearner:
    """
    在验证集上学习 “每个方法/剪枝树” 的全局权重 w（softmax(theta/T)）。
    训练目标：加权后概率与 y_val 的交叉熵（可选 DRO 重加权 + 正则）。
    """

    def __init__(self, cfg: GDWConfig = GDWConfig()):
        assert TORCH_OK, "需要 PyTorch：pip install torch"
        self.cfg = cfg
        self.theta_: Optional[torch.Tensor] = None  # 可训练参数（M,）
        self.w_: Optional[np.ndarray] = None        # 学到的权重（M,）
        self.history_: Dict[str, list] = {"loss": [], "lr": []}

    @staticmethod
    def _softmax_temp(x: torch.Tensor, T: float) -> torch.Tensor:
        return torch.softmax(x / T, dim=0)

    @staticmethod
    def _ce_loss(proba: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # proba: (N,C) 已是概率；转为 logits 的 CE 近似：加 epsilon 稳定
        eps = 1e-12
        proba = torch.clamp(proba, eps, 1 - eps)
        logp = torch.log(proba)
        return torch.nn.functional.nll_loss(logp, y, reduction="mean")

    def _dro_reweight(self, ce_each: torch.Tensor) -> torch.Tensor:
        """
        简洁 DRO 风格：对单样本 CE 做温度裁剪 + 指数放大，再平均
        ce_each: (N,) 未加权的 per-sample CE
        """
        T = self.cfg.dro_temperature
        if T is None:
            return ce_each.mean()
        capped = torch.clamp(ce_each, max=self.cfg.dro_cap)
        weights = torch.exp(capped / (T + 1e-12)).detach()  # 权重不回传梯度
        weights = weights / (weights.mean() + 1e-12)
        return (ce_each * weights).mean()

    def fit(self, proba_list_val: List[np.ndarray], y_val: np.ndarray) -> np.ndarray:
        """
        在验证集上训练，返回学到的 w（numpy，(M,)）
        """
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        P = _stack_probs(proba_list_val)      # (M,N,C)
        M, N, C = P.shape
        Pv = torch.tensor(P, dtype=torch.float32)  # (M,N,C)
        yv = torch.tensor(y_val, dtype=torch.long)

        theta = torch.zeros(M, requires_grad=True)
        opt = torch.optim.Adam([theta], lr=cfg.lr, weight_decay=cfg.weight_decay)

        best_loss = float("inf")
        best_w = None
        bad = 0  # 无提升计数

        for it in range(cfg.iters):
            w = self._softmax_temp(theta, cfg.temperature)      # (M,)
            proba = (w[:, None, None] * Pv).sum(0)              # (N,C)

            # ---- 基础 CE（逐样本） ----
            eps = 1e-12
            proba_clamped = torch.clamp(proba, eps, 1 - eps)
            logp = torch.log(proba_clamped)
            ce_each = torch.nn.functional.nll_loss(logp, yv, reduction="none")  # (N,)

            # ---- DRO 重加权（可选）----
            loss = self._dro_reweight(ce_each)

            # ---- 正则：L1(w) & 反熵（让 w 更尖锐，可选）----
            if cfg.l1_on_w > 0.0:
                loss = loss + cfg.l1_on_w * torch.sum(torch.abs(w))
            if cfg.ent_min > 0.0:
                ent = -torch.sum(w * torch.log(torch.clamp(w, 1e-12, 1.0)))
                loss = loss + cfg.ent_min * (-ent)  # -ent 越大越尖锐

            opt.zero_grad()
            loss.backward()
            opt.step()

            # 记录
            self.history_["loss"].append(float(loss.item()))
            self.history_["lr"].append(opt.param_groups[0]["lr"])

            # 早停/保存最优
            cur = float(loss.item())
            if cur < best_loss - 1e-6:
                best_loss = cur
                best_w = self._softmax_temp(theta.detach(), cfg.temperature).cpu().numpy()
                bad = 0
            else:
                bad += 1
                # plateau 降学习率
                if cfg.plateau_patience > 0 and bad % cfg.plateau_patience == 0:
                    new_lr = max(cfg.min_lr, opt.param_groups[0]["lr"] * cfg.plateau_factor)
                    for g in opt.param_groups:
                        g["lr"] = new_lr
                # early stop
                if cfg.early_stop_patience > 0 and bad >= cfg.early_stop_patience:
                    if (it + 1) % cfg.print_every != 0:
                        print(f"[GD-Global] early stop at iter {it+1}, best CE={best_loss:.6f}")
                    break

            if (it + 1) % cfg.print_every == 0:
                print(f"[GD-Global] iter {it+1}/{cfg.iters}  CE={cur:.6f}  best={best_loss:.6f}")

        # 收尾
        if best_w is None:
            best_w = self._softmax_temp(theta.detach(), cfg.temperature).cpu().numpy()

        self.theta_ = theta.detach()
        self.w_ = best_w
        return best_w

    # 便捷接口：直接融合测试集概率
    def fuse(self, proba_list_test: List[np.ndarray]) -> np.ndarray:
        assert self.w_ is not None, "请先调用 fit 学得权重 w_"
        return fuse_with_weights(proba_list_test, self.w_)

    def get_weights(self) -> Optional[np.ndarray]:
        return self.w_

    def get_history(self) -> Dict[str, List[float]]:
        return self.history_
